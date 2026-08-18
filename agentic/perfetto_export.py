#!/usr/bin/env python3
"""Export slime(:8080) proxy + kuberl/harbor(:8085) rollout data to Perfetto.

Emits a Chrome / Perfetto "Trace Event Format" JSON that loads directly at
https://ui.perfetto.dev (Open trace file).

Layout
------
Each trial/session becomes one Perfetto *process*; training is its own process.
Within a process, tracks (threads) are:

  * "kuberl phases" : env setup / agent setup / agent exec / verification
                      (parsed from the trial server.log)
  * "llm turns"     : one slice per turn, containing nested GPU-side sub-slices
                      queue / inference / tool_parse, and — between turns —
                      "tool exec" slices (the environment running the tool).

Inference vs tool exec:
  * inference  = intra-turn worker_inference_started..completed (GPU generation)
  * tool exec  = the gap between turn i response and turn i+1 request, i.e. the
                 environment executing the tool and producing the observation.

Usage
-----
  python perfetto_export.py [--slime URL] [--harbor URL] [-o out.json]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import types as _types

sys.modules.setdefault("torch", _types.SimpleNamespace(load=None))
import live_timeline as lt  # noqa: E402  (reuse HTTP + trial fetch + matching)


def _us(t):
    """Epoch seconds -> integer microseconds (Perfetto ts unit)."""
    return int(round(float(t) * 1_000_000))


class TraceBuilder:
    def __init__(self):
        self.events = []
        self._pid = 0

    def new_process(self, name):
        self._pid += 1
        pid = self._pid
        self.events.append({"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}})
        return pid

    def thread(self, pid, tid, name):
        self.events.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": name}})

    def slice(self, pid, tid, name, start, end, args=None):
        if not (lt._valid(start) and lt._valid(end)) or end < start:
            return
        self.events.append(
            {
                "ph": "X",
                "pid": pid,
                "tid": tid,
                "name": name,
                "ts": _us(start),
                "dur": _us(end) - _us(start),
                "args": args or {},
            }
        )


# --------------------------------------------------------------------------- #
def _fetch_sessions(base):
    try:
        listing = lt._get(f"{base}/sessions")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] /sessions failed: {exc}", file=sys.stderr)
        return {}
    sids = [s["session_id"] for s in listing.get("sessions", []) if not str(s["session_id"]).startswith("timing_")]

    def _one(sid):
        try:
            return sid, lt._get(f"{base}/sessions/{sid}")
        except Exception:  # noqa: BLE001
            return sid, None

    out = {}
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for sid, detail in pool.map(_one, sids):
            if detail:
                out[sid] = detail
    print(f"[info] fetched {len(out)} sessions", file=sys.stderr)
    return out


def _turn_span(turn):
    tm = turn.get("timing") or {}
    return tm.get("request_received_at"), tm.get("response_received_at")


def _add_phase_track(tb, pid, phases, finished_at):
    if not phases:
        return
    tb.thread(pid, 1, "kuberl phases")
    specs = [
        ("env setup", "env_setup_start", "env_setup_end"),
        ("agent setup", "agent_setup_start", "agent_setup_end"),
        ("agent exec", "agent_exec_start", "agent_exec_end"),
        ("verification", "verify_start", "verify_end"),
    ]
    for label, k0, k1 in specs:
        start, end = phases.get(k0), phases.get(k1)
        if label == "agent exec" and lt._valid(start) and not lt._valid(end):
            end = finished_at  # timed-out trials: synthesize the end
        tb.slice(pid, 1, label, start, end)


def _add_turns_track(tb, pid, turns):
    tb.thread(pid, 2, "llm turns")
    tool_outputs = lt._tool_outputs_by_call_id(turns)
    prev_resp = None
    prev_tool = None
    for i, turn in enumerate(turns):
        req, resp = _turn_span(turn)
        if not (lt._valid(req) and lt._valid(resp)):
            continue
        tm = turn.get("timing") or {}

        # tool exec = gap between previous turn's response and this request
        if prev_resp is not None and req > prev_resp:
            tb.slice(
                pid, 2, f"🔧 tool exec: {prev_tool or '?'}", prev_resp, req, {"kind": "tool_exec", "tool": prev_tool}
            )

        tool = tool_args = tool_output = None
        tcs = turn.get("tool_calls") or []
        if tcs and isinstance(tcs[0], dict):
            fn = tcs[0].get("function") or {}
            tool = fn.get("name")
            tool_args = lt._clip(lt._pretty_json(fn.get("arguments")), 2000)
            tool_output = lt._clip(tool_outputs.get(tcs[0].get("id"), ""), 4000)

        name = f"turn {i}" + (f" · {tool}" if tool else "")
        tb.slice(
            pid,
            2,
            name,
            req,
            resp,
            {
                "kind": "turn",
                "turn": i,
                "finish_reason": turn.get("finish_reason"),
                "tool": tool,
                "parameters": tool_args,
                "observation": tool_output,
                "gpu_id": turn.get("gpu_id"),
                "worker_id": turn.get("worker_id"),
                "node_id": turn.get("node_id"),
                "relay_round_trip_ms": tm.get("relay_round_trip_ms"),
                "completion_chars": len(turn.get("completion_text") or ""),
            },
        )
        # nested GPU-side sub-slices (sequential inside the turn)
        tb.slice(pid, 2, "queue", req, tm.get("worker_received_at"), {"kind": "queue"})
        tb.slice(
            pid,
            2,
            "inference",
            tm.get("worker_inference_started_at"),
            tm.get("worker_inference_completed_at"),
            {"kind": "inference"},
        )
        tb.slice(
            pid,
            2,
            "tool_parse",
            tm.get("worker_inference_completed_at"),
            tm.get("worker_tool_parse_completed_at"),
            {"kind": "tool_parse"},
        )

        prev_resp, prev_tool = resp, tool


def _add_training(tb, base):
    try:
        data = lt._get(f"{base}/training/timing")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] training/timing failed: {exc}", file=sys.stderr)
        return
    pid = tb.new_process("training")
    tb.thread(pid, 1, "steps")
    for t in data.get("timings", []):
        gs, ep = t.get("global_step"), t.get("epoch")
        tb.slice(
            pid,
            1,
            f"step {ep}/{gs}",
            t.get("step_start"),
            t.get("step_end"),
            {k: v for k, v in (t.get("phase_durations") or {}).items() if isinstance(v, (int, float))},
        )
        tb.slice(pid, 1, "inference", t.get("inference_start"), t.get("inference_end"))
        tb.slice(pid, 1, "weight_sync", t.get("weight_sync_start"), t.get("weight_sync_end"))
        tb.slice(pid, 1, "training", t.get("training_start"), t.get("training_end"))
        tb.slice(pid, 1, "update_actor", t.get("update_actor_start"), t.get("update_actor_end"))


def build_trace(slime, harbor, trial_filter=None):
    tb = TraceBuilder()

    trials = lt.fetch_trial_metadata(harbor)
    lt.attach_trial_phases(harbor, trials)
    sessions = _fetch_sessions(slime)
    if trial_filter:
        sessions = {k: v for k, v in sessions.items() if trial_filter in k}
        print(f"[info] filtered to {len(sessions)} sessions matching '{trial_filter}'", file=sys.stderr)

    # link kuberl trials <-> proxy sessions (exact task_id, then time overlap)
    agents = []
    for sid, detail in sessions.items():
        times = [_turn_span(t)[0] for t in detail.get("turns", [])]
        times = [x for x in times if lt._valid(x)]
        agents.append(
            {
                "session_id": sid,
                "start": min(times) if times else 0,
                "end": max(times) if times else 0,
                "detail": detail,
            }
        )
    lt.match_trials_to_agents(agents, trials)

    merged = 0
    for agent in agents:
        detail = agent["detail"]
        trial = agent.get("trial")
        turns = detail.get("turns") or []
        label = agent["session_id"]
        if trial:
            merged += 1
            trial["_matched"] = True
            reward, status = trial.get("reward"), trial.get("status")
            label = f"{label}  [r={reward} {status}]"
        pid = tb.new_process(label)
        if trial:
            _add_phase_track(tb, pid, trial.get("phases"), trial.get("finished_at"))
        _add_turns_track(tb, pid, turns)

    # kuberl trials with no matching proxy session -> phases-only process
    for trial in trials:
        if trial.get("_matched"):
            continue
        name = trial.get("trial_name") or trial.get("run_id") or trial.get("task_id")
        if trial_filter and trial_filter not in (trial.get("task_id") or "") and trial_filter not in (name or ""):
            continue
        pid = tb.new_process(f"{name}  [r={trial.get('reward')} {trial.get('status')}]")
        _add_phase_track(tb, pid, trial.get("phases"), trial.get("finished_at"))

    if not trial_filter:
        _add_training(tb, slime)

    # Normalize timestamps so the trace starts at 0 (avoids a huge absolute
    # epoch offset and makes the initial Perfetto view land on the data).
    slices = [e for e in tb.events if e["ph"] == "X"]
    if slices:
        t0 = min(e["ts"] for e in slices)
        for e in slices:
            e["ts"] -= t0
        span_s = (max(e["ts"] + e["dur"] for e in slices)) / 1e6
        print(f"[info] normalized to t0=0; trace span {span_s:.1f}s", file=sys.stderr)

    print(f"[info] linked {merged} merged trials; {len(tb.events)} trace events", file=sys.stderr)
    return {"traceEvents": tb.events, "displayTimeUnit": "ms"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slime", default="http://localhost:8080")
    ap.add_argument("--harbor", default="http://localhost:8085")
    ap.add_argument(
        "-o", "--out", default=str(Path(__file__).resolve().parent / "live_timeline_out" / "rollout.perfetto.json")
    )
    ap.add_argument(
        "--trial",
        default=None,
        help="only export sessions whose id contains this substring "
        "(e.g. django__django-10880); yields a small, readable trace",
    )
    args = ap.parse_args()

    trace = build_trace(args.slime, args.harbor, trial_filter=args.trial)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(trace, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"perfetto trace: {out}")
    print("open https://ui.perfetto.dev  ->  Open trace file  ->  select the JSON")


if __name__ == "__main__":
    main()
