#!/usr/bin/env python3
"""Build a trace timeline from live slime (:8080) + harbor (:8085) servers.

Fetches training-step timing and rollout session/turn timing from the slime
relay server, plus trial reward/status from the harbor server, converts them
into the trace-event schema understood by ``trace_timeline_viewer`` and reuses
that module's row builder + HTML template to emit a self-contained viewer.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# trace_timeline_viewer imports torch only for loading .pt files; we build the
# cache directly from HTTP data, so stub it out to avoid the heavy dependency.
if "torch" not in sys.modules:
    import types as _types
    sys.modules["torch"] = _types.SimpleNamespace(load=None)
import trace_timeline_viewer as ttv  # noqa: E402


def _get(url: str, timeout: float = 30.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _valid(ts) -> bool:
    return isinstance(ts, (int, float)) and ts > 0


def _msg_text(content) -> str:
    """Flatten a message content (str or list of {type,text}) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict):
                parts.append(str(chunk.get("text", chunk.get("content", ""))))
            else:
                parts.append(str(chunk))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _pretty_json(value) -> str:
    """Pretty-print a JSON-string tool-argument blob; fall back to raw text."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False)
    text = str(value)
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except (ValueError, TypeError):
        return text


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…<+{len(text) - limit} chars>"


def _ms(a, b):
    """Millisecond delta between two epoch timestamps, or None if invalid."""
    if _valid(a) and _valid(b) and b >= a:
        return round((b - a) * 1000, 1)
    return None


def _tool_outputs_by_call_id(turns) -> dict:
    """Tool results come back as role=='tool' messages in *later* turns."""
    out = {}
    for turn in turns:
        for msg in turn.get("request_messages") or []:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid and cid not in out:
                    out[cid] = _msg_text(msg.get("content"))
    return out


def _span(events, name, start, end, span_id, parent, attrs=None):
    """Append a start/end span pair when timestamps are sane and ordered."""
    if not (_valid(start) and _valid(end)) or end < start:
        return False
    events.append(
        {"ts": start, "type": "span_start", "name": name, "span_id": span_id,
         "parent_span_id": parent, "attrs": attrs or {}}
    )
    events.append(
        {"ts": end, "type": "span_end", "name": name, "span_id": span_id,
         "parent_span_id": parent, "attrs": {}}
    )
    return True


# --------------------------------------------------------------------------- #
# 8080: training steps
# --------------------------------------------------------------------------- #
def build_training_samples(base: str):
    try:
        data = _get(f"{base}/training/timing")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] training/timing failed: {exc}", file=sys.stderr)
        return []

    samples = []
    for t in data.get("timings", []):
        gs = t.get("global_step")
        ep = t.get("epoch")
        sid = f"step:{ep}:{gs}"
        ev = []
        _span(ev, f"step {ep}/{gs}", t.get("step_start"), t.get("step_end"), sid, None,
              attrs={k: v for k, v in (t.get("phase_durations") or {}).items()
                     if isinstance(v, (int, float))})
        _span(ev, "inference", t.get("inference_start"), t.get("inference_end"),
              f"{sid}:inf", sid)
        _span(ev, "weight_sync", t.get("weight_sync_start"), t.get("weight_sync_end"),
              f"{sid}:ws", sid)
        _span(ev, "training", t.get("training_start"), t.get("training_end"),
              f"{sid}:train", sid)
        _span(ev, "update_actor", t.get("update_actor_start"), t.get("update_actor_end"),
              f"{sid}:ua", f"{sid}:train")
        if not ev:
            continue
        samples.append({
            "index": f"train {ep}/{gs}",
            "source": "training",
            "status": "step",
            "label": f"step {ep}/{gs}",
            "reward": None,
            "metadata": {},
            "trace": {"events": ev, "trace_id": sid, "attempt": gs or 0},
        })
    return samples


# --------------------------------------------------------------------------- #
# 8080: rollout sessions (per-turn spans + sub-phases)
# --------------------------------------------------------------------------- #
def _session_sample(base: str, sid: str, harbor: dict):
    try:
        detail = _get(f"{base}/sessions/{sid}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] session {sid} failed: {exc}", file=sys.stderr)
        return None

    turns = detail.get("turns") or []
    tool_outputs = _tool_outputs_by_call_id(turns)
    ev = []
    starts, ends = [], []
    for i, turn in enumerate(turns):
        tm = turn.get("timing") or {}
        req = tm.get("request_received_at")
        resp = tm.get("response_received_at")
        if not (_valid(req) and _valid(resp)):
            continue
        starts.append(req)
        ends.append(resp)
        tool = tool_args = tool_output = None
        tcs = turn.get("tool_calls") or []
        if tcs and isinstance(tcs[0], dict):
            fn = tcs[0].get("function") or {}
            tool = fn.get("name")
            tool_args = _clip(_pretty_json(fn.get("arguments")), 800) or None
            tool_output = _clip(tool_outputs.get(tcs[0].get("id"), ""), 1400) or None
        tspan = f"{sid}:t{i}"
        # Sub-phase timings are folded into the tooltip (queue / inference /
        # tool-parse) instead of separate lanes, to keep the expanded row compact.
        attrs = {
            "turn": i,
            "finish_reason": turn.get("finish_reason"),
            "tool": tool,
            "tool_args": tool_args,
            "tool_output": tool_output,
            "gpu_id": turn.get("gpu_id"),
            "worker_id": turn.get("worker_id"),
            "node_id": turn.get("node_id"),
            "relay_round_trip_ms": tm.get("relay_round_trip_ms"),
            "queue_ms": _ms(req, tm.get("worker_received_at")),
            "inference_ms": _ms(tm.get("worker_inference_started_at"),
                                tm.get("worker_inference_completed_at")),
            "tool_parse_ms": _ms(tm.get("worker_inference_completed_at"),
                                 tm.get("worker_tool_parse_completed_at")),
            "completion_chars": len(turn.get("completion_text") or ""),
        }
        name = f"turn {i}" + (f" · {tool}" if tool else "")
        # Parent is the (dangling) session id; merged rows re-parent these under
        # the kuberl 'agent exec' phase. Unmatched rows keep them at lane 0.
        _span(ev, name, req, resp, tspan, f"sess:{sid}", attrs=attrs)

    if not ev:
        return None

    sess_start = min(starts)
    sess_end = max(ends)
    return {
        "index": sid,
        "source": "rollout",
        "status": harbor.get("status") or ("completed" if detail.get("completed") else "running"),
        "label": sid,
        "reward": harbor.get("reward"),
        "metadata": {},
        "trace": {"events": ev, "trace_id": sid, "attempt": 0},
        # private fields used to link kuberl trials to proxy sessions
        "_sid": sid,
        "_start": sess_start,
        "_end": sess_end,
    }


def build_rollout_samples(base: str, harbor_by_task: dict, max_sessions: int | None):
    try:
        listing = _get(f"{base}/sessions")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] /sessions failed: {exc}", file=sys.stderr)
        return []
    sids = [s["session_id"] for s in listing.get("sessions", [])]
    if max_sessions:
        sids = sids[:max_sessions]
    print(f"[info] fetching {len(sids)} sessions ...", file=sys.stderr)
    samples = []
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_session_sample, base, sid, harbor_by_task.get(sid, {})): sid
                for sid in sids}
        for fut in cf.as_completed(futs):
            s = fut.result()
            if s:
                samples.append(s)
    return samples


# --------------------------------------------------------------------------- #
# 8085: kuberl / harbor trials  (ported from kube-rl timeline_viewer.py)
# --------------------------------------------------------------------------- #
def _iso_epoch(value):
    if not value:
        return 0.0
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _parse_trials(items):
    result = []
    for t in items:
        started = _iso_epoch(t.get("started_at"))
        finished = _iso_epoch(t.get("finished_at"))
        dur = t.get("duration") or 0
        if not dur and started and finished:
            dur = finished - started
        result.append({
            "trial_name": t.get("trial_name") or t.get("name") or t.get("run_id", ""),
            "run_id": t.get("run_id") or t.get("name", ""),
            "job_id": t.get("job_id", ""),
            "status": t.get("status", ""),
            "started_at": started,
            "finished_at": finished,
            "duration": dur,
            "reward": t.get("reward"),
            "agent_name": t.get("agent_name", ""),
            "n_input_tokens": t.get("n_input_tokens") or t.get("input_tokens") or 0,
            "n_output_tokens": t.get("n_output_tokens") or t.get("output_tokens") or 0,
            "error_type": t.get("error_type"),
            "task_id": t.get("task_id", ""),
        })
    return result


def fetch_trial_metadata(base: str):
    """Fetch trials from the kube-rl API (/api/v1/trials), else the harbor
    viewer API (/api/jobs -> /api/jobs/{name}/trials, paginated)."""
    base = base.rstrip("/")
    try:
        data = _get(f"{base}/api/v1/trials?limit=1000")
        items = data if isinstance(data, list) else data.get("trials", data.get("items", []))
        trials = _parse_trials(items)
        print(f"[info] fetched {len(trials)} trials from kube-rl {base}", file=sys.stderr)
        return trials
    except Exception:  # noqa: BLE001
        pass
    try:
        jobs = _get(f"{base}/api/jobs?page_size=100").get("items", [])
        all_items = []
        for job in jobs:
            job_name = job.get("name", "")
            page = 1
            while True:
                pd = _get(f"{base}/api/jobs/{job_name}/trials?page={page}&page_size=100")
                all_items.extend(pd.get("items", []))
                if page >= pd.get("total_pages", 1):
                    break
                page += 1
        trials = _parse_trials(all_items)
        print(f"[info] fetched {len(trials)} trials from harbor {base}", file=sys.stderr)
        return trials
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] cannot fetch trials from {base}: {exc}", file=sys.stderr)
        return []


def _parse_phases_from_log(log_text: str):
    """Parse 'Phase: ... starting/completed' lines from a trial server.log."""
    import re as _re
    from datetime import datetime as _dt
    import calendar as _cal

    phases = {}
    patterns = [
        ("env_setup_start", r"Phase: environment setup starting"),
        ("env_setup_end", r"Phase: environment setup completed"),
        ("agent_setup_start", r"Phase: agent setup starting"),
        ("agent_setup_end", r"Phase: agent setup completed"),
        ("agent_exec_start", r"Phase: agent execution starting"),
        ("agent_exec_end", r"Phase: agent execution completed"),
        ("verify_start", r"Phase: verification starting"),
        ("verify_end", r"Phase: verification completed"),
    ]
    ts_re = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    for key, pattern in patterns:
        for line in log_text.split("\n"):
            if _re.search(pattern, line):
                m = _re.match(ts_re, line)
                if m:
                    dt = _dt.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
                    # server logs are UTC -> use timegm, not mktime
                    phases[key] = _cal.timegm(dt.timetuple()) + dt.microsecond / 1e6
                    break
    return phases or None


def fetch_trial_phases(base: str, trial):
    """Fetch and parse a trial's server.log for phase timestamps."""
    base = base.rstrip("/")
    name = trial.get("trial_name") or trial.get("run_id")
    job = trial.get("job_id")
    candidates = []
    if job:
        candidates.append(f"{base}/api/jobs/{job}/trials/{name}/files/server.log")
    candidates.append(f"{base}/api/jobs/local/trials/{name}/files/server.log")
    candidates.append(f"{base}/api/v1/trials/{name}/files/server.log")
    for url in candidates:
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                return _parse_phases_from_log(resp.read().decode(errors="replace"))
        except Exception:  # noqa: BLE001
            continue
    return None


def attach_trial_phases(base: str, trials):
    """Populate trial['phases'] from each trial's server.log (threaded)."""
    def _one(tr):
        tr["phases"] = fetch_trial_phases(base, tr)
        return tr
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_one, trials))
    n = sum(1 for t in trials if t.get("phases"))
    print(f"[info] trial phases parsed from server.log: {n}/{len(trials)}", file=sys.stderr)


def _extract_task_name(name: str) -> str:
    """'django__django-10880-VP7M...' / 'django-django-10914-19541d' -> 'django-10914'."""
    import re
    normalized = (name or "").replace("__", "-")
    m = re.match(
        r"((?:django|astropy|sympy|pytest|sphinx|requests|flask|scikit|matplotlib|numpy|pandas|rb|jsx)"
        r"[-_][\w]+[-_]\d+)", normalized)
    if m:
        return m.group(1)
    parts = normalized.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else normalized


def match_trials_to_agents(agents, trials):
    """Link kuberl trials to proxy sessions.

    Phase 1: exact match trial.task_id == session_id.
    Phase 2: fuzzy match remaining by task name + time overlap.
    Sets agent['trial'] in place.
    """
    if not trials:
        return
    sid_to_idx = {a["session_id"]: i for i, a in enumerate(agents) if a.get("session_id")}
    used, matched = set(), set()
    for trial in trials:
        tid = trial.get("task_id", "")
        if tid and tid in sid_to_idx:
            ai = sid_to_idx[tid]
            agents[ai]["trial"] = trial
            matched.add(ai)
            used.add(id(trial))
    exact = len(matched)

    remaining_a = [i for i in range(len(agents)) if i not in matched]
    remaining_t = [t for t in trials if id(t) not in used]
    if remaining_a and remaining_t:
        a_groups = {}
        for ai in remaining_a:
            a_groups.setdefault(_extract_task_name(agents[ai]["session_id"]), []).append(ai)
        t_groups = {}
        for trial in remaining_t:
            t_groups.setdefault(_extract_task_name(trial.get("run_id", "")), []).append(trial)
        for task, ais in a_groups.items():
            task_trials = t_groups.get(task) or remaining_t
            candidates = []
            for ai in ais:
                a_start, a_end = agents[ai].get("start", 0), agents[ai].get("end", 0)
                if not (a_start and a_end):
                    continue
                for trial in task_trials:
                    ts, te = trial["started_at"], trial["finished_at"]
                    if not (ts and te):
                        continue
                    overlap = max(0, min(a_end, te) - max(a_start, ts))
                    if overlap > 0:
                        bonus = 1000 if (a_start >= ts and a_end <= te) else 0
                        candidates.append((overlap + bonus, ai, trial))
            candidates.sort(key=lambda x: -x[0])
            for _, ai, trial in candidates:
                if ai in matched or id(trial) in used:
                    continue
                agents[ai]["trial"] = trial
                matched.add(ai)
                used.add(id(trial))
    print(f"[info] linked {len(matched)}/{len(agents)} sessions to trials "
          f"({exact} exact by task_id, {len(matched) - exact} fuzzy by overlap)", file=sys.stderr)


# Trial server.log phase spans, in chronological order.
_PHASE_SPANS = [
    ("env setup", "env_setup_start", "env_setup_end"),
    ("agent setup", "agent_setup_start", "agent_setup_end"),
    ("agent exec", "agent_exec_start", "agent_exec_end"),
    ("verification", "verify_start", "verify_end"),
]


def _add_phase_spans(events, phases, prefix, parent, exec_span_id=None):
    """Append kuberl phase spans (from server.log) under `parent`.

    If ``exec_span_id`` is given, the 'agent exec' phase uses that span id so
    proxy turn spans can be re-parented to nest inside it.
    """
    if not phases:
        return
    for label, k0, k1 in _PHASE_SPANS:
        span_id = (exec_span_id if (label == "agent exec" and exec_span_id)
                   else f"{prefix}:{label.replace(' ', '_')}")
        _span(events, f"⚙ {label}", phases.get(k0), phases.get(k1), span_id, parent)


def build_trial_samples(trials):
    """One row per *unmatched* kuberl trial (matched ones are merged into the
    proxy session row by link_sessions_to_trials)."""
    samples = []
    for tr in trials:
        if tr.get("_matched"):
            continue
        start, end = tr.get("started_at"), tr.get("finished_at")
        if not (_valid(start) and _valid(end)) or end < start:
            continue
        name = tr.get("trial_name") or tr.get("run_id") or tr.get("task_id")
        tid = f"trial:{name}"
        ev = []
        _span(ev, name, start, end, tid, None, attrs={
            "reward": tr.get("reward"),
            "status": tr.get("status"),
            "duration_s": tr.get("duration"),
            "agent": tr.get("agent_name"),
            "error_type": tr.get("error_type"),
            "input_tokens": tr.get("n_input_tokens"),
            "output_tokens": tr.get("n_output_tokens"),
            "task_id": tr.get("task_id"),
        })
        _add_phase_spans(ev, tr.get("phases"), tid, tid)
        samples.append({
            "index": name,
            "source": "trial",
            "status": tr.get("status") or "completed",
            "label": name,
            "reward": tr.get("reward"),
            "metadata": {},
            "trace": {"events": ev, "trace_id": name, "attempt": 0},
        })
    print(f"[info] trial rows: {len(samples)}", file=sys.stderr)
    return samples


def link_sessions_to_trials(rollout_samples, trials):
    """Merge each kuberl trial into the proxy session that shares its task_id.

    The proxy session row becomes the single merged trial row: kuberl phases
    (env/agent setup, agent exec, verification) form the outer lane, and the
    proxy turns are re-parented to nest inside the 'agent exec' phase where
    they actually run. Matched trials are flagged so build_trial_samples()
    won't emit a duplicate standalone row.
    """
    agents = [{"session_id": s["_sid"], "start": s.get("_start", 0),
               "end": s.get("_end", 0), "sample": s} for s in rollout_samples]
    match_trials_to_agents(agents, trials)
    merged = 0
    for agent in agents:
        trial = agent.get("trial")
        if not trial:
            continue
        trial["_matched"] = True
        merged += 1
        sample = agent["sample"]
        sid = agent["session_id"]
        events = sample["trace"]["events"]
        # This row now represents the whole trial (proxy turns + kuberl phases).
        sample["source"] = "trial"
        if trial.get("reward") is not None:
            sample["reward"] = trial.get("reward")

        trial_meta = {
            "trial_name": trial.get("trial_name") or trial.get("run_id"),
            "trial_reward": trial.get("reward"),
            "trial_status": trial.get("status"),
            "trial_duration_s": trial.get("duration"),
            "input_tokens": trial.get("n_input_tokens"),
            "output_tokens": trial.get("n_output_tokens"),
        }
        phases = dict(trial.get("phases") or {})
        if not phases:
            continue
        # Timed-out trials have no 'agent execution completed' line; synthesize
        # the end from the trial finish so turns can still nest.
        if _valid(phases.get("agent_exec_start")) and not _valid(phases.get("agent_exec_end")):
            phases["agent_exec_end"] = trial.get("finished_at") or agent.get("end")
        # Two-lane layout: kuberl phases at lane 0 (parent None), proxy turns
        # nested under the 'agent exec' phase at lane 1.
        has_exec = _valid(phases.get("agent_exec_start")) and _valid(phases.get("agent_exec_end"))
        exec_id = f"sess:{sid}:agent_exec" if has_exec else None
        if exec_id:
            for e in events:
                if (e.get("parent_span_id") == f"sess:{sid}"
                        and str(e.get("span_id", "")).startswith(f"{sid}:t")):
                    e["parent_span_id"] = exec_id
        _add_phase_spans(events, phases, f"sess:{sid}:phase", None, exec_span_id=exec_id)
        # Attach trial metadata to the agent-exec span (or env setup as fallback).
        meta_target = exec_id or f"sess:{sid}:phase:env_setup"
        for e in events:
            if e.get("type") == "span_start" and e.get("span_id") == meta_target:
                e["attrs"].update(trial_meta)
                break
    print(f"[info] merged {merged} kuberl trials into proxy session rows", file=sys.stderr)


def patch_tooltip_template():
    """Render tool call / parameters / observation as a dedicated tooltip block."""
    old = (
        "      const otherKeys = Object.keys(attrs).filter(k => !k.startsWith('pd_') && !k.startsWith('timeline_'));\n"
        "      for (const key of otherKeys) {\n"
        "        const value = attrs[key];\n"
        "        lines.push(`${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`);\n"
        "      }\n"
    )
    new = (
        "      const TOOL_KEYS = new Set(['tool', 'tool_args', 'tool_output']);\n"
        "      const otherKeys = Object.keys(attrs).filter(k => !k.startsWith('pd_') && !k.startsWith('timeline_') && !TOOL_KEYS.has(k));\n"
        "      if (attrs.tool || attrs.tool_args || attrs.tool_output) {\n"
        "        lines.push('──── tool call ────');\n"
        "        if (attrs.tool) lines.push(`\\u{1f527} tool: ${attrs.tool}`);\n"
        "        if (attrs.tool_args) { lines.push('\\u{1f4e5} parameters:'); lines.push(attrs.tool_args); }\n"
        "        if (attrs.tool_output) { lines.push('\\u{1f4e4} observation:'); lines.push(attrs.tool_output); }\n"
        "        lines.push('───────────────────');\n"
        "      }\n"
        "      for (const key of otherKeys) {\n"
        "        const value = attrs[key];\n"
        "        lines.push(`${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`);\n"
        "      }\n"
    )
    if old not in ttv.HTML_TEMPLATE:
        print("[warn] tooltip template anchor not found; skipping patch", file=sys.stderr)
        return
    ttv.HTML_TEMPLATE = ttv.HTML_TEMPLATE.replace(old, new)
    # Give the tooltip more room for multi-line params/observations.
    ttv.HTML_TEMPLATE = ttv.HTML_TEMPLATE.replace(
        "      max-width: 460px;\n",
        "      max-width: 640px;\n      max-height: 88vh;\n      overflow: hidden;\n",
    )
    # Position the block by its real measured size so it never clips off-screen,
    # and make a clicked (pinned) block scrollable.
    pos_old = (
        "      tooltip.textContent = itemTooltipLines(row, item).join('\\n');\n"
        "      tooltip.style.left = `${Math.min(window.innerWidth - 480, x + 14)}px`;\n"
        "      tooltip.style.top = `${Math.min(window.innerHeight - 200, y + 14)}px`;\n"
        "      tooltip.classList.add('visible');\n"
    )
    pos_new = (
        "      tooltip.textContent = itemTooltipLines(row, item).join('\\n');\n"
        "      tooltip.classList.add('visible');\n"
        "      const _pinned = !state.hoveredItem && !!state.selectedItem;\n"
        "      tooltip.style.pointerEvents = _pinned ? 'auto' : 'none';\n"
        "      tooltip.style.overflowY = _pinned ? 'auto' : 'hidden';\n"
        "      const _m = 8;\n"
        "      const _w = tooltip.offsetWidth;\n"
        "      const _h = tooltip.offsetHeight;\n"
        "      let _l = x + 14;\n"
        "      if (_l + _w + _m > window.innerWidth) _l = x - 14 - _w;\n"
        "      _l = Math.max(_m, Math.min(_l, window.innerWidth - _w - _m));\n"
        "      let _t = y + 14;\n"
        "      if (_t + _h + _m > window.innerHeight) _t = y - 14 - _h;\n"
        "      _t = Math.max(_m, Math.min(_t, window.innerHeight - _h - _m));\n"
        "      tooltip.style.left = `${_l}px`;\n"
        "      tooltip.style.top = `${_t}px`;\n"
    )
    if pos_old in ttv.HTML_TEMPLATE:
        ttv.HTML_TEMPLATE = ttv.HTML_TEMPLATE.replace(pos_old, pos_new)
    else:
        print("[warn] tooltip position anchor not found; skipping", file=sys.stderr)

    # Instance statistics: show only the total count and the per-return-code
    # (trial status) counts. Redefine updateStats() via an appended script so we
    # don't have to string-match the large original function body.
    stats_override = r"""<script>
updateStats = function () {
  const cursorTime = state.cursorTime ?? state.viewStart;
  const ct = document.getElementById('cursorText');
  if (ct) ct.textContent = `cursor: ${niceDuration(cursorTime - state.globalStart)}`;
  const insts = state.rows.filter(r => r.source !== 'training');
  const counts = new Map();
  for (const r of insts) {
    const code = (r.status ?? 'unknown') || 'unknown';
    counts.set(code, (counts.get(code) || 0) + 1);
  }
  const items = [{ name: 'total', value: insts.length, code: false }].concat(
    Array.from(counts.entries())
      .sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0]))
      .map(([name, value]) => ({ name, value, code: true }))
  );
  document.getElementById('stats').innerHTML = items.map(it =>
    `<div class="stat"><div class="name">${
      it.code ? `<span class="swatch" style="background:${hashColor(it.name, 0.9)}"></span>${escapeHtml(it.name)}` : escapeHtml(it.name)
    }</div><div class="value">${it.value}</div></div>`).join('');
};
</script>
"""
    if "</body>" in ttv.HTML_TEMPLATE:
        ttv.HTML_TEMPLATE = ttv.HTML_TEMPLATE.replace("</body>", stats_override + "</body>", 1)
    else:
        print("[warn] </body> anchor not found; stats override skipped", file=sys.stderr)


# --------------------------------------------------------------------------- #
def build_cache(samples):
    rows = []
    gstart = gend = None
    for idx, sample in enumerate(samples):
        row = ttv._build_items_from_trace(sample, idx)
        if row is None:
            continue
        rows.append(row)
        gstart = row["start"] if gstart is None else min(gstart, row["start"])
        gend = row["end"] if gend is None else max(gend, row["end"])
    return {
        "cache_version": ttv.CACHE_VERSION,
        "pt_path": "live://slime-8080+harbor-8085",
        "generated_at": time.time(),
        "sample_count": len(rows),
        "global_start": ttv._round_float(gstart),
        "global_end": ttv._round_float(gend),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slime", default="http://localhost:8080")
    ap.add_argument("--harbor", default="http://localhost:8085")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "live_timeline_out"))
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()

    trials = fetch_trial_metadata(args.harbor)
    attach_trial_phases(args.harbor, trials)
    harbor_by_task = {t["task_id"]: t for t in trials if t.get("task_id")}

    rollout_samples = build_rollout_samples(args.slime, harbor_by_task, args.max_sessions)
    link_sessions_to_trials(rollout_samples, trials)  # kuberl <-> proxy

    samples = build_training_samples(args.slime)
    samples += build_trial_samples(trials)
    samples += rollout_samples
    print(f"[info] total samples: {len(samples)}", file=sys.stderr)

    cache = build_cache(samples)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = ttv.TimelinePaths(
        pt_path=out_dir / "live_rollout.pt",
        cache_path=out_dir / "live_rollout.trace_timeline_cache.json",
        html_path=out_dir / "live_rollout.trace_timeline_viewer.html",
    )
    # Never clobber a good cache with an empty one (e.g. upstream servers down).
    if cache["sample_count"] == 0 and paths.cache_path.exists():
        print("[error] fetched 0 rows (servers down?); keeping existing cache, "
              "regenerating HTML only", file=sys.stderr)
        patch_tooltip_template()
        ttv.ensure_html(paths)
        if not args.no_serve:
            ttv.serve_directory(out_dir, args.port)
        return
    with paths.cache_path.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=True, separators=(",", ":"))
    patch_tooltip_template()
    ttv.ensure_html(paths)

    print(f"cache: {paths.cache_path}")
    print(f"html:  {paths.html_path}")
    print(f"rows:  {cache['sample_count']}")
    print(f"range: {cache['global_start']} -> {cache['global_end']}")

    if not args.no_serve:
        print(f"open:  http://127.0.0.1:{args.port}/{paths.html_path.name}")
        ttv.serve_directory(out_dir, args.port)


if __name__ == "__main__":
    main()
