# Agentic rollout timeline & Perfetto export

Two tools to visualize an **agentic disaggregated rollout** by joining data from
the two live servers used during training:

- **proxy server** (slime relay, default `:8080`) — training-step timing and
  per-turn LLM rollout timing (`/training/timing`, `/sessions`, `/sessions/{id}`)
- **kuberl / harbor server** (default `:8085`) — trial metadata, rewards, and
  per-trial phase timing parsed from each trial's `server.log`
  (`/api/v1/trials`, `/api/jobs/.../trials/.../files/server.log`)

A kuberl **trial** and a proxy **session** that share the same `task_id`
(e.g. `django__django-10880-gpbt4keQSEX8uYCursT8ax`) are **merged into one
instance**, so you see the whole lifecycle — container/env setup, agent setup,
the LLM turns (tool calls + observations), verification — on a single row.

## Files

| file | purpose |
|------|---------|
| `live_timeline.py` | Fetch + link both servers, emit a self-contained interactive HTML timeline and serve it. |
| `perfetto_export.py` | Same data, exported to a [Perfetto](https://ui.perfetto.dev) trace (Chrome Trace Event JSON). |
| `trace_timeline_viewer.py` | Dependency of `live_timeline.py` (row builder + HTML template). |

Generated output goes to `live_timeline_out/` (git-ignored).

## Requirements

- Python 3 (standard library only; `torch` is stubbed out — not required).
- The proxy (`:8080`) and kuberl/harbor (`:8085`) servers reachable from where
  you run the scripts.

## HTML timeline — `live_timeline.py`

```bash
python3 live_timeline.py                 # fetch, build, serve on :9999
python3 live_timeline.py --no-serve      # just write the files
# options: --slime URL  --harbor URL  --out DIR  --port N  --max-sessions N
```

Then open the printed URL, e.g.
`http://127.0.0.1:9999/live_rollout.trace_timeline_viewer.html`.

Each merged instance is one row. Click the row label ("instance section") to
expand it into **2 lanes**:

- **lane 0 — kuberl phases:** `env setup · agent setup · agent exec · verification`
- **lane 1 — proxy turns:** one bar per turn, nested inside the `agent exec` phase

Hover / click a turn to see a **tool call → parameters → observation** block plus
per-turn sub-phase timings (`queue_ms`, `inference_ms`, `tool_parse_ms`),
`gpu_id`, `worker_id`, `relay_round_trip_ms`, `finish_reason`.

**Instance statistics** (top panel) show the total instance count and a count per
**return code** (trial `status`, e.g. `completed` / `timeout`). Both respect the
filter box.

## Perfetto export — `perfetto_export.py`

```bash
python3 perfetto_export.py                                   # full run -> rollout.perfetto.json
python3 perfetto_export.py --trial <session_id> -o one.json  # one session (small, readable)
# options: --slime URL  --harbor URL  -o OUT  --trial SUBSTR
```

Open [ui.perfetto.dev](https://ui.perfetto.dev) → **Open trace file** → pick the JSON.

Layout (one Perfetto **process** per trial, plus a `training` process):

- track **`kuberl phases`** — env setup / agent setup / agent exec / verification
- track **`llm turns`** — one slice per turn (`turn N · <tool>`) with nested
  `queue` / `inference` / `tool_parse` sub-slices, and `🔧 tool exec` slices in
  the gaps between turns

Click a **turn's top bar** to read its args (`tool`, `parameters`, `observation`,
`gpu_id`, `worker_id`, …) in the Arguments panel.

### Inference vs tool exec

- **inference** = *inside* a turn: `worker_inference_started_at →
  worker_inference_completed_at` (GPU generation).
- **tool exec** = the *gap between* turn *i*'s response and turn *i+1*'s request —
  the agent/environment running the tool and producing the observation. It is not
  a turn sub-phase.

### Tips

- The full trace spans the whole run (hours) in sparse waves, so it opens fitted
  to the full range where second-scale slices are sub-pixel — **zoom in**
  (scroll to zoom, `W`/`S`, or select a track and press `F`). Timestamps are
  normalized to start at `0`.
- For an immediately-readable view, export a **single session** with `--trial`
  (a full `session_id`), which yields a short (~minutes) trace.
