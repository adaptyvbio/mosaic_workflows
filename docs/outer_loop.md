### Mosaic outer loop: implement once, think never

This guide specifies a tiny, Mosaic-native outer loop that runs multiple workflows (parents) and optional redesign workflows (children) while keeping all optimization logic inside standard Mosaic workflow dicts. The goal is minimal mental overhead and perfect alignment with existing `mosaic` and `mosaic_workflows` patterns.

---

### Core idea (one small function)

Add a single function that repeatedly calls `run_workflow` on workflows produced by small builder callables. Optional children (e.g., MPNN redesigns) are just more workflows spawned from the parent result.

```python
# src/mosaic_workflows/design.py
def run_many(
  *,
  specs,                  # list[dict], e.g. [{"binder_len": 60, "seed": 7}, ...]
  build,                  # (spec: dict) -> workflow: dict  (standard Mosaic workflow dict)
  spawn=None,             # optional: (spec, parent_result) -> list[tuple[child_spec, child_build]]
  emit=None,              # optional: (kind: "parent"|"child", row: dict) -> None  (save/log)
  stop=None,              # optional: (rows_so_far: list[dict]) -> bool
  resume: bool = True,    # simple JSON state for resume
) -> dict
```

- Parent = design trajectory for a spec (normal Mosaic workflow).
- Child = any redesign/evaluation workflow spawned from a parent (often predict-only folding workflow).
- `emit` is IO-only (CSV/JSONL/PDB/CIF, relax/DSSP/SC if desired).
- `stop` lets you early-exit (e.g., when N accepted designs accumulated).
- `resume` persists a tiny JSON with the current index and accumulated rows.

---

### Where logic lives (Mosaic patterns)

- Inside workflows (phases):
  - analyzers: compute metrics from aux (use `flatten_aux`).
  - validators: thresholds with `threshold_filter` over those metrics.
  - schedules/transforms: step-time behavior (temperature, masks, gradient shaping).
- In the outer loop:
  - coordinate multiple workflows and optional children.
  - IO-only persistence in `emit` (write rows/artifacts, optional non-JAX postprocessing).

This keeps “what to optimize” and “how to step” inside workflows, not the outer loop.

---

### Step 1 — Implement `run_many`

```python
import json, os
from typing import Any, Dict, List
from .design import run_workflow
from .analyzers import flatten_aux

def _state_path(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "state.json")

def _load_state(out_dir: str) -> Dict[str, Any]:
    try:
        with open(_state_path(out_dir), "r") as f:
            return json.load(f)
    except Exception:
        return {"index": 0, "rows": []}

def _save_state(out_dir: str, st: Dict[str, Any]) -> None:
    with open(_state_path(out_dir), "w") as f:
        json.dump(st, f)

def _build_row(kind: str, spec: dict, res: dict) -> dict:
    traj = res.get("trajectory") or []
    aux_last = (traj[-1].get("aux") if traj else {}) if traj else {}
    return {
        "kind": kind,
        "spec": spec,
        "best_sequence": res.get("best_sequence"),
        "metrics": flatten_aux(aux_last),
    }

def run_many(*, specs, build, spawn=None, emit=None, stop=None, resume: bool = True, out_dir: str = ".") -> dict:
    state = _load_state(out_dir) if resume else {"index": 0, "rows": []}
    rows: List[dict] = state.get("rows", [])
    idx = int(state.get("index", 0))

    for i in range(idx, len(specs)):
        spec = specs[i]

        # Parent workflow
        wf = build(spec)
        res = run_workflow(wf)
        row = _build_row("parent", spec, res)
        rows.append(row)
        if emit: emit("parent", row)

        # Children (optional)
        if spawn:
            for child_spec, child_build in (spawn(spec, res) or []):
                child_wf = child_build(child_spec)
                child_res = run_workflow(child_wf)
                child_row = _build_row("child", child_spec, child_res)
                child_row["parent_spec"] = spec
                rows.append(child_row)
                if emit: emit("child", child_row)

        # Early stop
        if stop and stop(rows):
            state = {"index": i + 1, "rows": rows}
            if resume: _save_state(out_dir, state)
            return {"rows": rows, "state": state, "dir": out_dir}

        # Persist progress
        state = {"index": i + 1, "rows": rows}
        if resume: _save_state(out_dir, state)

    return {"rows": rows, "state": state, "dir": out_dir}
```

Key point: the outer loop contains zero optimization logic. It only calls `run_workflow` and emits rows.

---

### Step 2 — Use analyzers + validators inside workflows

- Analyzers should produce metrics (flattened as needed) from aux or from structures, etc.
- Validators should express acceptance thresholds with `threshold_filter`.

```python
from mosaic_workflows.analyzers import flatten_aux
from mosaic_workflows.validators import threshold_filter

phase = {
  "name": "anneal",
  "build_loss": build_loss,
  "optimizer": sgd_logits,
  "steps": 100,
  "schedule": sched_anneal,
  "transforms": {...},
  "analyzers": [
    lambda aux: {"metrics": flatten_aux(aux)},
  ],
  "validators": [
    threshold_filter({"metrics.struct.PLDDTLoss/value": {"min": 70.0}}),
  ],
  "analyze_every": 10,
}
```

This keeps thresholds and metrics generation inside the workflow phases, not in the outer loop.

---

### Step 3 — Parent builder (small closure)

```python
def build_parent(spec):
  wf = make_workflow(
    binder_len=spec["binder_len"],
    motif_positions=...,               # domain inputs
    predict_fn=predict_fn,             # capture heavy object once
  )
  wf["seed"] = spec["seed"]
  return wf
```

---

### Step 4 — Children as workflows (e.g., MPNN redesign)

Children are simply more workflows:

```python
def spawn_mpnn(spec, parent_result):
  seqs = generate_mpnn_sequences_from_parent(parent_result, chain="B", num=32)
  children = []
  for i, seq in enumerate(seqs):
    child_spec = {"binder_len": len(seq), "seed": spec["seed"], "sequence": seq, "idx": i}
    def build_child(s=child_spec, seq=seq):
      # Predict-only workflow: single phase, loss=0, predictor writes outputs to aux
      return make_predict_only_workflow(sequence=seq, predict_fn=predict_fn)
    children.append((child_spec, build_child))
  return children
```

This keeps redesigns fully aligned with Mosaic—each redesign is a normal workflow.

---

### Step 5 — Emitting rows and artifacts (IO only)

Implement a single `emit(kind, row)` function to handle ALL persistence:

```python
def emit_bindcraft(kind, row):
  # 1) Save basic artifacts (best_sequence.txt, best_x.npy if you add it to row)
  # 2) Optional: call OpenMM/DSSP/SC here using paths produced by predict-only workflows
  # 3) Append row to CSV/JSONL in your chosen schema
  pass
```

All non-JAX steps stay in emit (pure IO), keeping workflows clean.

---

### Step 6 — Stop condition

```python
stop=lambda rows: sum(int(r.get("accepted", 0)) for r in rows if r.get("kind")=="parent") >= target_final_designs
```

---

### Step 7 — Put it together

```python
specs = [{"binder_len": 60, "seed": s} for s in range(100)]

out = run_many(
  specs=specs,
  build=build_parent,
  spawn=spawn_mpnn,            # optional; omit if no redesign
  emit=emit_bindcraft,         # IO writer
  stop=lambda rows: sum(int(r.get("accepted", 0)) for r in rows if r["kind"]=="parent") >= 50,
  resume=True,
  out_dir="./out/exp1",
)
```

---

### Optional utilities (nice-to-have)

- `make_predict_only_workflow(sequence|probs, predict_fn) -> workflow`:
  - single-phase workflow returning zero loss; predictor outputs live in aux; analyzers summarize pLDDT/iPTM/PAE; validators can gate acceptance.
- Tiny CSV/paths helpers in an `emit` module for consistent naming and schemas.

---

### Test checklist

- `run_many` runs N specs, resumes from mid-index, calls `emit` in “parent then child” order.
- Parent workflows: analyzers produce expected metrics; validators annotate pass/fail via metrics.
- Child predict-only workflows: aux contains predictor outputs; analyzers flatten them.
- `emit`: writes expected files and appends expected CSV rows.

---

### Why this works

- One tiny outer function; zero new paradigms.
- Every trajectory is still a standard Mosaic workflow (dict + small callables).
- Children are just more workflows—no special redesign APIs.
- Thresholding and metrics live inside workflows via validators and analyzers (Mosaic style).
- All persistence and non-JAX postprocessing are in one place (`emit`).


