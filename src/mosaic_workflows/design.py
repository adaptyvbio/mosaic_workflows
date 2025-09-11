import numpy as np
import jax
from typing import Any, Dict, List


def _default_schedule(global_step: int, phase_step: int) -> dict:
    return {}


def _apply_analyzers(analyzers, aux_or_ctx: dict) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for fn in analyzers or []:
        try:
            m = fn(aux_or_ctx) or {}
            if isinstance(m, dict):
                metrics |= m
        except Exception:
            # analyzers must not crash the run
            pass
    return metrics


def _run_phase(*, phase: dict, x: np.ndarray, key, global_step: int, callbacks):
    name = phase["name"]
    build_loss = phase["build_loss"]
    optimizer = phase["optimizer"]
    steps = int(phase["steps"])
    schedule = phase.get("schedule") or _default_schedule
    transforms = phase.get("transforms") or {}
    analyzers = phase.get("analyzers") or []
    analyze_every = int(phase.get("analyze_every", 0))

    loss_built = build_loss()
    # Support binder_games two-player minmax losses by dispatching to two-player optimizers when present.
    if isinstance(loss_built, dict) and "two_player" in loss_built:
        two_player_loss = loss_built["two_player"]
        # Wrap two-player loss to match optimizer expectation directly
        def loss_function_two_player(x_probs, y_probs, key=None):
            return two_player_loss(x_probs, y_probs, key)
        # prefer provided optimizer (should be a two-player optimizer)
        loss_function = loss_function_two_player
    else:
        loss_function = loss_built

    trajectory: List[Dict[str, Any]] = []

    def trajectory_fn(aux, x_arr):
        nonlocal trajectory
        rec = {"step": len(trajectory), "aux": aux, "x": x_arr}
        if analyze_every and (len(trajectory) % analyze_every == 0):
            rec["metrics"] = _apply_analyzers(analyzers, aux)
        trajectory.append(rec)
        return rec

    x, best_x, _ = optimizer(
        loss_function=loss_function,
        x=x,
        n_steps=steps,
        key=key,
        schedule=schedule,
        transforms=transforms,
        trajectory_fn=trajectory_fn,
        aux_context={"phase_name": name, "global_step": global_step},
    )

    # callbacks at end of phase
    for cb in callbacks or []:
        try:
            cb({"event": "end_phase", "phase": name, "trajectory": trajectory})
        except Exception:
            pass

    return x, best_x, trajectory


def _decode_best_sequence(best_x: np.ndarray) -> str:
    # one-hot decode over 20 AA tokens
    vocab = "ARNDCQEGHILKMFPSTWYV"
    idx = np.argmax(best_x, axis=-1)
    return "".join(vocab[i] for i in idx)


def run_workflow(workflow: dict) -> dict:
    phases = workflow["phases"]
    binder_len = int(workflow["binder_len"])
    seed = int(workflow.get("seed", 0))
    x0 = workflow.get("initial_x")
    callbacks = workflow.get("callbacks") or []

    if x0 is None:
        x0 = np.random.randn(binder_len, 20).astype(np.float32) * 0.1

    key = jax.random.key(seed)
    x = x0
    best_x = x0
    global_step = 0
    all_traj = []

    for phase in phases:
        x, best_x, traj = _run_phase(
            phase=phase,
            x=x,
            key=jax.random.fold_in(key, global_step),
            global_step=global_step,
            callbacks=callbacks,
        )
        global_step += phase["steps"]
        all_traj.extend(traj if isinstance(traj, list) else [])

    best_sequence = _decode_best_sequence(best_x)
    return {
        "x": x,
        "best_x": best_x,
        "trajectory": all_traj,
        "best_sequence": best_sequence,
    }


