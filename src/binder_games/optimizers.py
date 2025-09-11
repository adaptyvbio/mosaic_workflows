from typing import Callable, Dict, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def _apply_chain(fns, arr, ctx):
    if not fns:
        return arr
    for fn in fns:
        arr = fn(arr, ctx)
    return arr


def _t(side: str, kind: str, transforms: dict | None, arr, ctx):
    """Nested-or-flat transform dispatcher.

    If transforms is nested like {"x": {kind: [...]}, "y": {kind: [...]}}, use side chain.
    If transforms is flat like {kind: [...]}, apply to both sides.
    """
    if not transforms:
        return arr
    chains = []
    side_dict = transforms.get(side)
    if isinstance(side_dict, dict):
        chains.extend(side_dict.get(kind) or [])
    # flat dict fallback
    if isinstance(transforms.get(kind), list):
        chains.extend(transforms.get(kind) or [])
    return _apply_chain(chains, arr, ctx)


def _center_last_axis(g):
    return g - g.mean(axis=-1, keepdims=True)


@eqx.filter_jit
def _value_and_grads_two_player(loss_fn: Callable, x_probs, y_probs, key):
    def f(z, k):
        u, v = z
        return loss_fn(u, v, k)

    (v, aux), g = eqx.filter_value_and_grad(f, has_aux=True)((x_probs, y_probs), key)
    gx, gy = g
    return (v, aux), gx, gy


def _trajectory_call(trajectory_fn, value, aux, x_probs):
    if trajectory_fn is None:
        return None
    try:
        # Merge loss into a flat aux dict so analyzers can access top-level keys
        aux_flat = dict(aux) if isinstance(aux, dict) else {"aux": aux}
        aux_flat["loss"] = float(value)
        return trajectory_fn(aux_flat, x_probs)
    except Exception:
        return None


def _update_states_if_needed(loss_function, aux, update_loss_state: bool):
    if not update_loss_state:
        return loss_function
    try:
        from mosaic.optimizers import update_states
        return update_states(aux, loss_function)
    except Exception:
        return loss_function


def _ensure_key(key):
    if key is None:
        return jax.random.key(np.random.randint(0, 10000))
    return key


def _init_y_like(x, sched: Dict[str, Any]):
    init_mode = (sched or {}).get("y_init", "zeros")
    if init_mode == "random":
        return jnp.array(np.random.randn(*np.array(x).shape).astype(np.float32) * 0.1)
    if init_mode == "copy_x":
        return jnp.array(x)
    init_logits = (sched or {}).get("y_init_logits")
    if init_logits is not None:
        return jnp.asarray(init_logits)
    return jnp.zeros_like(x)


def minmax_logits(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, update_loss_state: bool = False, **kwargs):
    """Simultaneous descent-ascent in logits.

    loss_function expects signature: loss(x_probs, y_probs, key) -> (value, aux)
    """
    key = _ensure_key(key)
    best_val = np.inf
    best_x = x
    sched0 = schedule(0, 0) if callable(schedule) else (schedule or {})
    y = _init_y_like(x, sched0)

    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        lr_x = float(sched.get("lr_x", sched.get("learning_rate", 0.1)))
        lr_y = float(sched.get("lr_y", lr_x))
        ctx = {"schedule": sched, **(aux_context or {})}

        x_logits = _t("x", "pre_logits", transforms, x, ctx)
        y_logits = _t("y", "pre_logits", transforms, y, ctx)
        x_probs = _t("x", "pre_probs", transforms, jax.nn.softmax(x_logits, axis=-1), ctx)
        y_probs = _t("y", "pre_probs", transforms, jax.nn.softmax(y_logits, axis=-1), ctx)

        (value, aux), gx, gy = _value_and_grads_two_player(loss_function, x_probs, y_probs, key)
        loss_function = _update_states_if_needed(loss_function, aux, update_loss_state)
        key = jax.random.fold_in(key, 0)

        gx = _t("x", "grad", transforms, _center_last_axis(gx), ctx)
        gy = _t("y", "grad", transforms, _center_last_axis(gy), ctx)

        x = x_logits - lr_x * gx
        y = y_logits + lr_y * gy

        x = _t("x", "post_logits", transforms, x, ctx)
        y = _t("y", "post_logits", transforms, y, ctx)

        # enrich aux with components for analyzers
        try:
            aux = dict(aux) if isinstance(aux, dict) else {"aux": aux}
            aux["value"] = float(value)
            aux.setdefault("x", {})
            aux.setdefault("y", {})
            aux["x"]["probs"] = np.array(x_probs)
            aux["y"]["probs"] = np.array(y_probs)
            # Add simple gradient norm summaries for optimization dynamics analyses
            try:
                aux["x"]["grad_norm"] = float(np.linalg.norm(np.array(gx)))
            except Exception:
                pass
            try:
                aux["y"]["grad_norm"] = float(np.linalg.norm(np.array(gy)))
            except Exception:
                pass
        except Exception:
            pass

        if float(value) < best_val:
            best_val = float(value)
            best_x = x

        _trajectory_call(trajectory_fn, value, aux, x_probs)

    return x, best_x, None


def alternating_br_logits(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, update_loss_state: bool = False, **kwargs):
    """Alternating updates: one x-step, then k y-steps.

    k controlled by schedule["inner_y_steps"] (default 1).
    """
    key = _ensure_key(key)
    best_val = np.inf
    best_x = x
    sched0 = schedule(0, 0) if callable(schedule) else (schedule or {})
    y = _init_y_like(x, sched0)

    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        lr_x = float(sched.get("lr_x", sched.get("learning_rate", 0.1)))
        lr_y = float(sched.get("lr_y", lr_x))
        k_y = int(sched.get("inner_y_steps", 1))
        ctx = {"schedule": sched, **(aux_context or {})}

        # x step (descent)
        x_logits = _t("x", "pre_logits", transforms, x, ctx)
        y_logits = _t("y", "pre_logits", transforms, y, ctx)
        x_probs = _t("x", "pre_probs", transforms, jax.nn.softmax(x_logits, axis=-1), ctx)
        y_probs = _t("y", "pre_probs", transforms, jax.nn.softmax(y_logits, axis=-1), ctx)
        (value, aux), gx, gy = _value_and_grads_two_player(loss_function, x_probs, y_probs, key)
        loss_function = _update_states_if_needed(loss_function, aux, update_loss_state)
        key = jax.random.fold_in(key, 0)
        gx = _t("x", "grad", transforms, _center_last_axis(gx), ctx)
        x = _t("x", "post_logits", transforms, x_logits - lr_x * gx, ctx)

        # y steps (ascent)
        y_logits = _t("y", "pre_logits", transforms, y, ctx)
        for _ in range(k_y):
            x_probs = _t("x", "pre_probs", transforms, jax.nn.softmax(x, axis=-1), ctx)
            y_probs = _t("y", "pre_probs", transforms, jax.nn.softmax(y_logits, axis=-1), ctx)
            (value, aux), gx, gy = _value_and_grads_two_player(loss_function, x_probs, y_probs, key)
            key = jax.random.fold_in(key, 1)
            gy = _t("y", "grad", transforms, _center_last_axis(gy), ctx)
            y_logits = y_logits + lr_y * gy
        y = _t("y", "post_logits", transforms, y_logits, ctx)

        # enrich aux and track best
        try:
            aux = dict(aux) if isinstance(aux, dict) else {"aux": aux}
            aux["value"] = float(value)
            aux.setdefault("x", {})
            aux.setdefault("y", {})
            aux["x"]["probs"] = np.array(_t("x", "pre_probs", transforms, jax.nn.softmax(x, axis=-1), ctx))
            aux["y"]["probs"] = np.array(_t("y", "pre_probs", transforms, jax.nn.softmax(y, axis=-1), ctx))
        except Exception:
            pass

        if float(value) < best_val:
            best_val = float(value)
            best_x = x

        _trajectory_call(trajectory_fn, value, aux, _t("x", "pre_probs", transforms, jax.nn.softmax(x, axis=-1), ctx))

    return x, best_x, None


def stackelberg_logits(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, update_loss_state: bool = False, **kwargs):
    """Leader-follower: for each x step, run best-response ascent on y for br_steps.

    Non-differentiable BR by default (stop-grad). Controlled by schedule keys:
    - br_steps (int)
    - lr_x, lr_y
    - reinit_y_each_step (bool)
    """
    key = _ensure_key(key)
    best_val = np.inf
    best_x = x
    sched0 = schedule(0, 0) if callable(schedule) else (schedule or {})
    y = _init_y_like(x, sched0)

    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        lr_x = float(sched.get("lr_x", sched.get("learning_rate", 0.1)))
        lr_y = float(sched.get("lr_y", lr_x))
        br_steps = int(sched.get("br_steps", sched.get("inner_y_steps", 5)))
        reinit_y = bool(sched.get("reinit_y_each_step", False))
        ctx = {"schedule": sched, **(aux_context or {})}

        if reinit_y:
            y = _init_y_like(x, sched)

        # best-response loop on y (ascent)
        y_logits = _t("y", "pre_logits", transforms, y, ctx)
        for _ in range(br_steps):
            x_probs = _t("x", "pre_probs", transforms, jax.nn.softmax(_t("x", "pre_logits", transforms, x, ctx), axis=-1), ctx)
            y_probs = _t("y", "pre_probs", transforms, jax.nn.softmax(y_logits, axis=-1), ctx)
            (value, aux), gx, gy = _value_and_grads_two_player(loss_function, x_probs, y_probs, key)
            key = jax.random.fold_in(key, 2)
            gy = _t("y", "grad", transforms, _center_last_axis(gy), ctx)
            y_logits = y_logits + lr_y * gy
        y = _t("y", "post_logits", transforms, y_logits, ctx)

        # x step (descent) against the BR y
        x_logits = _t("x", "pre_logits", transforms, x, ctx)
        x_probs = _t("x", "pre_probs", transforms, jax.nn.softmax(x_logits, axis=-1), ctx)
        y_probs = _t("y", "pre_probs", transforms, jax.nn.softmax(y, axis=-1), ctx)
        (value, aux), gx, gy = _value_and_grads_two_player(loss_function, x_probs, y_probs, key)
        loss_function = _update_states_if_needed(loss_function, aux, update_loss_state)
        key = jax.random.fold_in(key, 0)
        gx = _t("x", "grad", transforms, _center_last_axis(gx), ctx)
        x = _t("x", "post_logits", transforms, x_logits - lr_x * gx, ctx)

        try:
            aux = dict(aux) if isinstance(aux, dict) else {"aux": aux}
            aux["value"] = float(value)
            aux.setdefault("x", {})
            aux.setdefault("y", {})
            aux["x"]["probs"] = np.array(x_probs)
            aux["y"]["probs"] = np.array(y_probs)
        except Exception:
            pass

        if float(value) < best_val:
            best_val = float(value)
            best_x = x

        _trajectory_call(trajectory_fn, value, aux, x_probs)

    return x, best_x, None


def extragradient_minmax_logits(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, update_loss_state: bool = False, **kwargs):
    """Extragradient (EG/OGDA-like) stabilized min–max on logits.
    """
    key = _ensure_key(key)
    best_val = np.inf
    best_x = x
    sched0 = schedule(0, 0) if callable(schedule) else (schedule or {})
    y = _init_y_like(x, sched0)

    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        lr_x = float(sched.get("lr_x", sched.get("learning_rate", 0.1)))
        lr_y = float(sched.get("lr_y", lr_x))
        ctx = {"schedule": sched, **(aux_context or {})}

        # grads at current point
        x_logits = _t("x", "pre_logits", transforms, x, ctx)
        y_logits = _t("y", "pre_logits", transforms, y, ctx)
        x_probs = _t("x", "pre_probs", transforms, jax.nn.softmax(x_logits, axis=-1), ctx)
        y_probs = _t("y", "pre_probs", transforms, jax.nn.softmax(y_logits, axis=-1), ctx)
        (value, aux), gx0, gy0 = _value_and_grads_two_player(loss_function, x_probs, y_probs, key)
        key = jax.random.fold_in(key, 0)
        gx0 = _t("x", "grad", transforms, _center_last_axis(gx0), ctx)
        gy0 = _t("y", "grad", transforms, _center_last_axis(gy0), ctx)

        # lookahead
        x_la = _t("x", "post_logits", transforms, x_logits - lr_x * gx0, ctx)
        y_la = _t("y", "post_logits", transforms, y_logits + lr_y * gy0, ctx)

        # grads at lookahead
        x_la_logits = _t("x", "pre_logits", transforms, x_la, ctx)
        y_la_logits = _t("y", "pre_logits", transforms, y_la, ctx)
        x_la_probs = _t("x", "pre_probs", transforms, jax.nn.softmax(x_la_logits, axis=-1), ctx)
        y_la_probs = _t("y", "pre_probs", transforms, jax.nn.softmax(y_la_logits, axis=-1), ctx)
        (value, aux), gx1, gy1 = _value_and_grads_two_player(loss_function, x_la_probs, y_la_probs, key)
        loss_function = _update_states_if_needed(loss_function, aux, update_loss_state)
        key = jax.random.fold_in(key, 1)
        gx1 = _t("x", "grad", transforms, _center_last_axis(gx1), ctx)
        gy1 = _t("y", "grad", transforms, _center_last_axis(gy1), ctx)

        # actual update
        x = _t("x", "post_logits", transforms, x_logits - lr_x * gx1, ctx)
        y = _t("y", "post_logits", transforms, y_logits + lr_y * gy1, ctx)

        try:
            aux = dict(aux) if isinstance(aux, dict) else {"aux": aux}
            aux["value"] = float(value)
            aux.setdefault("x", {})
            aux.setdefault("y", {})
            aux["x"]["probs"] = np.array(_t("x", "pre_probs", transforms, jax.nn.softmax(x, axis=-1), ctx))
            aux["y"]["probs"] = np.array(_t("y", "pre_probs", transforms, jax.nn.softmax(y, axis=-1), ctx))
        except Exception:
            pass

        if float(value) < best_val:
            best_val = float(value)
            best_x = x

        _trajectory_call(trajectory_fn, value, aux, _t("x", "pre_probs", transforms, jax.nn.softmax(x, axis=-1), ctx))

    return x, best_x, None


