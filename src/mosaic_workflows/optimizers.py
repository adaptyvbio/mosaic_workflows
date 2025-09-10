import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable
import optax

from mosaic.optimizers import simplex_APGM as mosaic_simplex_APGM
from mosaic.optimizers import gradient_MCMC as mosaic_gradient_MCMC
from mosaic.optimizers import update_states


def _eval_loss_and_grad(loss_function, x, key):
    x = np.array(x, dtype=np.float32)
    (v, aux), g = _jit_value_and_grad(loss_function, x=x, key=key)
    return (v, aux), g - g.mean(axis=-1, keepdims=True)


@eqx.filter_jit
def _jit_value_and_grad(loss, x, key):
    return eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)


def _apply_transforms(kind: str, transforms: dict | None, arr, ctx):
    if not transforms:
        return arr
    fns = transforms.get(kind) or []
    for fn in fns:
        arr = fn(arr, ctx)
    return arr

def simplex_APGM_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, logspace: bool = False, update_loss_state: bool = False, **kwargs):
    sched0 = schedule(0, 0) if callable(schedule) else (schedule or {})
    stepsize = float(sched0.get("stepsize", 0.1))
    scale = float(sched0.get("scale", 1.0))

    # pre-transform logits to probs if user asks
    logits = _apply_transforms("pre_logits", transforms, x, {"schedule": sched0})
    probs = jax.nn.softmax(logits)
    probs = _apply_transforms("pre_probs", transforms, probs, {"schedule": sched0})

    def traj(aux, x_soft):
        if trajectory_fn is None:
            return None
        try:
            return trajectory_fn(aux, x_soft)
        except Exception:
            return None

    x_soft, best_x_soft, tr = mosaic_simplex_APGM(
        loss_function=loss_function,
        x=probs if not logspace else logits,
        n_steps=n_steps,
        stepsize=stepsize,
        key=key,
        scale=scale,
        trajectory_fn=traj,
        logspace=logspace,
        update_loss_state=update_loss_state,
    )

    # map back to logits
    if logspace:
        x_logits = x_soft
        best_logits = best_x_soft
    else:
        x_logits = jnp.log(jnp.clip(x_soft, 1e-9, 1.0))
        best_logits = jnp.log(jnp.clip(best_x_soft, 1e-9, 1.0))

    x_logits = _apply_transforms("post_logits", transforms, x_logits, {"schedule": sched0})
    return x_logits, best_logits, tr


def gradient_MCMC_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, update_loss_state: bool = False, **kwargs):
    sched0 = schedule(0, 0) if callable(schedule) else (schedule or {})
    temp = float(sched0.get("temperature", 0.001))
    proposal_temp = float(sched0.get("proposal_temp", 0.01))

    logits = _apply_transforms("pre_logits", transforms, x, {"schedule": sched0})
    probs = jax.nn.softmax(logits)
    probs = _apply_transforms("pre_probs", transforms, probs, {"schedule": sched0})
    seq = jnp.argmax(probs, axis=-1).astype(jnp.int32)

    seq = mosaic_gradient_MCMC(
        loss=loss_function,
        sequence=np.array(seq),
        temp=temp,
        proposal_temp=proposal_temp,
        steps=n_steps,
        key=key,
    )
    x_logits = jax.nn.one_hot(seq, 20)
    x_logits = _apply_transforms("post_logits", transforms, x_logits, {"schedule": sched0})
    return x_logits, x_logits, None


def rao_gumbel_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, update_loss_state: bool = False, **kwargs):
    """Rao-Blackwellized straight-through Gumbel estimator (hard forward, conditional Gumbel surrogate).

    Forward uses a hard one-hot sample D. Gradients flow through a surrogate
    that averages K conditional Gumbel-Softmax relaxations softmax((logits + G|D)/T).
    Conditional noise is stop_gradient'ed to avoid backprop through G|D.
    """
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))
    best_val = np.inf
    best_x = x
    logits = x
    K = int(kwargs.get("num_samples", 4))

    def _conditional_gumbel_noise(rng_key, logits, D, k):
        # E ~ Exp(1) shape [k, L, A]
        E = jax.random.exponential(rng_key, shape=(k,) + logits.shape)
        # Ei: [k, L, 1]
        Ei = jnp.sum(D[None, ...] * E, axis=-1, keepdims=True)
        Z = jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)
        # adjusted logits s.t. argmax(logits + noise) = D
        adjusted = (D[None, ...] * (-jnp.log(Ei + 1e-12) + jnp.log(Z + 1e-12)) +
                    (1.0 - D[None, ...]) * -jnp.log(E / (jnp.exp(logits)[None, ...] + 1e-12) + Ei / (Z + 1e-12)))
        # conditional noise G|D = adjusted - logits
        cond = adjusted - logits[None, ...]
        return jax.lax.stop_gradient(cond)

    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        ctx = {"schedule": sched, **(aux_context or {})}
        logits = _apply_transforms("pre_logits", transforms, logits, ctx)
        temp = float(sched.get("temperature", 1.0))

        # Hard sample D (one-hot)
        idx = jax.random.categorical(jax.random.fold_in(key, step * 2 + 0), logits=logits, axis=-1)
        D = jax.nn.one_hot(idx, logits.shape[-1])

        # Conditional Gumbel surrogates
        cond = _conditional_gumbel_noise(jax.random.fold_in(key, step * 2 + 1), logits, D, K)
        adjusted = logits[None, ...] + cond
        surrogate = jax.nn.softmax(adjusted / max(1e-6, temp), axis=-1).mean(axis=0)

        # Replace-gradient: forward hard D, gradient from surrogate
        probs_input = D + (surrogate - jax.lax.stop_gradient(surrogate))
        probs_input = _apply_transforms("pre_probs", transforms, probs_input, ctx)

        (value, aux), g = _eval_loss_and_grad(loss_function, x=probs_input, key=key)
        if update_loss_state:
            try:
                loss_function = update_states(aux, loss_function)
            except Exception:
                pass
        key = jax.random.fold_in(key, 0)

        g = _apply_transforms("grad", transforms, g, ctx)
        logits = logits - float(sched.get("lr", 0.1)) * g
        logits = _apply_transforms("post_logits", transforms, logits, ctx)

        if float(value) < best_val:
            best_val = float(value)
            best_x = logits

        if trajectory_fn is not None:
            try:
                aux = {"loss": float(value), "aux": aux}
                trajectory_fn(aux, probs_input)
            except Exception:
                pass

    return logits, best_x, None


def st_gumbel_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, update_loss_state: bool = False, **kwargs):
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))
    best_val = np.inf
    best_x = x
    logits = x
    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        ctx = {"schedule": sched, **(aux_context or {})}
        logits = _apply_transforms("pre_logits", transforms, logits, ctx)
        temp = float(sched.get("temperature", 1.0))
        gumbel = -jnp.log(-jnp.log(jax.random.uniform(jax.random.fold_in(key, step), logits.shape) + 1e-8) + 1e-8)
        y = (logits + gumbel) / max(1e-6, temp)
        probs_relaxed = jax.nn.softmax(y, axis=-1)
        probs_relaxed = _apply_transforms("pre_probs", transforms, probs_relaxed, ctx)
        (value, aux), g = _eval_loss_and_grad(loss_function, x=probs_relaxed, key=key)
        if update_loss_state:
            try:
                loss_function = update_states(aux, loss_function)
            except Exception:
                pass
        key = jax.random.fold_in(key, 0)
        g = _apply_transforms("grad", transforms, g, ctx)
        logits = logits - float(sched.get("lr", 0.1)) * g
        logits = _apply_transforms("post_logits", transforms, logits, ctx)
        if value < best_val:
            best_val = float(value)
            best_x = logits
        if trajectory_fn is not None:
            try:
                aux = {"loss": float(value), "aux": aux}
                trajectory_fn(aux, probs_relaxed)
            except Exception:
                pass
    return logits, best_x, None


def zgr_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, update_loss_state: bool = False, **kwargs):
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))
    best_val = np.inf
    best_x = x
    logits = x
    clip = float(kwargs.get("gradient_clip", 10.0))
    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        ctx = {"schedule": sched, **(aux_context or {})}
        logits = _apply_transforms("pre_logits", transforms, logits, ctx)

        # Discrete sample for forward
        idx = jax.random.categorical(jax.random.fold_in(key, step), logits=logits, axis=-1)
        x_onehot = jax.nn.one_hot(idx, logits.shape[-1])

        # ZGR surrogate: 0.5*(ST + DARN(phi_bar))
        p = jax.nn.softmax(logits, axis=-1)
        log_p = jax.nn.log_softmax(logits, axis=-1)
        log_px = jnp.take_along_axis(log_p, idx[..., None], axis=-1)[..., 0]
        log_px = jnp.clip(log_px, a_min=-clip, a_max=clip)

        dx_st = p
        dx_darn = (x_onehot - jax.lax.stop_gradient(p)) * log_px[..., None]
        dx = 0.5 * (dx_st + dx_darn)

        # Straight-through: forward x_onehot; backward through dx
        probs_input = x_onehot + (dx - jax.lax.stop_gradient(dx))
        probs_input = _apply_transforms("pre_probs", transforms, probs_input, ctx)
        (value, aux), g = _eval_loss_and_grad(loss_function, x=probs_input, key=key)
        if update_loss_state:
            try:
                loss_function = update_states(aux, loss_function)
            except Exception:
                pass
        key = jax.random.fold_in(key, 0)
        g = _apply_transforms("grad", transforms, g, ctx)
        logits = logits - float(sched.get("lr", 0.1)) * g
        logits = _apply_transforms("post_logits", transforms, logits, ctx)
        if value < best_val:
            best_val = float(value)
            best_x = logits
        if trajectory_fn is not None:
            try:
                aux = {"loss": float(value), "aux": aux}
                trajectory_fn(aux, probs_input)
            except Exception:
                pass
    return logits, best_x, None


def semi_greedy_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, update_loss_state: bool = False, proposals_per_step: int = 10, position_weighting: str = "1-plddt", **kwargs):
    """Discrete mutation search guided by model confidence.

    Expects that calling `loss_function(probs, key)` returns (value, aux) where aux may
    contain "plddt_per_residue" (e.g., when combined with a reporter loss term).
    """
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    logits = x
    best_x = x
    best_val = np.inf

    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        ctx = {"schedule": sched, **(aux_context or {})}
        logits = _apply_transforms("pre_logits", transforms, logits, ctx)
        probs = jax.nn.softmax(logits, axis=-1)
        probs = _apply_transforms("pre_probs", transforms, probs, ctx)

        (value, aux), _ = _eval_loss_and_grad(loss_function, x=probs, key=key)
        if update_loss_state:
            try:
                loss_function = update_states(aux, loss_function)
            except Exception:
                pass
        key = jax.random.fold_in(key, 0)

        # derive per-position weights
        binder_len = probs.shape[0]
        if position_weighting == "1-plddt" and isinstance(aux, dict):
            # Try to extract pLDDT vector from aux in a Mosaic-compatible way
            plddt = None
            # Direct
            plddt = aux.get("plddt_per_residue") if plddt is None else plddt
            # Under Boltz1 Loss wrapper (may be a list of aux dicts)
            inner = aux.get("boltz1")
            if plddt is None and isinstance(inner, list):
                for item in inner:
                    if isinstance(item, dict) and "plddt_per_residue" in item:
                        plddt = item["plddt_per_residue"]
                        break
            if plddt is None and isinstance(inner, dict):
                plddt = inner.get("plddt_per_residue")

            if plddt is not None:
                w = np.array(jnp.asarray(1.0 - plddt))
            else:
                w = np.ones((binder_len,), dtype=np.float32)
        else:
            w = np.ones((binder_len,), dtype=np.float32)
        w = w / (w.sum() + 1e-8)

        # generate proposals
        rng = np.random.default_rng(int(jnp.abs(jnp.sum(jnp.asarray(probs)*1e6))) % (2**32-1))
        candidates = []
        scores = []
        for t in range(int(sched.get("proposals_per_step", proposals_per_step))):
            i = rng.choice(np.arange(binder_len), p=w)
            p_i = np.array(probs[i])
            p_i = p_i / (p_i.sum() + 1e-8)
            aa = rng.choice(np.arange(p_i.shape[-1]), p=p_i)
            # discrete one-hot sequence from probs, mutate position i
            seq = np.eye(probs.shape[-1], dtype=np.float32)[np.argmax(np.array(probs), axis=-1)]
            seq[i] = 0.0
            seq[i, aa] = 1.0
            seq = jnp.asarray(seq)
            v, _ = loss_function(seq, key=key)
            candidates.append(seq)
            scores.append(float(v))

        # pick best (lowest loss)
        best_idx = int(np.argmin(scores)) if scores else -1
        if best_idx >= 0 and scores[best_idx] < float(value):
            chosen = candidates[best_idx]
            logits = jnp.where(chosen > 0.5, 10.0, -10.0)
            value = scores[best_idx]

        logits = _apply_transforms("post_logits", transforms, logits, ctx)

        if float(value) < best_val:
            best_val = float(value)
            best_x = logits

        if trajectory_fn is not None:
            try:
                trajectory_fn({"loss": float(value), "aux": aux}, jax.nn.softmax(logits, axis=-1))
            except Exception:
                pass

    return logits, best_x, None


def rso_box(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, optim=None, update_loss_state: bool = False, **kwargs):
    """Box-constrained optax optimizer over probabilities.

    This optimizer treats `x` as probabilities in [0,1] (not necessarily simplex) and updates them
    directly using an Optax optimizer. It supports the `pre_probs` and `grad` transform chains.

    Notes:
    - If you need simplex constraints, use `simplex_APGM_adapter` instead.
    - Schedules can still be used; we scale updates by the schedule LR each step.
    """
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    # default optimizer: clip then SGD with unit LR; per-step LR comes from schedule
    if optim is None:
        optim = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(learning_rate=1.0))
    opt_state = optim.init(x)

    best_val = np.inf
    best_x = x

    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        ctx = {"schedule": sched, **(aux_context or {})}

        # Treat x as probs; allow transforms on pre_probs
        probs = _apply_transforms("pre_probs", transforms, x, ctx)

        (value, aux), g = _eval_loss_and_grad(loss_function, x=probs, key=key)
        if update_loss_state:
            try:
                loss_function = update_states(aux, loss_function)
            except Exception:
                pass
        key = jax.random.fold_in(key, 0)

        # apply grad transforms
        g = _apply_transforms("grad", transforms, g, ctx)

        updates, opt_state = optim.update(g, opt_state, x)

        # scale updates by schedule LR
        lr = float(sched.get("learning_rate", sched.get("lr", 0.1)))
        updates = jax.tree.map(lambda u: u * lr, updates)

        x = optax.apply_updates(x, updates)
        x = jnp.clip(x, 0.0, 1.0)

        if float(value) < best_val:
            best_val = float(value)
            best_x = x

        if trajectory_fn is not None:
            try:
                aux = {"loss": float(value), "aux": aux}
                trajectory_fn(aux, x)
            except Exception:
                pass

    return x, best_x, None


def optax_logits(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, optim=None, update_loss_state: bool = False, **kwargs):
    """Optax-based optimizer on logits (with softmax forward), Mosaic-style.

    - Applies pre_logits → softmax → pre_probs → loss
    - Gradients are taken w.r.t. probs and used to update the transformed logits (same approximation as sgd_logits)
    - Applies post_logits after each step
    - Learning rate comes from schedule; the optax chain should use lr=1.0 internally
    """
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    if optim is None:
        optim = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(learning_rate=1.0))
    opt_state = optim.init(x)

    best_val = np.inf
    best_x = x

    for step in range(n_steps):
        sched = schedule(step, step) if callable(schedule) else (schedule or {})
        ctx = {"schedule": sched, **(aux_context or {})}

        logits = _apply_transforms("pre_logits", transforms, x, ctx)
        probs = jax.nn.softmax(logits, axis=-1)
        probs = _apply_transforms("pre_probs", transforms, probs, ctx)

        (value, aux), g = _eval_loss_and_grad(loss_function, x=probs, key=key)
        if update_loss_state:
            try:
                loss_function = update_states(aux, loss_function)
            except Exception:
                pass
        key = jax.random.fold_in(key, 0)

        g = _apply_transforms("grad", transforms, g, ctx)

        updates, opt_state = optim.update(g, opt_state, logits)
        lr = float(sched.get("learning_rate", sched.get("lr", 0.1)))
        updates = jax.tree.map(lambda u: u * lr, updates)
        logits = optax.apply_updates(logits, updates)
        x = _apply_transforms("post_logits", transforms, logits, ctx)

        if float(value) < best_val:
            best_val = float(value)
            best_x = x

        if trajectory_fn is not None:
            try:
                aux = {"loss": float(value), "aux": aux}
                trajectory_fn(aux, jax.nn.softmax(x, axis=-1))
            except Exception:
                pass

    return x, best_x, None


def sgd_logits_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, clip: float = 1.0, momentum: float = 0.0, update_loss_state: bool = False, **kwargs):
    optim = optax.chain(optax.clip_by_global_norm(clip), optax.sgd(learning_rate=1.0, momentum=momentum))
    return optax_logits(
        loss_function=loss_function,
        x=x,
        n_steps=n_steps,
        key=key,
        schedule=schedule,
        transforms=transforms,
        trajectory_fn=trajectory_fn,
        aux_context=aux_context,
        optim=optim,
        update_loss_state=update_loss_state,
        **kwargs,
    )


def adamw_logits_adapter(*, loss_function, x, n_steps, key=None, schedule=None, transforms=None, trajectory_fn=None, aux_context=None, clip: float = 1.0, weight_decay: float = 0.01, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8, update_loss_state: bool = False, **kwargs):
    optim = optax.chain(optax.clip_by_global_norm(clip), optax.adamw(learning_rate=1.0, weight_decay=weight_decay, b1=b1, b2=b2, eps=eps))
    return optax_logits(
        loss_function=loss_function,
        x=x,
        n_steps=n_steps,
        key=key,
        schedule=schedule,
        transforms=transforms,
        trajectory_fn=trajectory_fn,
        aux_context=aux_context,
        optim=optim,
        update_loss_state=update_loss_state,
        **kwargs,
    )

