import jax
import jax.numpy as jnp


def temperature_on_logits_xy():
    """Two-player temperature transform for pre_logits.

    Uses schedule["temperature_x"] for x and schedule["temperature_y"] for y.
    Fallback to schedule["temperature"] if per-player keys are missing.
    Apply by passing in the optimizer's nested transform dispatcher.
    """

    def _x(x, ctx):
        sched = ctx.get("schedule") or {}
        t = float(sched.get("temperature_x", sched.get("temperature", 1.0)))
        return x / max(1e-6, t)

    def _y(y, ctx):
        sched = ctx.get("schedule") or {}
        t = float(sched.get("temperature_y", sched.get("temperature", 1.0)))
        return y / max(1e-6, t)

    return {"x": {"pre_logits": [_x]}, "y": {"pre_logits": [_y]}}


def gradient_normalizer_xy(mode: str = "l2", eps: float = 1e-6):
    def _make():
        def _fn(g, ctx):
            n = jnp.sqrt((g ** 2).sum())
            if mode == "clip":
                return jnp.where(n > eps, g * (eps / (n + 1e-9)), g)
            return jnp.where(n > eps, g / (n + eps), g)

        return _fn

    fn = _make()
    return {"x": {"grad": [fn]}, "y": {"grad": [fn]}}


def hard_one_hot_xy():
    def _fn(x, ctx):
        idx = jnp.argmax(x, axis=-1)
        oh = jax.nn.one_hot(idx, x.shape[-1])
        return oh * 10.0 + (1.0 - oh) * -10.0

    return {"x": {"post_logits": [ _fn ]}, "y": {"post_logits": [ _fn ]}}


