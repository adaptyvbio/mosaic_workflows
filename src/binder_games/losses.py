from typing import Callable, Sequence, Tuple, Dict, Any
import jax.numpy as jnp
import jax


def make_minmax_loss(loss_x: Callable, loss_y: Callable, weight_y: float = 1.0) -> Callable:
    """Return two-arg loss: (x_probs, y_probs, key) -> (value, aux)

    Expects loss_x and loss_y to be Mosaic-compatible callables over probabilities.
    """

    def loss_fn(x_probs, y_probs, key=None) -> Tuple[float, Dict[str, Any]]:
        vx, auxx = loss_x(x_probs, key=key)
        vy, auxy = loss_y(y_probs, key=key)
        v = vx - float(weight_y) * vy
        aux = {
            "value_x": jnp.asarray(vx),
            "value_y": jnp.asarray(vy),
            "x": auxx,
            "y": auxy,
        }
        return v, aux

    return loss_fn


def make_multi_adversary_loss(loss_x: Callable, losses_y: Sequence[Callable], agg: str = "max") -> Callable:
    """Aggregate multiple adversaries by max or mean.
    Returns loss(x_probs, y_probs_stack, key) where y_probs_stack is (K, L, C) or a list.
    """

    agg = str(agg).lower()

    def loss_fn(x_probs, y_probs_stack, key=None):
        vx, auxx = loss_x(x_probs, key=key)
        vals = []
        auxys = []
        for i, ly in enumerate(losses_y):
            yp = y_probs_stack[i] if hasattr(y_probs_stack, "__getitem__") else y_probs_stack
            vy, auxy = ly(yp, key=key)
            vals.append(float(vy))
            auxys.append(auxy)
        if agg == "mean":
            vy_val = sum(vals) / max(1, len(vals))
        else:
            vy_val = max(vals) if vals else 0.0
        v = vx - vy_val
        aux = {"value_x": jnp.asarray(vx), "value_y": jnp.asarray(vy_val), "x": auxx, "y": {"adversaries": auxys}}
        return v, aux

    return loss_fn


def make_dro_loss(loss_x: Callable, losses_y: Sequence[Callable], radius: float | None = None) -> Callable:
    """Distributionally robust: maximize over convex weights on adversaries (no projection here; use schedule to pass weights).

    Expects schedule to provide weights via aux_context or loss wrapper; this is a simple helper for wiring.
    """

    def loss_fn(x_probs, y_probs_stack, key=None, weights=None):
        vx, auxx = loss_x(x_probs, key=key)
        K = len(losses_y)
        if weights is None:
            w = [1.0 / K] * K
        else:
            w = list(weights)
            s = sum(w)
            w = [wi / s for wi in w]
        val = 0.0
        auxys = []
        for i, ly in enumerate(losses_y):
            yp = y_probs_stack[i] if hasattr(y_probs_stack, "__getitem__") else y_probs_stack
            vy, auxy = ly(yp, key=key)
            val += float(w[i]) * float(vy)
            auxys.append(auxy)
        v = vx - val
        aux = {"value_x": jnp.asarray(vx), "value_y": jnp.asarray(val), "x": auxx, "y": {"adversaries": auxys, "weights": w}}
        return v, aux

    return loss_fn


def worst_case_panel_loss(loss_on: Callable, losses_off: Sequence[Callable], beta: float = 1.0) -> Callable:
    """Single-arg loss: v(x) = L_on(x) + beta * max_i L_off_i(x).

    loss_on/off are Mosaic-compatible callables over probabilities.
    """

    def loss_fn(x_probs, key=None) -> Tuple[float, Dict[str, Any]]:
        v_on, aux_on = loss_on(x_probs, key=key)
        off_vals = []
        off_aux = []
        for loff in losses_off:
            v_i, a_i = loff(x_probs, key=key)
            off_vals.append(jnp.asarray(v_i))
            off_aux.append(a_i)
        off_vec = jnp.stack(off_vals) if off_vals else jnp.zeros((0,), dtype=jnp.float32)
        if off_vec.size == 0:
            v_off = jnp.asarray(0.0)
            idx = jnp.asarray(-1)
        else:
            idx = jnp.argmax(off_vec)
            v_off = off_vec[idx]
        v = v_on + float(beta) * v_off
        aux: Dict[str, Any] = {
            "value_x": jnp.asarray(v_on),
            "value_y": jnp.asarray(v_off),
            "x": aux_on,
            "y": {"off_values": off_aux, "off_vector": off_vec, "worst_idx": idx},
        }
        return v, aux

    return loss_fn


def make_dro_two_player_loss(loss_on: Callable, losses_off: Sequence[Callable], beta: float = 1.0, prior: Sequence[float] | None = None, tau: float = 0.0, epsilon: float = 1e-9) -> Callable:
    """Two-arg loss: v(x,y) = L_on(x) + beta * sum_i y_i L_off_i(x) - tau * KL(y || prior).

    y is a probability vector produced by softmax(y_logits) in the optimizer.
    """

    pi = None if prior is None else jnp.asarray(prior, dtype=jnp.float32)

    def loss_fn(x_probs, y_probs, key=None):
        v_on, aux_on = loss_on(x_probs, key=key)
        off_vals = []
        off_aux = []
        for loff in losses_off:
            v_i, a_i = loff(x_probs, key=key)
            off_vals.append(jnp.asarray(v_i))
            off_aux.append(a_i)
        off_vec = jnp.stack(off_vals) if off_vals else jnp.zeros((0,), dtype=jnp.float32)
        y = jnp.asarray(y_probs, dtype=jnp.float32)
        if off_vec.size == 0:
            mix = jnp.asarray(0.0)
        else:
            mix = (y * off_vec).sum()
        reg = jnp.asarray(0.0)
        if pi is not None and tau > 0.0:
            pi_n = pi / (jnp.sum(pi) + epsilon)
            y_n = y / (jnp.sum(y) + epsilon)
            kl = jnp.sum(y_n * (jnp.log(y_n + epsilon) - jnp.log(pi_n + epsilon)))
            reg = tau * kl
        v = v_on + float(beta) * mix - reg
        aux: Dict[str, Any] = {
            "value_x": jnp.asarray(v_on),
            "value_y": jnp.asarray(mix),
            "x": aux_on,
            "y": {"off_values": off_aux, "off_vector": off_vec, "weights": y, "prior": pi},
        }
        return v, aux

    return loss_fn


def bayesian_stackelberg_closed_form(loss_on: Callable, losses_off: Sequence[Callable], beta: float = 1.0, prior: Sequence[float] | None = None, tau: float = 0.5, epsilon: float = 1e-9) -> Callable:
    """Single-arg, differentiable closed-form follower: y*(x) = softmax((L_off(x)/tau) + log pi).

    v(x) = L_on(x) + beta * sum y*_i L_off_i(x) - tau * KL(y* || pi)
    """

    pi = None if prior is None else jnp.asarray(prior, dtype=jnp.float32)

    def loss_fn(x_probs, key=None):
        v_on, aux_on = loss_on(x_probs, key=key)
        vals = []
        auxs = []
        for loff in losses_off:
            v_i, a_i = loff(x_probs, key=key)
            vals.append(jnp.asarray(v_i))
            auxs.append(a_i)
        v_off = jnp.stack(vals) if vals else jnp.zeros((0,), dtype=jnp.float32)
        if v_off.size == 0:
            mix = jnp.asarray(0.0)
            y_star = jnp.zeros((0,), dtype=jnp.float32)
            reg = jnp.asarray(0.0)
        else:
            log_pi = jnp.log((pi / (jnp.sum(pi) + epsilon)) + epsilon) if pi is not None else 0.0
            logits = (v_off / max(1e-6, float(tau))) + log_pi
            y_star = jax.nn.softmax(logits)
            mix = jnp.sum(y_star * v_off)
            reg = jnp.asarray(0.0)
            if pi is not None and tau > 0.0:
                pi_n = pi / (jnp.sum(pi) + epsilon)
                kl = jnp.sum(y_star * (jnp.log(y_star + epsilon) - jnp.log(pi_n + epsilon)))
                reg = tau * kl
        v = v_on + float(beta) * mix - reg
        aux: Dict[str, Any] = {
            "value_x": jnp.asarray(v_on),
            "value_y": jnp.asarray(mix),
            "x": aux_on,
            "y": {"off_values": auxs, "off_vector": v_off, "weights": y_star, "prior": pi},
        }
        return v, aux

    return loss_fn


def bayesian_stackelberg_two_player_loss(loss_on: Callable, losses_off: Sequence[Callable], beta: float = 1.0, prior: Sequence[float] | None = None, tau: float = 0.5, epsilon: float = 1e-9) -> Callable:
    """Two-arg: v(x,y) = L_on(x) + beta * sum y_i L_off_i(x) - tau * KL(y || pi).

    Use with stackelberg_logits; y is the follower distribution.
    """

    pi = None if prior is None else jnp.asarray(prior, dtype=jnp.float32)

    def loss_fn(x_probs, y_probs, key=None):
        v_on, aux_on = loss_on(x_probs, key=key)
        vals = []
        auxs = []
        for loff in losses_off:
            v_i, a_i = loff(x_probs, key=key)
            vals.append(jnp.asarray(v_i))
            auxs.append(a_i)
        v_off = jnp.stack(vals) if vals else jnp.zeros((0,), dtype=jnp.float32)
        y = jnp.asarray(y_probs, dtype=jnp.float32)
        mix = jnp.sum(y * v_off) if v_off.size > 0 else jnp.asarray(0.0)
        reg = jnp.asarray(0.0)
        if pi is not None and tau > 0.0:
            pi_n = pi / (jnp.sum(pi) + epsilon)
            y_n = y / (jnp.sum(y) + epsilon)
            kl = jnp.sum(y_n * (jnp.log(y_n + epsilon) - jnp.log(pi_n + epsilon)))
            reg = tau * kl
        v = v_on + float(beta) * mix - reg
        aux: Dict[str, Any] = {
            "value_x": jnp.asarray(v_on),
            "value_y": jnp.asarray(mix),
            "x": aux_on,
            "y": {"off_values": auxs, "off_vector": v_off, "weights": y, "prior": pi},
        }
        return v, aux

    return loss_fn


