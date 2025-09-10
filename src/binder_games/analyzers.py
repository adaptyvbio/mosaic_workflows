from typing import Any, Dict
import numpy as np
import jax.numpy as jnp


def saddle_gap_estimate():
    """Estimate saddle gap if components present in aux.

    Looks for aux["value_x"], aux["value_y"], else falls back to aux["value"].
    Returns {"gap": ...}.
    """

    def _fn(aux_or_ctx: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            vx = aux_or_ctx.get("value_x")
            vy = aux_or_ctx.get("value_y")
            if vx is not None and vy is not None:
                out["gap"] = float(vx - vy)
            else:
                v = aux_or_ctx.get("value")
                if v is not None:
                    out["gap"] = float(abs(v))
        except Exception:
            pass
        return out

    return _fn


def value_components(keys=("value_x", "value_y")):
    def _fn(aux_or_ctx: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in keys:
            if k in aux_or_ctx:
                try:
                    out[k] = float(aux_or_ctx[k])
                except Exception:
                    pass
        return out

    return _fn


def decode_sequences_xy(vocab: str = "ARNDCQEGHILKMFPSTWYV"):
    idx_to_aa = list(vocab)

    def _decode(p):
        try:
            idx = np.argmax(np.array(p), axis=-1)
            return "".join(idx_to_aa[i] for i in idx)
        except Exception:
            return None

    def _fn(aux_or_ctx: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            x_probs = ((aux_or_ctx.get("x") or {}).get("probs"))
            y_probs = ((aux_or_ctx.get("y") or {}).get("probs"))
            if x_probs is not None:
                out["seq_x"] = _decode(x_probs)
            if y_probs is not None:
                out["seq_y"] = _decode(y_probs)
        except Exception:
            pass
        return out

    return _fn


def off_target_weights_summary():
    """Summarize follower weights y and worst-case index if present.
    Expects aux["y"]["weights"], aux["y"]["worst_idx"], or aux["y"]["off_vector"].
    """

    def _fn(aux_or_ctx: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            y = ((aux_or_ctx.get("y") or {}).get("weights"))
            if y is not None:
                y_np = np.array(y)
                out["y_max_w"] = float(y_np.max())
                out["y_entropy"] = float(-(y_np * np.log(y_np + 1e-9)).sum())
            worst = ((aux_or_ctx.get("y") or {}).get("worst_idx"))
            if worst is not None:
                out["worst_idx"] = int(np.array(worst))
            ov = ((aux_or_ctx.get("y") or {}).get("off_vector"))
            if ov is not None and y is not None:
                out["weighted_off"] = float((np.array(ov) * np.array(y)).sum())
        except Exception:
            pass
        return out

    return _fn



