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



def probs_entropy_xy():
    """Mean categorical entropy for x and y per step, from probs.

    Returns keys: {"ent_x": float, "ent_y": float}
    """

    import numpy as _np

    def _H(p):
        try:
            pp = _np.clip(_np.array(p, dtype=_np.float64), 1e-9, 1.0)
            return float(-_np.mean(_np.sum(pp * _np.log(pp), axis=-1)))
        except Exception:
            return None

    def _fn(aux_or_ctx: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            x_probs = ((aux_or_ctx.get("x") or {}).get("probs"))
            y_probs = ((aux_or_ctx.get("y") or {}).get("probs"))
            hx = _H(x_probs) if x_probs is not None else None
            hy = _H(y_probs) if y_probs is not None else None
            if hx is not None:
                out["ent_x"] = hx
            if hy is not None:
                out["ent_y"] = hy
        except Exception:
            pass
        return out

    return _fn


def kl_divergence_xy():
    """Symmetric KL summary between x and y probs.

    Returns keys: {"kl_x_to_y": float, "kl_y_to_x": float}
    """

    import numpy as _np

    def _KL(p, q):
        try:
            p = _np.clip(_np.array(p, dtype=_np.float64), 1e-9, 1.0)
            q = _np.clip(_np.array(q, dtype=_np.float64), 1e-9, 1.0)
            return float(_np.mean(_np.sum(p * (_np.log(p) - _np.log(q)), axis=-1)))
        except Exception:
            return None

    def _fn(aux_or_ctx: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            x_probs = ((aux_or_ctx.get("x") or {}).get("probs"))
            y_probs = ((aux_or_ctx.get("y") or {}).get("probs"))
            if x_probs is not None and y_probs is not None:
                kxy = _KL(x_probs, y_probs)
                kyx = _KL(y_probs, x_probs)
                if kxy is not None:
                    out["kl_x_to_y"] = kxy
                if kyx is not None:
                    out["kl_y_to_x"] = kyx
        except Exception:
            pass
        return out

    return _fn


def sequence_hamming_xy(vocab: str = "ARNDCQEGHILKMFPSTWYV"):
    """Hamming distance between decoded x and y sequences (per-step)."""

    idx_to_aa = list(vocab)

    def _decode(p):
        try:
            idx = _np.argmax(_np.array(p), axis=-1)
            return "".join(idx_to_aa[i] for i in idx)
        except Exception:
            return None

    import numpy as _np

    def _fn(aux_or_ctx: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            x_probs = ((aux_or_ctx.get("x") or {}).get("probs"))
            y_probs = ((aux_or_ctx.get("y") or {}).get("probs"))
            sx = _decode(x_probs) if x_probs is not None else None
            sy = _decode(y_probs) if y_probs is not None else None
            if sx is not None and sy is not None and len(sx) == len(sy):
                hd = int(sum(1 for a, b in zip(sx, sy) if a != b))
                out["hamming_xy"] = float(hd)
                out["identity_xy"] = float((len(sx) - hd) / max(1, len(sx)))
        except Exception:
            pass
        return out

    return _fn


def per_position_entropy_xy():
    """Per-position entropy and summary stats for x and y.

    Returns: {
      "pos_ent_x": np.ndarray[L], "pos_ent_y": np.ndarray[L],
      "pos_ent_mean_x": float, "pos_ent_mean_y": float,
      "pos_ent_min_x": float, "pos_ent_min_y": float,
      "pos_ent_max_x": float, "pos_ent_max_y": float,
    }
    """
    import numpy as _np

    def _pos_entropy(p_xy):
        p = _np.clip(_np.array(p_xy, dtype=_np.float64), 1e-9, 1.0)
        H = -_np.sum(p * _np.log(p), axis=-1)
        return H

    def _fn(aux_or_ctx: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            x_probs = ((aux_or_ctx.get("x") or {}).get("probs"))
            y_probs = ((aux_or_ctx.get("y") or {}).get("probs"))
            if x_probs is not None:
                Hx = _pos_entropy(x_probs)
                out["pos_ent_x"] = Hx
                out["pos_ent_mean_x"] = float(_np.mean(Hx))
                out["pos_ent_min_x"] = float(_np.min(Hx))
                out["pos_ent_max_x"] = float(_np.max(Hx))
            if y_probs is not None:
                Hy = _pos_entropy(y_probs)
                out["pos_ent_y"] = Hy
                out["pos_ent_mean_y"] = float(_np.mean(Hy))
                out["pos_ent_min_y"] = float(_np.min(Hy))
                out["pos_ent_max_y"] = float(_np.max(Hy))
        except Exception:
            pass
        return out

    return _fn


def composition_charge_hydropathy_xy():
    """AA composition (20-dim), mean charge, and mean hydropathy for x and y.
    Uses expected composition from probs averaged across positions.
    """
    import numpy as _np

    # Kyte-Doolittle hydropathy for 20 AA in order ARNDCQEGHILKMFPSTWYV
    hydro = _np.array([
        -1.8, 4.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.2, 4.5,
         3.8, -3.9, 1.9, 2.8, -0.8, -0.7, -0.9, -0.4, -3.4, 4.2
    ], dtype=_np.float64)
    # Approximate net charge at neutral pH (K,R,H positive; D,E negative)
    charge = _np.array([
         1.0,  0.0, -1.0, -1.0,  0.0,  0.0, -1.0, -1.0,  0.0,  0.0,
          0.0,  1.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
    ], dtype=_np.float64)

    def _summaries(p):
        p = _np.array(p, dtype=_np.float64)
        comp = _np.mean(p, axis=0)  # average across positions
        mean_h = float(_np.sum(comp * hydro))
        mean_c = float(_np.sum(comp * charge))
        return comp, mean_c, mean_h

    def _fn(aux_or_ctx: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            x_probs = ((aux_or_ctx.get("x") or {}).get("probs"))
            y_probs = ((aux_or_ctx.get("y") or {}).get("probs"))
            if x_probs is not None:
                comp_x, charge_x, hyd_x = _summaries(x_probs)
                out["comp_x"] = comp_x
                out["charge_x"] = charge_x
                out["hydropathy_x"] = hyd_x
            if y_probs is not None:
                comp_y, charge_y, hyd_y = _summaries(y_probs)
                out["comp_y"] = comp_y
                out["charge_y"] = charge_y
                out["hydropathy_y"] = hyd_y
        except Exception:
            pass
        return out

    return _fn


def grad_norms_xy():
    """Gradient norm summaries for x and y if present in aux (added by optimizer)."""
    def _fn(aux_or_ctx: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            gx = ((aux_or_ctx.get("x") or {}).get("grad_norm"))
            gy = ((aux_or_ctx.get("y") or {}).get("grad_norm"))
            if gx is not None:
                out["grad_norm_x"] = float(gx)
            if gy is not None:
                out["grad_norm_y"] = float(gy)
        except Exception:
            pass
        return out

    return _fn

