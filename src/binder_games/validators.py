from typing import Any, Dict


def gap_threshold(spec: Dict[str, Dict[str, float]]):
    """Validate that estimated gap stays within thresholds.

    Example spec: {"gap": {"max": 0.5}}
    Returns callable(aux) -> (bool, details)
    """

    def _fn(aux: Dict[str, Any]):
        try:
            g = float((aux or {}).get("gap", float("inf")))
            lim = spec.get("gap") or {}
            ok_min = True if ("min" not in lim) else (g >= float(lim["min"]))
            ok_max = True if ("max" not in lim) else (g <= float(lim["max"]))
            ok = bool(ok_min and ok_max)
            return ok, {"gap": g, "limits": lim}
        except Exception:
            return False, {"error": "gap_threshold evaluation failed"}

    return _fn


def worst_case_threshold(spec: Dict[str, Dict[str, float]]):
    """Generic threshold over aux keys.

    Example: {"x/plddt": {"min": 0.7}, "y/contact": {"max": 5.0}}
    """

    def _get(aux: Dict[str, Any], path: str):
        try:
            parts = path.split("/")
            cur: Any = aux
            for p in parts:
                cur = cur.get(p)
                if cur is None:
                    return None
            return cur
        except Exception:
            return None

    def _fn(aux: Dict[str, Any]):
        details: Dict[str, Any] = {}
        ok = True
        for k, lim in (spec or {}).items():
            v = _get(aux, k)
            details[k] = v
            if v is None:
                ok = False
                continue
            if "min" in lim and not (v >= float(lim["min"])):
                ok = False
            if "max" in lim and not (v <= float(lim["max"])):
                ok = False
        return ok, details

    return _fn


