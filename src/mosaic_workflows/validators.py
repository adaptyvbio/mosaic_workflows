from typing import Any, Dict, Tuple


def threshold_filter(thresholds: dict):
    """
    thresholds: nested dict like {"confidence.plddt": {"min": 50.0}, "clashes": {"max": 5}}
    Returns a callable: metrics -> (pass: bool, details: dict)
    """
    def _fn(metrics: dict) -> Tuple[bool, Dict[str, Any]]:
        passed: bool = True
        details: Dict[str, Any] = {}
        for k, rule in (thresholds or {}).items():
            # nested access using dot-keys
            cur_any: Any = metrics
            for part in k.split("."):
                if isinstance(cur_any, dict) and part in cur_any:
                    cur_any = cur_any[part]
                else:
                    cur_any = None
                    break
            v = cur_any
            ok = True
            if v is None:
                ok = False
            else:
                if "min" in rule:
                    ok = ok and (v >= rule["min"])
                if "max" in rule:
                    ok = ok and (v <= rule["max"])
            passed = passed and ok
            details[k] = {"value": v, "ok": ok, **rule}
        return passed, details
    return _fn


