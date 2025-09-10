from typing import Any, Dict


def _flatten(prefix: str, node: Any, out: Dict[str, Any]):
    if isinstance(node, dict):
        for k, v in node.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _flatten(f"{prefix}.{i}" if prefix else str(i), v, out)
    else:
        out[prefix] = node


def flatten_aux(aux: dict) -> Dict[str, Any]:
    """Flatten nested aux/metrics into dotted-key dict for validators and logging."""
    flat: Dict[str, Any] = {}
    _flatten("", aux, flat)
    return flat


