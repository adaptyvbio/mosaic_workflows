from typing import Callable, Dict, Any
import numpy as np

_VOCAB = "ARNDCQEGHILKMFPSTWYV"
_IDX = {a: i for i, a in enumerate(_VOCAB)}


def sequence_to_onehot_logits(sequence: str, on: float = 10.0, off: float = -10.0) -> np.ndarray:
    L = len(sequence)
    x = np.full((L, 20), off, dtype=np.float32)
    for i, aa in enumerate(sequence):
        x[i, _IDX[aa]] = on
    return x


def make_predict_only_workflow(*, sequence: str, predict_fn: Callable[[Any], Any]) -> Dict[str, Any]:
    def build_loss():
        class _ZeroLoss:
            def __call__(self, probs, *, key=None, **kwds):
                out = predict_fn(probs, key=key, state={})
                return 0.0, {"predict": out}

        return _ZeroLoss()

    def _noop_optimizer(**kwargs):
        x = kwargs["x"]
        return x, x, []

    return {
        "phases": [
            {
                "name": "fold",
                "build_loss": build_loss,
                "optimizer": _noop_optimizer,
                "steps": 1,
                "analyzers": [],
                "validators": [],
                "analyze_every": 1,
            }
        ],
        "binder_len": len(sequence),
        "seed": 0,
        "initial_x": sequence_to_onehot_logits(sequence),
    }


