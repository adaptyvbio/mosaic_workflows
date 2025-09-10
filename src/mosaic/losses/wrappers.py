import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Literal

from ..common import LossTerm


class RiskWrappedLoss(LossTerm):
    """Generic risk-aggregation wrapper around a base LossTerm.

    Evaluates the base loss on K relaxed samples of the input sequence and
    aggregates per-sample losses with a chosen risk functional.

    This wrapper keeps the LossTerm signature and composes with other
    loss-level wrappers (e.g., SetPositions, ClippedGradient, NormedGradient).
    """

    base: LossTerm
    risk_type: Literal["mean", "cvar", "entropic", "mean_variance"] = "cvar"
    num_samples: int = 16
    alpha: float = 0.3        # for CVaR
    eta: float = 1.0          # for entropic risk
    lam: float = 0.0          # for mean-variance
    temperature: float = 1.0  # for Gumbel-Softmax relaxation

    def _sample_relaxed(self, probs, key):
        logits = jnp.log(jnp.clip(probs, 1e-9, 1.0))

        def one(k):
            g = -jnp.log(-jnp.log(jax.random.uniform(jax.random.fold_in(key, k), logits.shape) + 1e-8) + 1e-8)
            y = (logits + g) / jnp.maximum(1e-6, self.temperature)
            return jax.nn.softmax(y, axis=-1)

        return jax.vmap(one)(jnp.arange(self.num_samples))

    def _aggregate(self, losses):
        if self.risk_type == "mean":
            return jnp.mean(losses)
        if self.risk_type == "cvar":
            # Compute a static Python int for slicing under JIT
            import numpy as _np  # local to avoid global dependency
            k = max(1, int(_np.ceil(float(self.alpha) * int(self.num_samples))))
            return jnp.mean(jnp.sort(losses)[-k:])
        if self.risk_type == "entropic":
            # numerically stable log-sum-exp
            m = jnp.min(losses)
            return (1.0 / self.eta) * (jnp.log(jnp.mean(jnp.exp(self.eta * (losses - m)))) + self.eta * m)
        if self.risk_type == "mean_variance":
            m = jnp.mean(losses)
            s = jnp.std(losses)
            return m + self.lam * s
        return jnp.mean(losses)

    def __call__(self, seq, *, key, **kwds):
        # draw relaxed samples around seq probabilities and evaluate base.
        samples = self._sample_relaxed(seq, key)

        def eval_one(xk):
            v, aux = self.base(xk, key=key, **kwds)
            return v, aux

        vals, auxs = jax.vmap(eval_one, in_axes=0)(samples)
        value = self._aggregate(vals)
        aux = {
            "risk": {
                "type": self.risk_type,
                "value": value,
                "mean": jnp.mean(vals),
                "std": jnp.std(vals),
                "alpha": self.alpha,
                "eta": self.eta,
                "lam": self.lam,
                "num_samples": int(self.num_samples),
                "temperature": float(self.temperature),
            },
            "base_aux": auxs,
        }
        return value, aux


