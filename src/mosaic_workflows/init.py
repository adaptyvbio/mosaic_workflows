import numpy as np

# Gumbel logits init to match BoltzDesign1-style initialization
def init_logits_boltzdesign1(binder_len: int, noise_scaling: float, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng()
    # True Gumbel(0,1) logits, scaled
    g = rng.gumbel(loc=0.0, scale=1.0, size=(binder_len, 20)).astype(np.float32)
    return (noise_scaling * g).astype(np.float32)


