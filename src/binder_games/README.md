binder_games
============

Minimal Mosaic-style game optimizers and helpers for robust binder design.

Design philosophy: loss-first composition, plain dicts/lists, small pure functions, and reusing the `mosaic_workflows` optimizer interface and phase/workflow schema.

What you import
---------------
```
from binder_games import (
  # optimizers
  minmax_logits, alternating_br_logits, stackelberg_logits, extragradient_minmax_logits,

  # builders
  build_minmax_phase, build_stackelberg_phase, build_multi_adversary_phase,

  # transforms
  temperature_on_logits_xy, gradient_normalizer_xy, hard_one_hot_xy,

  # losses
  make_minmax_loss, make_multi_adversary_loss, make_dro_loss,

  # analyzers & validators
  saddle_gap_estimate, value_components, decode_sequences_xy,
  gap_threshold, worst_case_threshold,
)
```

Optimizer interface
-------------------
All optimizers follow the Mosaic-Workflows signature and return `(x_logits, best_x_logits, trajectory_or_none)`.
```
def optimizer(*, loss_function, x, n_steps, key=None,
              schedule=None, transforms=None, trajectory_fn=None,
              aux_context=None, update_loss_state: bool = False, **kwargs):
    ...
```

Two-player loss
---------------
```
def build_loss():
    # returns two-argument loss: (x_probs, y_probs, key) -> (value, aux)
    return make_minmax_loss(loss_x, loss_y, weight_y=1.0)
```

Phase builder
-------------
```
phase = build_minmax_phase(
  name="minmax",
  build_loss=build_loss,
  steps=150,
  schedule=lambda g,p: {"lr_x":0.1, "lr_y":0.1, "temperature_x":1.0, "temperature_y":1.0},
  transforms={
    "x": {"pre_logits": [temperature_on_logits_xy()["x"]["pre_logits"][0]]},
    "y": {"pre_logits": [temperature_on_logits_xy()["y"]["pre_logits"][0]]},
  },
)
```

Notes
-----
- You can pass flat transform chains (applied to both players) or nested per-player dicts.
- Schedules may provide per-player knobs (e.g., `lr_x`, `lr_y`, `temperature_x`, `temperature_y`).
- Analyzers are safe: failures do not crash design; results merge into trajectory records.



