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

End-to-end example (Boltz1 min–max)
-----------------------------------
```
import numpy as np
import mosaic.losses.structure_prediction as sp
from mosaic.losses.boltz import load_boltz, make_binder_features, Boltz1Loss
from mosaic_workflows import run_workflow, init_logits_boltzdesign1
from mosaic_workflows.transforms import temperature_on_logits, e_soft_on_logits, gradient_normalizer, zero_disallowed
from binder_games import build_minmax_phase, make_minmax_loss
from binder_games.analyzers import saddle_gap_estimate, decode_sequences_xy

seed, binder_len = 42, 20
joltz = load_boltz()
features, _ = make_binder_features(binder_len=binder_len, target_sequence="MFEARLVQGSI", use_msa=False, use_msa_server=False)

def two_arg_loss():
  base = (
    1.0 * sp.BinderTargetContact(contact_distance=21.0)
    + 1.0 * sp.WithinBinderContact(max_contact_distance=14.0, num_contacts_per_residue=4, min_sequence_separation=8)
    + (-0.3) * sp.HelixLoss()
  )
  loss_x = Boltz1Loss(joltz1=joltz, name="x", loss=base, features=features, recycling_steps=0, deterministic=True)
  loss_y = Boltz1Loss(joltz1=joltz, name="y", loss=base, features=features, recycling_steps=0, deterministic=True)
  return make_minmax_loss(loss_x, loss_y)

phase = build_minmax_phase(
  name="minmax_boltz1",
  build_loss=two_arg_loss,
  steps=120,
  schedule=lambda g,p: {"lr_x": 0.05, "lr_y": 0.05, "temperature": 1.0},
  transforms={
    "x": {"pre_logits": [temperature_on_logits(), e_soft_on_logits()], "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])]},
    "y": {"pre_logits": [temperature_on_logits(), e_soft_on_logits()], "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])]},
  },
  analyzers=[saddle_gap_estimate(), decode_sequences_xy()],
  analyze_every=10,
)

x0 = init_logits_boltzdesign1(binder_len=binder_len, noise_scaling=0.1, rng=np.random.default_rng(seed))
out = run_workflow({"phases": [phase], "binder_len": binder_len, "seed": seed, "initial_x": x0})
print("Best sequence (x):", out.get("best_sequence"))
```

See also: `scripts/run_binder_games_boltz1_minmax.py`.



