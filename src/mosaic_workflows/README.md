mosaic_workflows
================

A minimal, Mosaic-style workflow layer for protein design. It mirrors Mosaic’s design philosophy: loss-first composition, small pure functions, plain dicts/lists, and optional stateful losses.

Vocabulary
----------
- Workflow: one trajectory’s design run; list of phases and global settings.
- Phase: one optimizer block; its own loss builder, schedule, transforms, validators/analyzers, and step count. Phases run sequentially.
- Segment: an intra-phase schedule slice returned by the phase’s schedule; internal only.
- Outer loop: repeat workflows across trajectories; sits outside this API.

What you import
---------------
```
from mosaic_workflows import (
  run_workflow,
  # optimizers
  adamw_logits, sgd_logits, simplex_APGM_adapter, gradient_MCMC_adapter, rao_gumbel_adapter,
  # transforms
  temperature_on_logits, e_soft_on_logits, scale_logits,
  token_restrict, token_restrict_post_logits,
  zero_disallowed, gradient_normalizer, hard_one_hot,
  # validators & callbacks & init
  threshold_filter, checkpoint, memory_housekeeping, init_logits_boltzdesign1,
)
```

Optimizer interface
-------------------
All optimizers share one signature.
```
def optimizer(*, loss_function, x, n_steps, key=None,
              schedule=None, transforms=None, trajectory_fn=None,
              aux_context=None, update_loss_state: bool = False, **kwargs) -> tuple:
    """Returns (x_logits, best_x_logits, trajectory_or_none)."""
```
- loss_function: Mosaic LossTerm | LinearCombination; expects probabilities unless noted.
- x: logits (N x 20) unless noted; optimizers apply softmax internally when needed.
- schedule: callable (global_step, phase_step) -> dict of knobs (e.g., lr, temperature, e_soft, stepsize, scale, proposal_temp).
- transforms: dict of callable lists (all optional):
  - pre_logits, post_logits: fn(logits, ctx) -> logits
  - pre_probs, post_probs: fn(probs, ctx) -> probs
  - grad: fn(grad, ctx) -> grad
- update_loss_state: when True, updates stateful loss modules using Mosaic’s update_states.

Transforms provided
-------------------
- temperature_on_logits(): divide logits by schedule["temperature"].
- e_soft_on_logits(): multiply logits by schedule["e_soft"].
- scale_logits(x, alpha): multiply logits by alpha.
- token_restrict(allowed_tokens, avoid_residues): zero out disallowed probabilities.
- token_restrict_post_logits(...): add -inf to disallowed logits.
- gradient_normalizer(mode="l2"|"clip"|"per_chain"): normalize or clip gradients.
- zero_disallowed(...): mask gradients for disallowed tokens.
- hard_one_hot(): convert to crisp logits (+10/-10) via argmax.

Step-time transforms vs loss-internal transformations (Mosaic)
--------------------------------------------------------------
There are two distinct kinds of "transformations" in this codebase:

1) Step-time transforms (here, in mosaic_workflows)
   - Scope: change how optimization steps are taken (sampling, masking, scaling, gradient shaping).
   - Where: configured in a phase's `transforms` dict under keys `pre_logits`, `pre_probs`, `grad`, `post_logits`.
   - Inputs: receive the current array (logits, probs, or grad) and a small context `ctx` that includes the phase `schedule`.
   - Timing: applied every optimizer step; fully schedule-driven and optimizer-agnostic.
   - Examples: `temperature_on_logits`, `e_soft_on_logits`, `token_restrict(_post_logits)`, `gradient_normalizer`, `hard_one_hot`.

2) Loss-internal transformations (in mosaic.losses.transformations)
   - Scope: change what is being optimized by wrapping/modifying a Mosaic LossTerm.
   - Where: used inside `build_loss()` when composing the objective; they are themselves `LossTerm`s and combine via `LinearCombination`.
   - Inputs: operate within the loss evaluation (on sequences/probabilities and the loss' internals); may participate in autograd.
   - Timing: fixed for the duration of a phase unless the loss is stateful and updated via `update_loss_state=True`.
   - Examples: `SetPositions` (freeze certain residues), `ClippedLoss`, `FixedPositionsPenalty`, `ClippedGradient`, `NormedGradient`.

Guidance
--------
- Use mosaic_workflows step-time transforms to implement schedule-controlled behaviors that should adjust during optimization (e.g., temperature annealing, gradient clipping, feasibility masks).
- Use mosaic loss-internal transformations to encode design semantics and constraints as part of the objective itself (e.g., allowed positions, penalties, loss clipping).
- Do not mix them for the same concern: if a behavior must anneal with steps, prefer step-time transforms; if it defines the target objective, prefer loss-internal transformations.

Validators & callbacks
----------------------
- threshold_filter({"metric.path": {"min": x}|{"max": y}, ...}) -> (bool, details)
- checkpoint(output_dir, save_interval=50, save_logits=True)
- memory_housekeeping()

Workflow and Phase schema (plain dicts)
---------------------------------------
```
phase = {
  "name": "warmup",
  "build_loss": lambda: loss_term,   # returns LossTerm | LinearCombination
  "optimizer": adamw_logits,
  "steps": 30,
  "schedule": lambda g,p: {"learning_rate": 0.1, "temperature": 1.0, "e_soft": 0.8},
  "transforms": {
    "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
    "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])],
  },
  "validators": [],
  "analyzers": [],
  "analyze_every": 10,
}

workflow = {
  "phases": [phase1, phase2, ...],
  "binder_len": 60,
  "seed": 42,
  "initial_x": x0,
  "callbacks": [checkpoint("./checkpoints", 50, True), memory_housekeeping()],
}
```

Stateful losses
---------------
Mosaic’s stateful-loss pattern is supported. When an optimizer runs with update_loss_state=True, it will call the same update_states(aux, loss) logic Mosaic uses. Use this to interleave predictor recycling with optimization or to cache/update heavy intermediates.

Example: BD1 control (Boltz1, Mosaic-style)
---------------------------------------------
```
import numpy as np
import mosaic.losses.structure_prediction as sp
from mosaic.losses.boltz import load_boltz, make_binder_features, Boltz1Loss
from mosaic_workflows import (
  run_workflow, adamw_logits, sgd_logits,
  temperature_on_logits, e_soft_on_logits, gradient_normalizer, zero_disallowed,
  init_logits_boltzdesign1,
)

binder_len, seed = 60, 42
boltz1 = load_boltz()

def build_loss():
  feats, _ = make_binder_features(
    binder_len=binder_len,
    target_sequence="...",
    use_msa=False,
    use_msa_server=False,
  )
  terms = (
    1.0 * sp.BinderTargetContact(contact_distance=20.0) +
    1.0 * sp.WithinBinderContact(
      max_contact_distance=14.0,
      num_contacts_per_residue=2,
      min_sequence_separation=8,
    ) +
    (-0.3) * sp.HelixLoss()
  )
  return Boltz1Loss(joltz1=boltz1, name="boltz1", loss=terms, features=feats, recycling_steps=0, deterministic=True)

warmup = {
  "name": "warmup",
  "build_loss": build_loss,
  "optimizer": adamw_logits,
  "steps": 30,
  "schedule": lambda g,p: {"learning_rate": 0.1, "temperature": 1.0, "e_soft": 0.8, "alpha": 2.0},
  "transforms": {
    "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
    "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])],
  },
}

def three_stage(g,p):
  if p < 75: return {"lr": 0.1, "temperature": 1.0}
  if p < 125:
    frac = (p - 75) / 50; temp = 1.0 - (1.0 - 0.01) * (frac**2)
    return {"lr": 0.1, "temperature": temp}
  return {"lr": 0.1, "temperature": 0.01}

design = {
  "name": "design",
  "build_loss": build_loss,
  "optimizer": sgd_logits,
  "steps": 130,
  "schedule": three_stage,
  "transforms": {
    "pre_logits": [temperature_on_logits()],
    "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])],
  },
}

x0 = init_logits_boltzdesign1(binder_len=binder_len, noise_scaling=0.1, rng=np.random.default_rng(seed))
wf = {"phases": [warmup, design], "binder_len": binder_len, "seed": seed, "initial_x": x0}
result = run_workflow(wf)
```

Testing
-------
A minimal test exists at tests/test_mosaic_workflows.py using a toy loss to exercise the runner and an optimizer.

TODO (adaptyv_boltzcraft)
-------------------------
- Implement true Rao-Blackwellized Gumbel-Softmax, ST-Gumbel, and ZGR (current rao_gumbel_adapter is a proxy).
- Add analyzer utilities: sequence_analyzer, clash_analyzer, interface_analyzer, dssp_analyzer.
- Add predictor/IO helpers: predict_boltz1, predict_af2, relax_pyrosetta, mpnn_redesign.
- Extend loss coverage: termini_disto_loss, termini_loss, tim_loss, binding_labels_loss; EC loss wrapper via ec_utils.py; verify PAE/iPTM mapping completeness.
- Aux logging convenience: helper to flatten LinearCombination aux into a single dict keyed by loss names. (flatten_aux added)
- Improve callbacks: richer checkpoint payloads (logits, metrics) and periodic summaries.
- Example notebooks demonstrating BD1 control, AF2 guidance, MCMC finetuning.
- Environment notes: ensure JAX/EQX/Mosaic deps available; clear away type warnings.


