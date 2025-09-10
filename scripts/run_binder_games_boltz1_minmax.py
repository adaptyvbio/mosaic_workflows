import sys
from pathlib import Path
import numpy as np
import jax

# Ensure local packages are importable when running directly
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "src"))                 # for binder_games & mosaic_workflows
sys.path.append(str(root / "src" / "mosaic" / "src"))  # for mosaic

import mosaic.losses.structure_prediction as sp
from mosaic.losses.boltz import (
    load_boltz,
    make_binder_features,
    Boltz1Loss,
)

from mosaic_workflows import run_workflow, init_logits_boltzdesign1
from mosaic_workflows.transforms import temperature_on_logits, e_soft_on_logits, gradient_normalizer, zero_disallowed

from binder_games import (
    build_minmax_phase,
    make_minmax_loss,
)
from binder_games.analyzers import (
    saddle_gap_estimate,
    decode_sequences_xy,
    value_components,
)


def build_two_arg_boltz1_loss(*, joltz, features, helix_weight: float = -0.3):
    # Shared structure-prediction loss recipe for both players
    base_terms = (
        1.0 * sp.BinderTargetContact(contact_distance=21.0)
        + 1.0 * sp.WithinBinderContact(max_contact_distance=14.0, num_contacts_per_residue=4, min_sequence_separation=8)
        + helix_weight * sp.HelixLoss()
        + 0.0 * sp.PLDDTPerResidueReport()
    )

    loss_x = Boltz1Loss(
        joltz1=joltz,
        name="boltz1_x",
        loss=base_terms,
        features=features,
        recycling_steps=0,
        deterministic=True,
    )
    loss_y = Boltz1Loss(
        joltz1=joltz,
        name="boltz1_y",
        loss=base_terms,
        features=features,
        recycling_steps=0,
        deterministic=True,
    )

    # Competition: min_x max_y [L(x) - L(y)]
    return make_minmax_loss(loss_x, loss_y, weight_y=1.0)


def main():
    seed = 42
    binder_len = 20
    target_sequence = "MFEARLVQGSI"  # example target

    # Load model and build features (binder + target)
    joltz = load_boltz()
    features, _ = make_binder_features(
        binder_len=binder_len,
        target_sequence=target_sequence,
        use_msa=False,
        use_msa_server=False,
    )

    # Two-argument loss for x and y
    def build_loss():
        return build_two_arg_boltz1_loss(joltz=joltz, features=features, helix_weight=-0.3)

    # Phase: min–max with per-player transforms and analyzers
    phase = build_minmax_phase(
        name="minmax_boltz1",
        build_loss=build_loss,
        steps=120,
        schedule=lambda g, p: {"lr_x": 0.05, "lr_y": 0.05, "temperature": 1.0},
        transforms={
            "x": {
                "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
                "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])],
            },
            "y": {
                "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
                "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])],
            },
        },
        analyzers=[saddle_gap_estimate(), value_components(), decode_sequences_xy()],
        analyze_every=10,
    )

    x0 = init_logits_boltzdesign1(binder_len=binder_len, noise_scaling=0.1, rng=np.random.default_rng(seed))
    wf = {"phases": [phase], "binder_len": binder_len, "seed": seed, "initial_x": x0}
    out = run_workflow(wf)

    print("Best sequence (x):", out.get("best_sequence"))
    # Show last recorded analyzer metrics
    traj = out.get("trajectory", [])
    last = None
    for rec in reversed(traj):
        if rec.get("metrics"):
            last = rec["metrics"]
            break
    if last:
        keys = [k for k in ("gap", "value_x", "value_y", "seq_x", "seq_y") if k in last]
        print("Last metrics:", {k: last[k] for k in keys})


if __name__ == "__main__":
    main()



