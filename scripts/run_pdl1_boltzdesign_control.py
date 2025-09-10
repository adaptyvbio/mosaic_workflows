import sys
from pathlib import Path
import numpy as np
import jax

# Ensure local packages are importable when running directly
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "src"))                 # for mosaic_workflows
sys.path.append(str(root / "src" / "mosaic" / "src"))  # for mosaic

import mosaic.losses.structure_prediction as sp
from mosaic.losses.boltz import load_boltz, make_binder_features, Boltz1Loss
from mosaic_workflows import run_workflow, adamw_logits, sgd_logits, zgr_adapter, init_logits_boltzdesign1
from mosaic_workflows.transforms import temperature_on_logits, e_soft_on_logits, gradient_normalizer, zero_disallowed
from mosaic_workflows.optimizers import semi_greedy_adapter


def make_build_loss(joltz, features, binder_len: int, target_sequence: str, *,
                    intra_k: int, intra_seqsep: int,
                    intra_contact_distance: float, inter_contact_distance: float,
                    helix_weight: float,
                    plddt_weight: float = 0.1,
                    bb_pae_weight: float = 0.4,
                    bt_pae_weight: float = 0.1,
                    rg_weight: float = 0.0,
                    deterministic: bool = True,
                    recycling_steps: int = 0):
    def build_loss():
        losses = (
            1.0 * sp.WithinBinderContact(
                max_contact_distance=intra_contact_distance,
                num_contacts_per_residue=intra_k,
                min_sequence_separation=intra_seqsep,
            )
            + 1.0 * sp.BinderTargetContact(contact_distance=inter_contact_distance)
            + helix_weight * sp.HelixLoss()
        )
        if plddt_weight != 0.0:
            losses = losses + plddt_weight * sp.PLDDTLoss() + 0.0 * sp.PLDDTPerResidueReport()
        if bb_pae_weight != 0.0:
            losses = losses + bb_pae_weight * sp.WithinBinderPAE()
        if bt_pae_weight != 0.0:
            losses = losses + bt_pae_weight * sp.BinderTargetPAE()
        if rg_weight != 0.0:
            losses = losses + rg_weight * sp.DistogramRadiusOfGyration()

        return Boltz1Loss(
            joltz1=joltz,
            name="boltz1",
            loss=losses,
            features=features,
            recycling_steps=recycling_steps,
            deterministic=deterministic,
        )

    return build_loss


def main():
    seed = 42
    binder_len = 20
    target_sequence = "MFEARLVQGSI"

    joltz = load_boltz()
    features, _ = make_binder_features(
        binder_len=binder_len,
        target_sequence=target_sequence,
        use_msa=False,
        use_msa_server=False,
    )

    build_loss = make_build_loss(
        joltz,
        features,
        binder_len,
        target_sequence,
        intra_k=4,
        intra_seqsep=8,
        intra_contact_distance=14.0,
        inter_contact_distance=21.0,
        helix_weight=-0.3,
        plddt_weight=0.0,
        bb_pae_weight=0.0,
        bt_pae_weight=0.0,
        deterministic=True,
        recycling_steps=0,
    )

    warmup = {
        "name": "warmup",
        "build_loss": build_loss,
        "optimizer": adamw_logits,
        "steps": 3,
        "schedule": lambda g, p: {"learning_rate": 0.2, "temperature": 1.0, "e_soft": 0.8},
        "transforms": {
            "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
            "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])],
        },
        "analyze_every": 1,
    }

    soft = {
        "name": "soft",
        "build_loss": build_loss,
        "optimizer": sgd_logits,
        "steps": 5,
        "schedule": lambda g, p: {"lr": 0.1, "temperature": 1.0, "e_soft": 0.8},
        "transforms": {
            "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
            "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])],
        },
    }

    def anneal_sched(g, p):
        frac = p / 45.0
        temp = 1.0 - (1.0 - 0.01) * (frac**2)
        return {"lr": 0.1, "temperature": max(0.01, float(temp))}

    anneal = {
        "name": "anneal",
        "build_loss": build_loss,
        "optimizer": sgd_logits,
        "steps": 5,
        "schedule": anneal_sched,
        "transforms": {
            "pre_logits": [temperature_on_logits()],
            "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])],
        },
    }

    hard = {
        "name": "hard",
        "build_loss": build_loss,
        "optimizer": zgr_adapter,
        "steps": 1,
        "schedule": lambda g, p: {"lr": 0.1, "temperature": 0.01},
        "transforms": {"pre_logits": [temperature_on_logits()]},
    }

    # Optional discrete finisher using reporter pLDDT; set steps>0 to enable
    semi = {
        "name": "semi-greedy",
        "build_loss": build_loss,
        "optimizer": semi_greedy_adapter,
        "steps": 2,
        "schedule": lambda g, p: {"proposals_per_step": 10, "position_weighting": "1-plddt"},
    }

    phases = [warmup, soft, anneal, hard] + ([semi] if semi["steps"] > 0 else [])

    x0 = init_logits_boltzdesign1(binder_len=binder_len, noise_scaling=0.1, rng=np.random.default_rng(seed))
    wf = {"phases": phases, "binder_len": binder_len, "seed": seed, "initial_x": x0}
    out = run_workflow(wf)
    print("Best sequence:", out["best_sequence"])


if __name__ == "__main__":
    main()


