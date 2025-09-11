import argparse
from pathlib import Path
import sys
import os
import numpy as np
import jax
import jax.numpy as jnp

# Ensure local src is importable
_WS = "/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/src"
os.environ.setdefault("PYTHONPATH", _WS + os.pathsep + os.environ.get("PYTHONPATH", ""))
if _WS not in sys.path:
    sys.path.insert(0, _WS)

from mosaic_workflows.mhetase_scaffold import (
    make_workflow,
    build_boltz2_predict_fn_mhetase,
    CatalyticMotifLoss,
    MotifStructureRMSD,
    MotifDistogramCCE,
    HallucinationEntropyDist,
    HallucinationEntropyLeaky,
    SurfaceNonPolarLoss,
    NetChargeLoss,
)
import mosaic.losses.structure_prediction as sp


def _parse_pdb_backbone(pdb_path: Path, residue_numbers: list[int]) -> np.ndarray:
    res_set = set(int(x) for x in residue_numbers)
    # Collect N, CA, C per residue number (first occurrence)
    coords: dict[int, dict[str, np.ndarray | None]] = {rn: {"N": None, "CA": None, "C": None} for rn in res_set}
    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            try:
                atom = line[12:16].strip()
                resi = int(line[22:26])
                if resi in res_set and atom in ("N", "CA", "C"):
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    if coords[resi][atom] is None:
                        coords[resi][atom] = np.array([x, y, z], dtype=np.float32)
            except Exception:
                continue
    bb: list[np.ndarray] = []
    for rn in residue_numbers:
        c = coords[int(rn)]
        assert c["N"] is not None and c["CA"] is not None and c["C"] is not None, f"Missing backbone for residue {rn}"
        bb.append(np.stack([c["N"], c["CA"], c["C"]], axis=0))
    return np.stack(bb, axis=0)


def _mock_predict_fn(binder_len: int, num_bins: int = 64):
    bins = jnp.append(0.0, jnp.linspace(2.3125, 21.6875, num_bins - 1))
    key0 = jax.random.key(0)
    W = jax.random.normal(key0, (20, 3)) * 0.1
    def predict(probs, *, key, state):
        L = probs.shape[0]
        dir_vecs = probs @ W
        dir_vecs = dir_vecs / (jnp.linalg.norm(dir_vecs, axis=-1, keepdims=True) + 1e-6)
        steps = 3.8 * dir_vecs
        ca = jnp.cumsum(steps, axis=0)
        n = ca - 1.33 * dir_vecs
        c = ca + 1.33 * dir_vecs
        backbone = jnp.stack([n, ca, c, c], axis=1)
        d = jnp.sqrt(jnp.maximum(1e-6, jnp.sum((ca[:, None] - ca[None, :]) ** 2, axis=-1)))
        gamma = 1.5
        d_exp = d[..., None]
        logits = -gamma * (d_exp - bins[None, None, :]) ** 2
        class _Out:
            distogram_bins = bins
            distogram_logits = logits
            backbone_coordinates = backbone
            plddt = jnp.ones((L,)) * 0.9
        return _Out()
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-path", required=True)
    ap.add_argument("--pdb-residues", required=True, help="comma-separated residue indices, e.g. 225,448,484")
    ap.add_argument("--binder-len", type=int, required=True)
    ap.add_argument("--ser", type=int, required=True)
    ap.add_argument("--his", type=int, required=True)
    ap.add_argument("--asp", type=int, required=True)
    ap.add_argument("--use-boltz", action="store_true")
    ap.add_argument("--ligand-ccd", type=str, default=None)
    ap.add_argument("--ligand-smiles", type=str, default="CCO")
    ap.add_argument("--enzyme-chain", type=str, default="A")
    ap.add_argument("--ligand-chain", type=str, default="L")
    ap.add_argument("--recycling-steps", type=int, default=1)
    ap.add_argument("--num-sampling-steps", type=int, default=20)
    args = ap.parse_args()

    pdb_path = Path(args.pdb_path)
    residues = [int(x) for x in args.pdb_residues.split(",")]
    motif_bb = _parse_pdb_backbone(pdb_path, residues)

    motif_positions = {"ser": int(args.ser), "his": int(args.his), "asp": int(args.asp)}
    if args.use_boltz:
        predict = build_boltz2_predict_fn_mhetase(
            binder_len=int(args.binder_len),
            enzyme_chain=args.enzyme_chain,
            ligand_chain=args.ligand_chain,
            ligand_ccd=args.ligand_ccd,
            ligand_smiles=args.ligand_smiles,
            recycling_steps=int(args.recycling_steps),
            num_sampling_steps=int(args.num_sampling_steps),
        )
    else:
        predict = _mock_predict_fn(args.binder_len)

    wf = make_workflow(
        binder_len=int(args.binder_len),
        motif_positions=motif_positions,
        tmol_context={"coords": None},
        predict_fn=predict,
        motif_template_backbone=motif_bb.astype(np.float32),
        pae_on=bool(args.use_boltz),
        helix_weight=-0.3,
        use_leaky_entropy_soft=False,
        use_leaky_entropy_anneal=True,
    )

    anneal = [p for p in wf["phases"] if p["name"] == "anneal"][0]
    build_loss = anneal["build_loss"]()

    vocab = "ARNDCQEGHILKMFPSTWYV"
    allowed = np.ones((args.binder_len, 20), dtype=np.float32)
    allowed[motif_positions["ser"], :] = 0.0; allowed[motif_positions["ser"], vocab.index("S")] = 1.0
    allowed[motif_positions["his"], :] = 0.0; allowed[motif_positions["his"], vocab.index("H")] = 1.0
    allowed[motif_positions["asp"], :] = 0.0; allowed[motif_positions["asp"], vocab.index("D")] = 1.0
    probs0 = jnp.asarray(allowed)
    probs0 = probs0 / (probs0.sum(-1, keepdims=True) + 1e-8)

    # Build a single mock output for isolated losses
    # predict already assigned
    mock_out = predict(probs0, key=jax.random.key(0), state={})

    # Inspect motif backbone extraction
    ca = motif_bb[:, 1, :]
    dmat = np.sqrt(np.sum((ca[:, None, :] - ca[None, :, :]) ** 2, axis=-1))
    print({"motif_K": motif_bb.shape[0], "motif_backbone_shape": tuple(motif_bb.shape), "motif_ca_dists": dmat.tolist()})

    tests = []
    # 1) Motif identity (sequence-only)
    tests.append(("CatalyticMotifLoss", CatalyticMotifLoss(ser_pos=args.ser, his_pos=args.his, asp_pos=args.asp)))
    # 2) Motif structural RMSD (backbone)
    tests.append(("MotifStructureRMSD", MotifStructureRMSD(motif_positions=(args.ser, args.his, args.asp), target_backbone=motif_bb.astype(np.float32))))
    # 3) Motif distogram CCE
    tests.append(("MotifDistogramCCE", MotifDistogramCCE(motif_positions=(args.ser, args.his, args.asp), motif_template_ca=ca.astype(np.float32), max_pair_distance=20.0)))
    # 4) Hallucination entropy (standard and leaky)
    tests.append(("HallucinationEntropyDist", HallucinationEntropyDist(excluded_positions=(args.ser, args.his, args.asp), beta=10.0, max_contact_bin=5.0)))
    tests.append(("HallucinationEntropyLeaky", HallucinationEntropyLeaky(excluded_positions=(args.ser, args.his, args.asp), beta=10.0, max_contact_bin=5.0)))
    # 5) Structural priors
    tests.append(("WithinBinderContact", sp.WithinBinderContact(max_contact_distance=14.0, num_contacts_per_residue=4, min_sequence_separation=8)))
    tests.append(("PLDDTLoss", sp.PLDDTLoss()))
    tests.append(("HelixLoss", sp.HelixLoss()))
    # 6) Composition
    tests.append(("SurfaceNonPolarLoss", SurfaceNonPolarLoss()))
    tests.append(("NetChargeLoss", NetChargeLoss(target_max_charge=-5.0)))

    def run_test(name, loss_obj, needs_output=True):
        # CatalyticMotifLoss only takes (sequence, key), others take (sequence, output, key)
        is_seq_only = isinstance(loss_obj, CatalyticMotifLoss)
        if needs_output and not is_seq_only:
            def f(p):
                out = predict(p, key=jax.random.key(0), state={})
                v, _ = loss_obj(p, out, jax.random.key(0))
                return v
            v, aux = loss_obj(probs0, predict(probs0, key=jax.random.key(0), state={}), jax.random.key(0))
        else:
            def f(p):
                if is_seq_only:
                    v, _ = loss_obj(p, jax.random.key(0))
                else:
                    out = predict(p, key=jax.random.key(0), state={})
                    v, _ = loss_obj(p, out, jax.random.key(0))
                return v
            if is_seq_only:
                v, aux = loss_obj(probs0, jax.random.key(0))
            else:
                v, aux = loss_obj(probs0, predict(probs0, key=jax.random.key(0), state={}), jax.random.key(0))
        g = jax.grad(f)(probs0)
        gnorm = jnp.linalg.norm(g)
        ok = bool(jnp.isfinite(v) & jnp.isfinite(gnorm) & (gnorm > 0))
        print({"loss": name, "value": float(v), "grad_norm": float(gnorm), "finite": ok, "aux_keys": list(aux.keys()) if isinstance(aux, dict) else type(aux).__name__})
        return ok

    # Run individual tests; specify which need output
    results = []
    for (nm, lo) in tests:
        needs_out = not isinstance(lo, (CatalyticMotifLoss, NetChargeLoss))
        results.append((nm, run_test(nm, lo, needs_output=needs_out)))
    # Summary
    print({"summary": results})

    # Extra: Direct RMSD smoke test (no predictor), to validate implementation numerically
    # Build a minimal differentiable backbone where motif residues are translated by a small delta
    K = motif_bb.shape[0]
    L = int(args.binder_len)
    sel = (int(args.ser), int(args.his), int(args.asp))
    rmsd_loss = MotifStructureRMSD(motif_positions=sel, target_backbone=motif_bb.astype(np.float32))

    def rmsd_value(delta: float):
        # Construct backbone with zeros then insert motif_backbone + delta shift
        bb = jnp.zeros((L, 4, 3), dtype=jnp.float32)
        Q = jnp.asarray(motif_bb.astype(np.float32))  # [K,3,3]
        scale = jnp.asarray(1.0 + delta, dtype=jnp.float32)
        P_ncac = Q * scale  # isotropic scaling to avoid being removed by Kabsch
        # scatter into bb at selected indices (N,CA,C -> channels 0,1,2)
        idx = jnp.array(sel, dtype=jnp.int32)
        bb = bb.at[idx, 0, :].set(P_ncac[:, 0, :])
        bb = bb.at[idx, 1, :].set(P_ncac[:, 1, :])
        bb = bb.at[idx, 2, :].set(P_ncac[:, 2, :])
        class _Out:
            backbone_coordinates = bb
        v, _ = rmsd_loss(probs0, _Out(), jax.random.key(0))
        return v

    val0 = float(rmsd_value(0.0))
    val1 = float(rmsd_value(0.5))
    dval = float(jax.grad(rmsd_value)(0.5))
    print({"rmsd_smoke": {"rmsd_delta0": val0, "rmsd_delta0.5": val1, "grad_at_0.5": dval}})


if __name__ == "__main__":
    main()


