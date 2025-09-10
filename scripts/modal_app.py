import os
import sys
from pathlib import Path
from typing import Any, Dict

import modal


# -------- Modal configuration --------

# Base image with Python + common scientific stack; adjust as needed.
# For GPU runs, set gpu=modal.gpu.A10G() on the function below and add CUDA wheels.
image = (
    modal.Image.debian_slim(python_version="3.12.0")
    .apt_install("git")
    .env({
        "BOLTZ_CACHE": "/root/.boltz",
        "JAX_PLATFORMS": "cuda"
    })
    .run_commands(
        "python -m pip install -U pip setuptools wheel && "
        # CUDA PyTorch for GPU (pulls CUDA runtime libs)
        "python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1 && "
        # JAX core and CUDA plugin (pin compatible versions)
        "python -m pip install --upgrade jax==0.7.1 && "
        "python -m pip install --upgrade jax-cuda12-plugin==0.7.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && "
        # PTX toolchain
        "python -m pip install nvidia-cuda-nvcc-cu12==12.8.93 && "
        # Core JAX ecosystem deps used by workflows
        "python -m pip install optax==0.2.4 dm-haiku>=0.0.13 flax>=0.10.2 ml-collections>=1.0.0 httpx>=0.28.1 && "
        # Git-only deps needed at runtime
        "python -m pip install git+https://github.com/escalante-bio/jablang.git && "
        "python -m pip install git+https://github.com/escalante-bio/esmj.git && "
        "python -m pip install git+https://github.com/escalante-bio/protenij.git"
    )
)


app = modal.App("adaptyv-boltzcraft", image=image)


boltz_cache = modal.Volume.from_name("boltz-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("results-boltzcraft", create_if_missing=True)

def _add_paths(workspace: Path):
    sys.path.append(str(workspace / "src"))


def _default_steps(total: int = 20) -> Dict[str, int]:
    """Split total steps into BD1-style phases.

    warmup: hallucination entropy + pLDDT only (stabilize backbone)
    soft:   full loss at temp=1, e_soft=0.8
    anneal: full loss with temperature decay
    """
    w = max(1, total // 5)
    s = max(1, total // 3)
    a = max(1, total - (w + s))
    return {"warmup": w, "soft": s, "anneal": a}


@app.function(gpu="A10G", timeout=3 * 60 * 60, volumes={"/root/.boltz": boltz_cache, "/results": results_vol}, secrets=[modal.Secret.from_name("github-token")])
def run_mhetase(
    *,
    binder_len: int = 20,
    motif_positions: Dict[str, Any] = {"ser": 3, "his": 10, "asp": 15},
    ligand: Dict[str, Any] = {"enzyme_chain": "A", "ligand_chain": "L", "smiles": "OCCOC(=O)c1ccc(cc1)C(=O)O"},
    total_steps: int = 20,
    seed: int = 0,
    motif_template_ca: list[list[float]] | None = None,
    motif_template_backbone: list[list[list[float]]] | None = None,
):
    """Launch the MHETase scaffolding workflow on Modal with a small budget.

    Parameters
    ----------
    binder_len : int
        Length of the designed enzyme chain.
    motif_positions : dict
        Positions for the catalytic triad (0-indexed): {"ser": i, "his": j, "asp": k}.
    ligand : dict
        Ligand spec for boltz2 predictor: keys include enzyme_chain, ligand_chain, smiles or ccd.
    total_steps : int
        Total optimization steps across warmup/design/refine.
    seed : int
        Random seed.
    """

    # Ensure Boltz cache uses a persisted Modal volume
    os.environ.setdefault("BOLTZ_CACHE", "/root/.boltz")
    Path(os.environ["BOLTZ_CACHE"]).mkdir(parents=True, exist_ok=True)

    # Fail fast if no GPU is visible to JAX
    try:
        import jax
        devs = jax.devices()
        if not any(getattr(d, "platform", None) == "gpu" for d in devs):
            raise RuntimeError(f"No GPU detected by JAX. Devices: {devs}")
    except Exception as e:
        raise RuntimeError(f"GPU preflight failed: {e}")

    # Clone repo at runtime (avoid uploading local code); use secret GITHUB_TOKEN if available
    repo_dir = Path("/repo").resolve()
    try:
        if not repo_dir.exists() or not any(repo_dir.iterdir()):
            import subprocess
            token = os.environ.get("GITHUB_TOKEN", "")
            url = "https://github.com/adaptyvbio/mosaic_workflows.git" if not token else f"https://x-access-token:{token}@github.com/adaptyvbio/mosaic_workflows.git"
            subprocess.run(["git", "clone", "--depth", "1", url, str(repo_dir)], check=True)
    except Exception:
        pass

    workspace = repo_dir
    _add_paths(workspace)

    from mosaic_workflows import run_workflow
    from mosaic_workflows.mhetase_scaffold import (
        build_boltz2_predict_fn_mhetase,
        make_workflow,
    )

    # Minimal context (predictor-only losses downstream)
    tmol_context = {"ligand": ligand, "coords": None}

    # Predictor with small sampling budget for quick iteration
    predict_fn = build_boltz2_predict_fn_mhetase(
        binder_len=binder_len,
        enzyme_chain=ligand.get("enzyme_chain", "A"),
        ligand_chain=ligand.get("ligand_chain", "L"),
        ligand_ccd=ligand.get("ccd"),
        ligand_smiles=ligand.get("smiles"),
        num_sampling_steps=80,
        recycling_steps=2,
    )

    wf = make_workflow(
        binder_len=binder_len,
        motif_positions=motif_positions,
        tmol_context=tmol_context,
        predict_fn=predict_fn,
        es_star_forced_bonds=None,
        motif_template_ca=(np := __import__("numpy")).array(motif_template_ca, dtype=float) if motif_template_ca is not None else None,
        motif_template_backbone=(np := __import__("numpy")).array(motif_template_backbone, dtype=float) if motif_template_backbone is not None else None,
        # Loss knobs (can be parameterized via CLI later if needed)
        ligand_metric=os.environ.get("LIGAND_METRIC", "iptm"),
        pae_on=os.environ.get("PAE_ON", "1") not in ("0", "false", "False"),
        helix_weight=float(os.environ.get("HELIX_WEIGHT", "-0.3")),
    )

    # Assign steps to phases: motif_lock + soft + anneal = 100 by default
    # Allocate 30/40/30 if total_steps=100; scale proportionally otherwise
    ml = max(1, int(0.3 * total_steps))
    an = max(1, int(0.3 * total_steps))
    sf = max(1, int(total_steps - (ml + an)))
    for p in wf["phases"]:
        if p["name"] == "motif_lock":
            p["steps"] = ml
        elif p["name"] == "soft":
            p["steps"] = sf
        elif p["name"] == "anneal":
            p["steps"] = an
        else:
            p["steps"] = max(1, total_steps // len(wf["phases"]))
    wf["seed"] = int(seed)
    wf["initial_x"] = (np := __import__("numpy")).random.randn(binder_len, 20).astype(np.float32) * 0.1

    out = run_workflow(wf)

    # Save outputs to results volume
    import json, time
    run_id = f"mhetase_{int(time.time())}_seed{seed}_len{binder_len}"
    out_dir = Path("/results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save best sequence and best_x
    (out_dir / "best_sequence.txt").write_text(str(out.get("best_sequence", "")))
    np.save(out_dir / "best_x.npy", out.get("best_x"))

    # Save trajectory as JSONL (losses and aux per step)
    traj = out.get("trajectory") or []
    with open(out_dir / "trajectory.jsonl", "w") as f:
        for rec in traj:
            # include phase name/metrics when available
            rec_s = {
                "step": int(rec.get("step", 0)),
                "metrics": rec.get("metrics", {}),
                "aux": rec.get("aux", {}),
            }
            f.write(json.dumps(rec_s, default=lambda o: float(o) if hasattr(o, "item") else None) + "\n")

    # Fold final best sequence with ligand using Boltz2 (3 recycle, 200 diffusion) and save coords + PDB
    try:
        import jax, jax.numpy as jnp
        probs_best = jax.nn.softmax(out["best_x"], axis=-1)
        key = jax.random.key(seed)
        from mosaic_workflows.mhetase_scaffold import (
            build_boltz2_predict_fn_mhetase as _bp,
            _build_mhetase_yaml,
        )
        from mosaic.losses.boltz2 import (
            load_features_and_structure_writer as _load_b2,
            set_binder_sequence as _set_b2_seq,
        )
        import gemmi
        pred = _bp(
            binder_len=binder_len,
            enzyme_chain=ligand.get("enzyme_chain", "A"),
            ligand_chain=ligand.get("ligand_chain", "L"),
            ligand_ccd=ligand.get("ccd"),
            ligand_smiles=ligand.get("smiles"),
            num_sampling_steps=200,
            recycling_steps=3,
        )
        pout = pred(probs_best, key=key, state={})
        coords = getattr(pout, "structure_coordinates", None)
        if coords is not None:
            np.save(out_dir / "coords.npy", np.array(coords))
            # Use writer to emit PDB
            es_yaml = _build_mhetase_yaml(
                binder_len=binder_len,
                enzyme_chain=ligand.get("enzyme_chain", "A"),
                ligand_chain=ligand.get("ligand_chain", "L"),
                ligand_ccd=ligand.get("ccd"),
                ligand_smiles=ligand.get("smiles"),
                bond_constraints=None,
            )
            features_w, writer = _load_b2(es_yaml, cache=Path(os.environ.get("BOLTZ_CACHE", "/root/.boltz")).expanduser())
            hard_seq = (probs_best == probs_best.max(axis=-1, keepdims=True)).astype(np.float32)
            features_w = _set_b2_seq(hard_seq, features_w)
            st = writer(coords)
            # Write mmCIF as a robust fallback
            cif = gemmi.cif.Document()
            block = st.make_mmcif_block()
            cif.add_block(block)
            with open(out_dir / "final.cif", "w") as fh:
                fh.write(cif.as_string())
    except Exception as e:
        (out_dir / "warn.txt").write_text(f"structure save failed: {e}")

    print({"results_dir": str(out_dir)})


@app.local_entrypoint()
def main(
    workflow: str = "mhetase",
    binder_len: int = 20,
    ser: int = 3,
    his: int = 10,
    asp: int = 15,
    oxyanion: str | None = None,
    ligand_smiles: str = "OCCOC(=O)c1ccc(cc1)C(=O)O",
    total_steps: int = 20,
    seed: int = 0,
    pdb_path: str | None = None,
    pdb_residues: str | None = None,
    pdb_oxyanion_residues: str | None = None,
    ligand_metric: str = "iptm",
    pae_on: bool = True,
    helix_weight: float = -0.3,
    random_len_min: int = 0,
    random_len_max: int = 0,
):
    """Local entrypoint to kick off a workflow on Modal.

    Examples (local):
      modal run scripts.modal_app --workflow mhetase --binder-len 20 --ser 3 --his 10 --asp 15 --total-steps 20
    """
    if workflow == "mhetase":
        # Optional random binder length sampling
        if int(random_len_max) > int(random_len_min) and int(random_len_min) > 0:
            import random
            binder_len = random.randint(int(random_len_min), int(random_len_max))
        ligand = {"enzyme_chain": "A", "ligand_chain": "L", "smiles": ligand_smiles}
        motif_ca = None
        motif_bb = None
        if pdb_path and pdb_residues:
            # parse CA coords for given residue numbers (comma-separated)
            residues = [int(x.strip()) for x in pdb_residues.split(",") if x.strip()]
            oxy_residues = []
            if pdb_oxyanion_residues:
                oxy_residues = [int(x.strip()) for x in pdb_oxyanion_residues.split(",") if x.strip()]
            ca = []
            bb: dict[str, list[tuple[int, list[float]]]] = {"N": [], "CA": [], "C": []}
            with open(pdb_path, "r") as f:
                for line in f:
                    if not line.startswith("ATOM"): continue
                    atname = line[12:16].strip()
                    if atname not in ("N", "CA", "C"):
                        continue
                    # residue sequence number in columns 22-26
                    try:
                        resseq = int(line[22:26].strip())
                    except Exception:
                        continue
                    if resseq in residues or resseq in oxy_residues:
                        x = float(line[30:38].strip()); y = float(line[38:46].strip()); z = float(line[46:54].strip())
                        if atname == "CA":
                            ca.append((resseq, [x, y, z]))
                        elif atname in bb:
                            bb[atname].append((resseq, [x, y, z]))
            # order by residues as provided
            ca_map = {r: xyz for (r, xyz) in ca}
            N_map = {r: xyz for (r, xyz) in bb["N"]}
            CA_map = ca_map
            C_map = {r: xyz for (r, xyz) in bb["C"]}
            motif_ca = [ca_map[r] for r in residues if r in ca_map]
            # append oxyanion residues after catalytic ones, in provided order
            for r in oxy_residues:
                if r in ca_map:
                    motif_ca.append(ca_map[r])
            # backbone in order N, CA, C per residue (only if all present)
            bb_list = []
            for r in residues + oxy_residues:
                if (r in N_map) and (r in CA_map) and (r in C_map):
                    bb_list.append([N_map[r], CA_map[r], C_map[r]])
            motif_bb = bb_list if len(bb_list) == len(motif_ca) else None
        # Build motif positions dict, including optional oxyanion binder indices
        motif_pos: dict[str, int | list[int]] = {"ser": ser, "his": his, "asp": asp}
        if oxyanion:
            try:
                motif_pos["oxyanion"] = [int(x.strip()) for x in oxyanion.split(",") if x.strip()]
            except Exception:
                pass
        # Plumb loss knobs via environment to keep function signature minimal on Modal
        os.environ["LIGAND_METRIC"] = str(ligand_metric)
        os.environ["PAE_ON"] = "1" if pae_on else "0"
        os.environ["HELIX_WEIGHT"] = str(helix_weight)
        run_mhetase.remote(
            binder_len=binder_len,
            motif_positions=motif_pos,
            ligand=ligand,
            total_steps=total_steps,
            seed=seed,
            motif_template_ca=motif_ca,
            motif_template_backbone=motif_bb,
        )
    else:
        raise ValueError(f"Unknown workflow: {workflow}")


