import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

import modal


# Base image modeled after adaptyv_bindcraft modal app with extras for Mosaic
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "wget",
        "git",
        "aria2",
        "ffmpeg",
        "build-essential",
        "libxml2-dev",
        "libxslt1-dev",
        "cmake",
    )
    .pip_install(
        # BindCraft stack
        "pdb-tools==2.4.8",
        "ffmpeg-python==0.2.0",
        "plotly==5.18.0",
        "kaleido==0.2.1",
        "pyarrow",
        "fastparquet",
        "boto3",
        "python-dotenv",
        "loguru",
        "openmm>=7.7.0",
        "mdtraj",
        "biopython",
        "freesasa",
        "scipy",
        "scikit-learn",
    )
    .pip_install("git+https://github.com/openmm/pdbfixer.git")
    .pip_install("git+https://github.com/sokrypton/ColabDesign.git")
    .run_commands(
        "mkdir -p /root/BindCraft/functions && mkdir -p /params"
    )
    .run_commands(
        "ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign"
    )
    .run_commands(
        "aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
        " && mkdir -p /root/BindCraft/params"
        " && tar -xf alphafold_params_2022-12-06.tar -C /root/BindCraft/params"
    )
    # JAX
    .pip_install("jax[cuda]")
    .pip_install("jaxlib")
    # Mosaic dependencies: Joltz and Boltz wrappers
    .pip_install("git+https://github.com/adaptyvbio/joltz.git")
    .pip_install("git+https://github.com/jwohlwend/boltz.git")
    # Add this repo source and bindcraft code (expect both present at build time)
    .add_local_dir(str(Path(__file__).resolve().parents[1] / "src"), "/repo/src", copy=True)
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/adaptyv_bindcraft/src", "/root/adaptyv_bindcraft", copy=True)
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/adaptyv_bindcraft/src/BindCraft", "/root/BindCraft", copy=True)
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/adaptyv_bindcraft/utilities", "/root/utilities", copy=True)
)


app = modal.App("mosaic-bindcraft-compat", image=image)


out_vol = modal.Volume.from_name("mosaic-bindcraft-out", create_if_missing=True)
boltz_vol = modal.Volume.from_name("boltz-cache", create_if_missing=False)


def _add_paths():
    # Add mosaic_workflows and BindCraft into sys.path
    for p in ("/repo/src", "/root/adaptyv_bindcraft", "/root/BindCraft", "/root/utilities"):
        if p not in sys.path and Path(p).exists():
            sys.path.insert(0, p)


@app.function(gpu="A10", timeout=8 * 60 * 60, volumes={"/output": out_vol, "/root/.boltz": boltz_vol})
def run_bindcraft_outer(
    *,
    task_name: str = "TEST",
    binder_chain: str = "B",
    target_sequence: str = "MFEARLVQGSI",
    chains: str = "A",
    lengths: List[int] = [20],
    number_of_final_designs: int = 1,
    max_trajectories: int = 1,
    runtime_seed: int | None = 1234,
):
    _add_paths()

    from mosaic_workflows.bindcraft_compat import run_bindcraft_compat

    design_path = f"/output/{task_name}"
    Path(design_path).mkdir(parents=True, exist_ok=True)

    # Minimal settings
    target_settings: Dict[str, Any] = {
        "binder_name": task_name,
        "task_name": task_name,
        "target_sequence": target_sequence,
        "starting_pdb": str(Path("/root/BindCraft") / "example" / "PDL1.pdb"),
        "chains": chains,
        "lengths": lengths,
        "number_of_final_designs": number_of_final_designs,
        "target_hotspot_residues": "",
    }
    advanced_settings: Dict[str, Any] = {
        "binder_chain": binder_chain,
        "use_multimer_design": True,
        "num_recycles_design": 1,
        "num_recycles_validation": 1,
        "af_params_dir": "/root/BindCraft/params",
        "filters_json": str(Path("/root/BindCraft") / "settings_filters" / "openmm_filters.json"),
        # MPNN defaults
        "backbone_noise": 0.0,
        "model_path": "v_48_020",
        "mpnn_weights": "soluble",
        "mpnn_fix_interface": False,
        "omit_AAs": None,
        "sampling_temp": 0.1,
        "num_seqs": 2,
        # Predict flags
        "rm_template_seq_predict": False,
        "rm_template_sc_predict": False,
        # Design algo label for CSV parity
        "design_algorithm": "3stage",
        # External tools
        "dssp_path": "/root/BindCraft/functions/dssp",
        "dalphaball_path": "/root/BindCraft/functions/DAlphaBall.gcc",
        # Minimal predict fn to satisfy child workflow interface (emit does real work)
        "predict_fn_complex": (lambda sequence: {"aux": {}}),
    }
    filters: Dict[str, Any] = {}

    out = run_bindcraft_compat(
        repo_root="/root",  # BindCraft content under /root/BindCraft and /root/adaptyv_bindcraft
        design_path=design_path,
        build_parent=None,
        spawn_children=None,
        emit_row=None,
        max_trajectories=max_trajectories,
        target_settings=target_settings,
        advanced_settings=advanced_settings,
        filters=filters,
        runtime_seed=runtime_seed,
        runtime_length=(lengths[0] if lengths else 20),
        stop=lambda rows: sum(int(r.get("accepted", 0)) for r in rows if r.get("kind") == "parent") >= number_of_final_designs,
    )

    return {"design_dir": design_path, "num_rows": len(out.get("rows", []))}


@app.local_entrypoint()
def main(
    task_name: str = "TEST",
    binder_chain: str = "B",
    target_sequence: str = "MFEARLVQGSI",
    chains: str = "A",
    length: int = 20,
    number_of_final_designs: int = 1,
    max_trajectories: int = 1,
    runtime_seed: int | None = 1234,
):
    res = run_bindcraft_outer.remote(
        task_name=task_name,
        binder_chain=binder_chain,
        target_sequence=target_sequence,
        chains=chains,
        lengths=[int(length)],
        number_of_final_designs=number_of_final_designs,
        max_trajectories=max_trajectories,
        runtime_seed=runtime_seed,
    )
    print(json.dumps(res, indent=2))


