import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable

from .outer import run_many
from .analyzers import flatten_aux
from functools import lru_cache


def _add_bindcraft_paths(repo_root: str) -> None:
    src = Path(repo_root) / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _bindcraft_dirs(base: str) -> Dict[str, str]:
    names = [
        "Accepted", "Trajectory", "MPNN", "Rejected", "AF2", "CSV", "logs", "settings",
        "Accepted/Ranked", "Accepted/Animation", "Accepted/Plots", "Accepted/Pickle",
        "Trajectory/Pickle", "Trajectory/Relaxed", "Trajectory/Plots", "Trajectory/Clashing",
        "Trajectory/LowConfidence", "Trajectory/Animation",
        "MPNN/Binder", "MPNN/Sequences", "MPNN/Relaxed",
        "AF2/Ranked",
    ]
    paths: Dict[str, str] = {}
    for n in names:
        p = Path(base) / n
        p.mkdir(parents=True, exist_ok=True)
        paths[n] = str(p)
    # canonical CSV names
    paths["trajectory_csv"] = str(Path(base) / "trajectory_stats.csv")
    paths["mpnn_csv"] = str(Path(base) / "mpnn_design_stats.csv")
    paths["final_csv"] = str(Path(base) / "final_design_stats.csv")
    paths["failure_csv"] = str(Path(base) / "failure_csv.csv")
    return paths


def _init_csvs(paths: Dict[str, str]) -> None:
    # Import BindCraft functions for headers and CSV creation
    from BindCraft.functions import generate_dataframe_labels, create_dataframe, generate_filter_pass_csv

    traj_labels, design_labels, final_labels = generate_dataframe_labels()
    create_dataframe(paths["trajectory_csv"], traj_labels)
    create_dataframe(paths["mpnn_csv"], design_labels)
    create_dataframe(paths["final_csv"], final_labels)
    # failure csv based on filter settings is created by generate_filter_pass_csv; we call it in emit


def _append_csv_row(csv_path: str, row: List[Any]) -> None:
    import csv
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)


def _make_design_name(binder_name: str, length: int, seed: int) -> str:
    return f"{binder_name}_l{length}_s{seed}"


def _clean_pdb(pdb_file: str) -> None:
    """Clean PDB in-place by keeping only relevant record types.

    Avoids file-iterator quirks by reading the full file and rewriting.
    """
    with open(pdb_file, "rb") as f_in:
        data = f_in.read()
    lines = data.decode("utf-8", errors="ignore").splitlines(keepends=True)
    keep = ("ATOM", "HETATM", "MODEL", "TER", "END", "LINK")
    filtered = [ln for ln in lines if ln.startswith(keep)]
    with open(pdb_file, "wb") as f_out:
        f_out.writelines([ln.encode("utf-8") for ln in filtered])


def default_emit_row(*, kind: str, row: dict, paths: Dict[str, str], target_settings: Dict[str, Any], advanced_settings: Dict[str, Any], filters: Dict[str, Any]) -> None:
    """BindCraft-parity emitter: relax/DSSP/interface scoring + CSV writes.

    - kind: "parent" or "child"
    - row: {"spec", "best_sequence", "metrics", ...}
    - paths: directory and csv paths from _bindcraft_dirs
    """
    from BindCraft.functions import (
        openmm_relax, score_interface, calc_ss_percentage,
        generate_dataframe_labels, insert_data, generate_filter_pass_csv,
        calculate_averages, check_filters, update_failures,
        target_pdb_rmsd, unaligned_rmsd, save_fasta,
    )

    # Initialize failure CSV once
    generate_filter_pass_csv(paths["failure_csv"], advanced_settings.get("filters_json", ""))

    binder_name = target_settings.get("binder_name", target_settings.get("task_name", "design"))
    length = int(row.get("spec", {}).get("binder_len", 0))
    seed = int(row.get("spec", {}).get("seed", 0))
    design_name = _make_design_name(binder_name, length, seed)

    # Save FASTA for child rows when present
    seq = row.get("best_sequence")
    if seq:
        save_fasta(design_name, seq, paths)

    # Expect structure_path provided upstream or created by predictor-specific emit
    pdb_in = row.get("structure_path")

    # If no structure_path yet, do a quick AF/ColabDesign complex prediction for the parent
    if kind == "parent" and (not pdb_in) and seq:
        from colabdesign import mk_afdesign_model
        # build prediction model and prep inputs
        model = mk_afdesign_model(
            protocol="binder",
            debug=False,
            data_dir=advanced_settings.get("af_params_dir", "/root/BindCraft/params"),
            use_multimer=advanced_settings.get("use_multimer_design", True),
            num_recycles=advanced_settings.get("num_recycles_validation", 1),
        )
        model.prep_inputs(
            pdb_filename=target_settings.get("starting_pdb"),
            chain=target_settings.get("chains", "A"),
            binder_len=length,
            rm_target_seq=advanced_settings.get("rm_template_seq_predict", False),
            rm_target_sc=advanced_settings.get("rm_template_sc_predict", False),
        )
        # predict one model and save to Trajectory
        model.predict(seq=seq, models=[0], num_recycles=advanced_settings.get("num_recycles_validation", 1), verbose=False)
        traj_pdb = os.path.join(paths["Trajectory"], f"{design_name}.pdb")
        model.save_pdb(traj_pdb)
        pdb_in = traj_pdb
        row["structure_path"] = pdb_in
    pdb_relaxed = None
    ss = None
    if pdb_in:
        relaxed_out = os.path.join(paths["Trajectory/Relaxed"], f"{design_name}.pdb")
        ok_relax = openmm_relax(pdb_in, relaxed_out)
        pdb_relaxed = relaxed_out if ok_relax else None
        row["pdb_relaxed"] = pdb_relaxed
        # sanitize PDB to avoid parser issues
        # Skip PDB cleaning due to sporadic file descriptor errors on remote volumes

    # DSSP + Interface scoring if relaxed available
    if pdb_relaxed and os.path.exists(pdb_relaxed) and os.path.getsize(pdb_relaxed) > 0:
        chain = advanced_settings.get("binder_chain", "B")
        ss = calc_ss_percentage(pdb_relaxed, advanced_settings, chain)
        row["dssp"] = ss
        iface_tuple = score_interface(
            pdb_relaxed,
            chain,
            pae_matrix=row.get("pae_matrix"),
            chains=row.get("chains"),
            pae_cutoff=advanced_settings.get("pae_cutoff", 10.0),
            pae_logits=None,
            breaks=None,
        )
        if isinstance(iface_tuple, tuple) and len(iface_tuple) >= 3:
            iface_scores, iface_AA, iface_residues = iface_tuple
            row["interface"] = iface_scores
            row["interface_AA"] = iface_AA
            row["interface_residues"] = iface_residues
        else:
            row["interface"] = iface_tuple

    # Write to CSVs depending on kind
    traj_labels, design_labels, final_labels = generate_dataframe_labels()
    if kind == "parent":
        # Minimal trajectory row: fill known positions, leave others blank/zero
        # ['Design','Protocol','Length','Seed','Helicity','Target_Hotspot','Sequence','InterfaceResidues', ...]
        data = [
            design_name,
            advanced_settings.get("design_protocol", "Default"),
            length,
            seed,
            advanced_settings.get("weights_helicity", 0),
            target_settings.get("target_hotspot_residues", ""),
            seq or "",
            row.get("interface_residues", ""),
        ]
        # Fill confidence/interface/ss metrics with simple lookups; default None
        metrics_map = {
            "pLDDT": row.get("metrics", {}).get("predict.plddt_mean"),
            "pTM": row.get("metrics", {}).get("predict.ptm"),
            "i_pTM": row.get("metrics", {}).get("predict.i_ptm"),
            "pAE": row.get("metrics", {}).get("predict.pae_mean"),
            "i_pAE": row.get("metrics", {}).get("predict.i_pae"),
            "i_pLDDT": row.get("metrics", {}).get("predict.i_plddt"),
            "ss_pLDDT": row.get("metrics", {}).get("predict.ss_plddt"),
        }
        data.extend([metrics_map.get(k) for k in ["pLDDT","pTM","i_pTM","pAE","i_pAE","i_pLDDT","ss_pLDDT"]])
        # Clashes pre/post relax (if present)
        data.extend([row.get("unrelaxed_clashes"), row.get("relaxed_clashes")])
        # Binder energy (optional)
        data.append(row.get("binder_energy"))
        # Interface metrics (some may be None)
        iface = row.get("interface", {}) or {}
        data.extend([
            iface.get('surface_hydrophobicity'),
            iface.get('interface_sc'),
            iface.get('interface_packstat'),
            iface.get('interface_dG'),
            iface.get('interface_dSASA'),
            iface.get('interface_dG_SASA_ratio'),
            iface.get('interface_fraction'),
            iface.get('interface_hydrophobicity'),
            iface.get('interface_nres'),
            iface.get('interface_interface_hbonds'),
            iface.get('interface_hbond_percentage'),
            iface.get('interface_delta_unsat_hbonds'),
            iface.get('interface_delta_unsat_hbonds_percentage'),
        ])
        # SS breakdown
        if isinstance(ss, (list, tuple)) and len(ss) >= 8:
            alpha, beta, loops, alpha_i, beta_i, loops_i, i_plddt, ss_plddt = ss
            data.extend([alpha_i, beta_i, loops_i, alpha, beta, loops])
        else:
            data.extend([None, None, None, None, None, None])
        # InterfaceAAs (string) and target RMSD if available
        data.append(iface.get('InterfaceAAs', ""))
        data.append(row.get("target_rmsd"))
        # Electric fields (optional)
        data.extend([
            iface.get('electric_field_mean'), iface.get('electric_field_max'), iface.get('electric_field_std'),
            iface.get('electric_field_divergence'), iface.get('electric_field_alignment'),
            iface.get('electrostatic_complementarity'), iface.get('field_projection'), iface.get('field_gradient')
        ])
        # Trajectory time and notes & placeholders for settings and filters
        # Sanitize settings to exclude non-JSON-serializable values (e.g., callables)
        adv_safe = {k: v for k, v in advanced_settings.items() if not callable(v)}
        tgt_safe = {k: v for k, v in target_settings.items() if not callable(v)}
        data.extend(["", row.get("notes", ""), json.dumps(tgt_safe), json.dumps({}), json.dumps(adv_safe)])
        _append_csv_row(paths["trajectory_csv"], data)
    else:
        # Child redesign: run BindCraft AF predictions, relax, DSSP/interface, build full MPNN row
        from BindCraft.functions import (
            mk_afdesign_model, clear_mem, calculate_averages,
            load_af2_models, check_filters, insert_data,
        )
        from adaptyv_bindcraft.bindcraft_pipeline import prepare_mpnn_data, save_successful_design, handle_failed_design

        # Required inputs
        # Prefer explicit sequence from spec (MPNN output) over any model-decoded string
        import re
        seq_raw = row.get("spec", {}).get("sequence") or row.get("best_sequence") or ""
        seq_child = re.sub("[^A-Z]", "", seq_raw.upper())
        idx = int(row.get("spec", {}).get("idx", 0)) + 1
        mpnn_design_name = f"{design_name}_mpnn{idx}"
        length_child = len(seq_child)
        # Use parent's design name for trajectory reference (length may differ from child)
        parent_spec = row.get("parent_spec", {}) or {}
        parent_len = int(parent_spec.get("binder_len", length))
        parent_seed = int(parent_spec.get("seed", seed))
        parent_name = _make_design_name(binder_name, parent_len, parent_seed)
        # Prefer relaxed parent complex if present
        traj_relaxed_path = os.path.join(paths["Trajectory/Relaxed"], f"{parent_name}.pdb")
        traj_unrelaxed_path = os.path.join(paths["Trajectory"], f"{parent_name}.pdb")
        trajectory_pdb = traj_relaxed_path if os.path.exists(traj_relaxed_path) else traj_unrelaxed_path

        # AF models selection
        design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings.get("use_multimer_design", True))

        # Compile prediction models
        clear_mem()
        complex_prediction_model = mk_afdesign_model(
            protocol="binder",
            num_recycles=advanced_settings.get("num_recycles_validation", 1),
            data_dir=advanced_settings.get("af_params_dir", "/root/BindCraft/params"),
            use_multimer=multimer_validation,
            use_initial_guess=False,
            initial_guess=False,
            use_initial_atom_pos=False,
        )
        complex_prediction_model.prep_inputs(
            pdb_filename=target_settings.get("starting_pdb"),
            chain=target_settings.get("chains", "A"),
            binder_len=length_child,
            rm_target_seq=advanced_settings.get("rm_template_seq_predict", False),
            rm_target_sc=advanced_settings.get("rm_template_sc_predict", False),
        )

        # Predict complex and relax (BindCraft utility handles AF2 filters and relaxation)
        from BindCraft.functions.colabdesign_utils import predict_binder_complex, predict_binder_alone
        prediction_stats, pass_af2_filters, complex_prediction_model = predict_binder_complex(
            complex_prediction_model,
            seq_child,
            mpnn_design_name,
            target_settings.get("starting_pdb"),
            target_settings.get("chains", "A"),
            length_child,
            trajectory_pdb,
            prediction_models,
            advanced_settings,
            {},
            paths,
            paths["failure_csv"],
        )

        # Averages for complex
        mpnn_complex_averages = calculate_averages(prediction_stats, handle_aa=True)

        # Binder-alone prediction
        binder_prediction_model = mk_afdesign_model(
            protocol="hallucination",
            use_templates=False,
            initial_guess=False,
            use_initial_atom_pos=False,
            num_recycles=advanced_settings.get("num_recycles_validation", 1),
            data_dir=advanced_settings.get("af_params_dir", "/root/BindCraft/params"),
            use_multimer=multimer_validation,
        )
        binder_prediction_model.prep_inputs(length=length_child)
        binder_stats = predict_binder_alone(
            binder_prediction_model,
            seq_child,
            mpnn_design_name,
            length_child,
            trajectory_pdb,
            advanced_settings.get("binder_chain", "B"),
            prediction_models,
            advanced_settings,
            paths,
        )
        binder_averages = calculate_averages(binder_stats)

        # Build MPNN row using BindCraft helper
        analysis_results = {
            "binder_chain": advanced_settings.get("binder_chain", "B"),
            "trajectory_interface_residues": row.get("interface_residues", ""),
        }
        mpnn_sequence = {"seq": seq_child, "score": 0.0, "seqid": 0}
        mpnn_data = prepare_mpnn_data(
            mpnn_design_name,
            advanced_settings,
            length_child,
            seed,
            analysis_results,
            mpnn_sequence,
            mpnn_complex_averages,
            prediction_stats,
            binder_averages,
            binder_stats,
            target_settings,
            prediction_models,
            "unknown",
            "",
        )

        # Append to MPNN CSV
        insert_data(paths["mpnn_csv"], mpnn_data)

        # Filter and accept/reject
        from BindCraft.functions import generate_dataframe_labels
        _, design_labels, _ = generate_dataframe_labels()
        filter_conditions = check_filters(mpnn_data, design_labels, {})
        if filter_conditions is True:
            save_successful_design(mpnn_design_name, os.path.join(paths["MPNN/Relaxed"], f"{mpnn_design_name}_model1.pdb"), design_name, paths, mpnn_data, paths["final_csv"])
            row["accepted"] = True
        else:
            handle_failed_design(mpnn_design_name, os.path.join(paths["MPNN/Relaxed"], f"{mpnn_design_name}_model1.pdb"), filter_conditions, paths, paths["failure_csv"])
            row["accepted"] = False


def default_spawn_children(*, spec: dict, parent_result: dict, parent_row: dict, target_settings: Dict[str, Any], advanced_settings: Dict[str, Any]) -> List[Tuple[dict, Callable[[dict], dict]]]:
    """Generate MPNN redesigns using BindCraft's mpnn_gen_sequence and create child predict-only workflows.

    Returns list of (child_spec, child_build) pairs.
    """
    from BindCraft.functions import mpnn_gen_sequence
    from .predict import make_predict_only_workflow

    binder_chain = advanced_settings.get("binder_chain", "B")
    pdb_relaxed = parent_row.get("pdb_relaxed")
    interface_residues = parent_row.get("interface_residues", "")
    if pdb_relaxed is None:
        return []

    mpnn_traj = mpnn_gen_sequence(pdb_relaxed, binder_chain, interface_residues, advanced_settings)
    seqs = mpnn_traj.get("seq", [])
    children: List[Tuple[dict, Callable[[dict], dict]]] = []
    for idx, seq in enumerate(seqs):
        child_spec = {"binder_len": len(seq), "seed": spec.get("seed", 0), "sequence": seq, "idx": idx}
        def child_build(s=child_spec):
            # Minimal no-op workflow; emit handles real predictions
            return {"phases": [], "binder_len": s["binder_len"], "seed": s.get("seed", 0)}
        children.append((child_spec, child_build))
    return children


def sample_specs_bindcraft_style(*, max_trajectories: int, target_settings: Dict[str, Any], runtime_seed: int | None, runtime_length: int | None) -> List[dict]:
    # Use adaptyv_bindcraft.state_manager initializer to match semantics
    from adaptyv_bindcraft.state_manager import initialize_derived_seeds_and_lengths
    class _S:  # minimal state shim
        derived_seeds: List[int] | None = None
        sampled_lengths: List[int] | None = None
    s = _S()
    initialize_derived_seeds_and_lengths(s, max_trajectories, target_settings, runtime_seed, runtime_length)
    specs: List[dict] = []
    for i in range(max_trajectories):
        specs.append({
            "seed": int(s.derived_seeds[i] if s.derived_seeds else 0),
            "binder_len": int(s.sampled_lengths[i] if s.sampled_lengths else target_settings.get("lengths", [60])[0]),
            "idx": i,
        })
    return specs


def run_bindcraft_compat(
    *,
    repo_root: str,                  # "/Users/.../adaptyv_bindcraft"
    design_path: str,                # output directory
    build_parent: Callable[[dict], dict] | None,
    spawn_children: Callable[[dict, dict, dict], List[Tuple[dict, Callable[[dict], dict]]]] | None,
    emit_row: Callable[[str, dict, Dict[str, str]], None] | None,
    max_trajectories: int,
    target_settings: Dict[str, Any],
    advanced_settings: Dict[str, Any],
    filters: Dict[str, Any],
    runtime_seed: int | None = None,
    runtime_length: int | None = None,
    stop: Callable[[List[dict]], bool] | None = None,
) -> dict:
    """One-shot BindCraft-compatible loop using Mosaic workflows.

    - build_parent(spec) -> workflow dict (Mosaic)
    - spawn_children(spec, parent_result) -> list[(child_spec, child_build)]
    - emit_row(kind, row, paths) -> IO and CSV writes
    - Uses BindCraft CSV schemas and deterministic seeds/lengths
    """
    _add_bindcraft_paths(repo_root)
    paths = _bindcraft_dirs(design_path)
    _init_csvs(paths)

    specs = sample_specs_bindcraft_style(
        max_trajectories=max_trajectories,
        target_settings=target_settings,
        runtime_seed=runtime_seed,
        runtime_length=runtime_length,
    )

    # Provide a default Mosaic Boltz1-based builder if none provided and target_sequence exists
    if build_parent is None:
        target_seq = target_settings.get("target_sequence")
        if not target_seq:
            raise ValueError("build_parent is None and target_settings lacks 'target_sequence'")
        build_parent = make_build_parent_boltzdesign1(target_sequence=target_seq)

    def _emit(kind: str, row: dict) -> None:
        if emit_row is not None:
            emit_row(kind, row, paths)
        else:
            default_emit_row(kind=kind, row=row, paths=paths, target_settings=target_settings, advanced_settings=advanced_settings, filters=filters)

    out = run_many(
        specs=specs,
        build=build_parent,
        spawn=(lambda a,b,c: default_spawn_children(spec=a, parent_result=b, parent_row=c, target_settings=target_settings, advanced_settings=advanced_settings)) if spawn_children is None else spawn_children,
        emit=_emit,
        stop=stop,
        resume=False,
        out_dir=design_path,
    )
    return out


@lru_cache(maxsize=1)
def _load_boltz_cached():
    from mosaic.losses.boltz import load_boltz
    return load_boltz()


def make_build_parent_boltzdesign1(
    *,
    target_sequence: str,
    intra_k: int = 4,
    intra_seqsep: int = 8,
    intra_contact_distance: float = 14.0,
    inter_contact_distance: float = 21.0,
    helix_weight: float = -0.3,
    plddt_weight: float = 0.0,
    bb_pae_weight: float = 0.0,
    bt_pae_weight: float = 0.0,
    recycling_steps: int = 0,
    deterministic: bool = True,
):
    """Return a build(spec)->workflow that runs Boltz1 (boltzdesign1-style) in Mosaic."""
    from mosaic.losses.boltz import make_binder_features, Boltz1Loss
    from mosaic_workflows import (
        adamw_logits_adapter as adamw_logits,
        sgd_logits_adapter as sgd_logits,
        zgr_adapter,
        init_logits_boltzdesign1,
    )
    from mosaic_workflows.transforms import (
        temperature_on_logits, e_soft_on_logits, gradient_normalizer, zero_disallowed,
    )
    import numpy as np

    joltz = _load_boltz_cached()

    def make_build_loss(binder_len: int):
        def build_loss():
            import mosaic.losses.structure_prediction as sp
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

            features, _ = make_binder_features(
                binder_len=binder_len,
                target_sequence=target_sequence,
                use_msa=False,
                use_msa_server=False,
            )
            return Boltz1Loss(
                joltz1=joltz,
                name="boltz1",
                loss=losses,
                features=features,
                recycling_steps=recycling_steps,
                deterministic=deterministic,
            )

        return build_loss

    def build_parent(spec: dict) -> dict:
        binder_len = int(spec["binder_len"])
        seed = int(spec.get("seed", 0))

        build_loss = make_build_loss(binder_len)

        warmup = {
            "name": "warmup",
            "build_loss": build_loss,
            "optimizer": adamw_logits,
            "steps": 30,
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
            "steps": 120,
            "schedule": lambda g, p: {"lr": 0.05, "temperature": 1.0, "e_soft": 0.8},
            "transforms": {
                "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
                "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])],
            },
        }

        def anneal_sched(g, p):
            total = 240.0
            frac = p / total
            temp = 1.0 - (1.0 - 0.01) * (frac ** 2)
            return {"lr": 0.05, "temperature": max(0.01, float(temp))}

        anneal = {
            "name": "anneal",
            "build_loss": build_loss,
            "optimizer": sgd_logits,
            "steps": 240,
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
            "schedule": lambda g, p: {"lr": 0.05, "temperature": 0.01},
            "transforms": {"pre_logits": [temperature_on_logits()]},
        }

        phases = [warmup, soft, anneal, hard]
        x0 = init_logits_boltzdesign1(binder_len=binder_len, noise_scaling=0.1, rng=np.random.default_rng(seed))
        return {"phases": phases, "binder_len": binder_len, "seed": seed, "initial_x": x0}

    return build_parent


