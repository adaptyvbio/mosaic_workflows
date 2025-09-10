import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class CatalyticTriad:
    ser_pos: int
    his_pos: int
    asp_pos: int


@dataclass
class OxyanionHole:
    gly_or_backbone_donors: Sequence[int]


@dataclass
class MHETaseMotifSpec:
    triad: CatalyticTriad
    oxyanion: Optional[OxyanionHole] = None
    # Target AA sets per role
    ser_candidates: tuple[str, ...] = ("S",)
    his_candidates: tuple[str, ...] = ("H",)
    asp_candidates: tuple[str, ...] = ("D", "E")

    # Soft sequence weights (identity penalties)
    w_identity_ser: float = 1.0
    w_identity_his: float = 1.0
    w_identity_asp: float = 1.0
    w_identity_oxyanion: float = 0.25

    # Optional geometric targets (for future structure-aware extension)
    # Defaults approximate serine hydrolase active sites
    target_ser_his_hbond: float = 2.8
    target_his_asp_hbond: float = 2.8
    target_tetrahedral_angle_deg: float = 109.5


def default_mhetase_motif(ser_pos: int, his_pos: int, asp_pos: int, oxyanion_positions: Optional[Sequence[int]] = None) -> MHETaseMotifSpec:
    return MHETaseMotifSpec(
        triad=CatalyticTriad(ser_pos=ser_pos, his_pos=his_pos, asp_pos=asp_pos),
        oxyanion=None if oxyanion_positions is None else OxyanionHole(gly_or_backbone_donors=tuple(oxyanion_positions)),
    )


def parse_simple_pdb_for_atoms(pdb_path: str, chain_id: str, residue_number: int, atom_names: Sequence[str]):
    coords = {}
    wanted = set(atom_names)
    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain = line[21].strip()
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue
            if chain != chain_id or resseq != residue_number:
                continue
            atom = line[12:16].strip()
            if atom in wanted:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords[atom] = (x, y, z)
            if len(coords) == len(wanted):
                break
    return coords


def distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def mine_triad_geometry_from_pdb(
    *,
    pdb_path: str,
    ser: tuple[str, int],
    his: tuple[str, int],
    asp: tuple[str, int],
):
    ser_atoms = parse_simple_pdb_for_atoms(pdb_path, ser[0], ser[1], ["OG", "O" "CB"]) or {}
    his_atoms = parse_simple_pdb_for_atoms(pdb_path, his[0], his[1], ["ND1", "NE2"]) or {}
    asp_atoms = parse_simple_pdb_for_atoms(pdb_path, asp[0], asp[1], ["OD1", "OD2"]) or {}

    # Approximate hydrogen bonds: Ser OG – His NE2 and His ND1 – Asp ODx
    ser_his = None
    if "OG" in ser_atoms and "NE2" in his_atoms:
        ser_his = distance(ser_atoms["OG"], his_atoms["NE2"])
    his_asp = None
    od = "OD1" if "OD1" in asp_atoms else ("OD2" if "OD2" in asp_atoms else None)
    if od is not None and "ND1" in his_atoms:
        his_asp = distance(his_atoms["ND1"], asp_atoms[od])

    return {
        "ser_his_hbond": ser_his,
        "his_asp_hbond": his_asp,
    }



