from dataclasses import dataclass
from typing import Optional, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np

from ..common import LossTerm, LinearCombination
from .structure_prediction import AbstractStructureOutput


@dataclass
class LJGlobalParams:
    # Mirrors tmol/tmol/score/ljlk/potentials/common.hh and params.hh usage
    lj_hbond_dis: float
    lj_hbond_OH_donor_dis: float
    lj_hbond_hdis: float


@dataclass
class LJTypeParams:
    # Per-atom parameters (already mapped to the atom list in the pose)
    lj_radius: Float[Array, "A"]
    lj_wdepth: Float[Array, "A"]
    is_acceptor: Array
    is_donor: Array
    is_hydroxyl: Array
    is_polarh: Array


def _connectivity_weight(bonded_path_length: Array) -> Array:
    # Exact match of tmol logic: >4 -> 1.0; ==4 -> 0.2; else 0.0
    return jnp.where(
        bonded_path_length > 4,
        1.0,
        jnp.where(bonded_path_length == 4, 0.2, 0.0),
    )


def _lj_sigma(
    i_is_donor: Array,
    i_is_hydroxyl: Array,
    i_is_polarh: Array,
    i_is_acceptor: Array,
    j_is_donor: Array,
    j_is_hydroxyl: Array,
    j_is_polarh: Array,
    j_is_acceptor: Array,
    i_lj_radius: Array,
    j_lj_radius: Array,
    gp: LJGlobalParams,
) -> Array:
    # Mirrors tmol/tmol/score/ljlk/potentials/common.hh::lj_sigma
    # standard donor/acceptor pair
    cond_std = (i_is_donor & (~i_is_hydroxyl) & j_is_acceptor) | (
        j_is_donor & (~j_is_hydroxyl) & i_is_acceptor
    )
    # hydroxyl donor/acceptor pair
    cond_oh = (i_is_donor & i_is_hydroxyl & j_is_acceptor) | (
        j_is_donor & j_is_hydroxyl & i_is_acceptor
    )
    # hydrogen/acceptor pair
    cond_hacc = (i_is_polarh & j_is_acceptor) | (j_is_polarh & i_is_acceptor)

    sigma_std = gp.lj_hbond_dis
    sigma_oh = gp.lj_hbond_OH_donor_dis
    sigma_hacc = gp.lj_hbond_hdis
    sigma_lj = i_lj_radius + j_lj_radius

    return jnp.where(
        cond_std,
        sigma_std,
        jnp.where(
            cond_oh,
            sigma_oh,
            jnp.where(
                cond_hacc,
                sigma_hacc,
                sigma_lj,
            ),
        ),
    )


def _vdw_V(dist: Array, sigma: Array, epsilon: Array) -> Array:
    # tmol/tmol/score/ljlk/potentials/lj.hh::vdw::V
    sd = sigma / jnp.clip(dist, a_min=1e-8)
    sd2 = sd * sd
    sd6 = sd2 * sd2 * sd2
    sd12 = sd6 * sd6
    return epsilon * (sd12 - 2.0 * sd6)

def _vdw_V_elementwise(dist: float, sigma: float, epsilon: float) -> float:
    """Elementwise version for gradient computation"""
    sd = sigma / jnp.maximum(dist, 1e-8)
    sd2 = sd * sd
    sd6 = sd2 * sd2 * sd2
    sd12 = sd6 * sd6
    return epsilon * (sd12 - 2.0 * sd6)


def _lj_split_attractive_repulsive(
    dist: Array, sigma: Array, epsilon: Array
) -> tuple[Array, Array]:
    # Implements the piecewise + linearization + cutoff blending as in lj.hh::lj_score::V
    cpoly_dmax = 6.0
    d_lin = 0.6 * sigma
    # cpoly_dmin = max(4.5, min(sigma, cpoly_dmax-0.1)) exactly as in header
    cpoly_dmin = jnp.maximum(4.5, jnp.minimum(sigma, cpoly_dmax - 0.1))

    # Weight by connectivity handled outside
    # Evaluate vdw at helpful points
    v_at_d_lin = _vdw_V(d_lin, sigma, epsilon)
    # Compute gradient element-wise
    dvdd_at_d_lin = jax.vmap(jax.vmap(jax.grad(_vdw_V_elementwise)))(d_lin, sigma, epsilon)
    v_at_cpoly_dmin = _vdw_V(cpoly_dmin, sigma, epsilon)
    dvdd_at_cpoly_dmin = jax.vmap(jax.vmap(jax.grad(_vdw_V_elementwise)))(cpoly_dmin, sigma, epsilon)

    # Regions
    # dist > cpoly_dmax -> 0
    region_far = dist > cpoly_dmax
    # d_lin < dist <= cpoly_dmin: raw vdw
    region_mid = (dist > d_lin) & (dist <= cpoly_dmin)
    # dist <= d_lin: linearization at d_lin
    region_lin = dist <= d_lin
    # cpoly_dmin < dist <= cpoly_dmax: interpolate_to_zero
    region_tail = (dist > cpoly_dmin) & (dist <= cpoly_dmax)

    # Linearized near d_lin
    v_lin = v_at_d_lin + dvdd_at_d_lin * (dist - d_lin)
    # Tail interpolation to zero at cpoly_dmax using cubic Hermite with (v, dv)
    # interpolate_to_zero matches tmol common::interpolate_to_zero
    def _interp_to_zero(x, x0, f0, df0, x1):
        # Hermite basis
        t = (x - x0) / jnp.clip((x1 - x0), a_min=1e-8)
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        # target f(x1)=0, f'(x1)=0
        return h00 * f0 + h10 * (x1 - x0) * df0

    v_tail = _interp_to_zero(dist, cpoly_dmin, v_at_cpoly_dmin, dvdd_at_cpoly_dmin, cpoly_dmax)
    v_mid = _vdw_V(dist, sigma, epsilon)

    lj = jnp.where(
        region_far,
        0.0,
        jnp.where(region_tail, v_tail, jnp.where(region_mid, v_mid, v_lin)),
    )

    # Split into attractive/repulsive components using sigma cutoff
    atr = jnp.where(dist < sigma, -epsilon, lj)
    rep = jnp.where(dist < sigma, lj + epsilon, 0.0)
    return atr, rep


def lj_energy_pairs(
    coords: Float[Array, "A 3"],
    lj_type: LJTypeParams,
    lj_global: LJGlobalParams,
    bonded_path_length: Array,
    pair_mask: Optional[Array] = None,
) -> tuple[Array, dict]:
    # Pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist = jnp.linalg.norm(diff + 1e-12, axis=-1)

    # Sigma and epsilon per pair
    sigma = _lj_sigma(
        lj_type.is_donor[:, None],
        lj_type.is_hydroxyl[:, None],
        lj_type.is_polarh[:, None],
        lj_type.is_acceptor[:, None],
        lj_type.is_donor[None, :],
        lj_type.is_hydroxyl[None, :],
        lj_type.is_polarh[None, :],
        lj_type.is_acceptor[None, :],
        lj_type.lj_radius[:, None],
        lj_type.lj_radius[None, :],
        lj_global,
    )
    epsilon = jnp.sqrt(lj_type.lj_wdepth[:, None] * lj_type.lj_wdepth[None, :])

    Vatr, Vrep = _lj_split_attractive_repulsive(dist, sigma, epsilon)

    # Connectivity weights (uses representative path length between atoms)
    w = _connectivity_weight(bonded_path_length)

    # Masking: ignore self and optionally any padding
    A = coords.shape[0]
    base_mask = 1.0 - jnp.eye(A)
    if pair_mask is not None:
        base_mask = base_mask * pair_mask

    # Sum over unique pairs only (avoid i-j and j-i double counting)
    tri_mask = jnp.triu(jnp.ones_like(base_mask), k=1)
    pair_mask_full = base_mask * tri_mask
    Eatr = (w * Vatr * pair_mask_full).sum() * 1.0
    Erep = (w * Vrep * pair_mask_full).sum() * 1.0
    E = Eatr + Erep
    return E, {"lj_atr": Eatr, "lj_rep": Erep}


# ---------------- Electrostatics (exact formula replication) -----------------


@dataclass
class ElecGlobalParams:
    D: float
    D0: float
    S: float
    min_dis: float
    max_dis: float


@dataclass
class ElecTypeParams:
    charges: Float[Array, "A"]


def _elec_eps(dist: Array, D: float, D0: float, S: float) -> Array:
    # tmol/tmol/score/elec/potentials/potentials.hh::eps
    return D - 0.5 * (D - D0) * (2 + 2 * dist * S + (dist * dist) * (S * S)) * jnp.exp(
        -dist * S
    )


def _interpolate(x, x0, f0, df0, x1, f1, df1):
    # Cubic Hermite interpolation between (x0, f0, df0) and (x1, f1, df1)
    t = (x - x0) / jnp.clip((x1 - x0), a_min=1e-8)
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * f0 + h10 * (x1 - x0) * df0 + h01 * f1 + h11 * (x1 - x0) * df1


def _interpolate_to_zero(x, x0, f0, df0, x1):
    # Special case used by TMOL where endpoint has f1=0, df1=0
    t = (x - x0) / jnp.clip((x1 - x0), a_min=1e-8)
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    return h00 * f0 + h10 * (x1 - x0) * df0


def elec_energy_pairs(
    coords: Float[Array, "A 3"],
    elec_type: ElecTypeParams,
    elec_global: ElecGlobalParams,
    bonded_path_length: Array,
    pair_mask: Optional[Array] = None,
) -> tuple[Array, dict]:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = jnp.linalg.norm(diff + 1e-12, axis=-1)

    q = elec_type.charges
    eiej = q[:, None] * q[None, :]

    # Constants
    D = float(elec_global.D)
    D0 = float(elec_global.D0)
    S = float(elec_global.S)
    min_dis = float(elec_global.min_dis)
    max_dis = float(elec_global.max_dis)

    low_poly_start = min_dis - 0.25
    low_poly_end = min_dis + 0.25
    hi_poly_start = max_dis - 1.0
    hi_poly_end = max_dis

    # C1, C2 from TMOL
    C1 = 322.0637
    C2 = C1 / (max_dis * _elec_eps(max_dis, D, D0, S))

    # Base masks
    A = coords.shape[0]
    base_mask = 1.0 - jnp.eye(A)
    if pair_mask is not None:
        base_mask = base_mask * pair_mask

    # Regions
    region0 = dist < low_poly_start
    region1 = (dist >= low_poly_start) & (dist < low_poly_end)
    region2 = (dist >= low_poly_end) & (dist < hi_poly_start)
    region3 = (dist >= hi_poly_start) & (dist < hi_poly_end)

    # Precompute endpoint values/derivatives
    min_dis_score = C1 / (min_dis * _elec_eps(min_dis, D, D0, S)) - C2
    eps_elec_end1 = _elec_eps(low_poly_end, D, D0, S)
    deps_end1 = jax.grad(lambda d: _elec_eps(d, D, D0, S))(low_poly_end)
    dmax_elec = eiej * (C1 / (low_poly_end * eps_elec_end1) - C2)
    # Faithful derivative of energy wrt distance at low_poly_end:
    # dE/dd = -C1 * eiej * (eps + r * deps) / (r^2 * eps^2)
    dmax_elec_d_dist = -C1 * eiej * (eps_elec_end1 + low_poly_end * deps_end1) / (
        (low_poly_end * low_poly_end) * (eps_elec_end1 * eps_elec_end1)
    )

    # Region energies
    E0 = eiej * min_dis_score

    E1 = _interpolate(
        dist,
        low_poly_start,
        eiej * min_dis_score,
        0.0,
        low_poly_end,
        dmax_elec,
        dmax_elec_d_dist,
    )

    eps_elec_mid = _elec_eps(dist, D, D0, S)
    E2 = eiej * (C1 / (jnp.clip(dist, a_min=1e-8) * eps_elec_mid) - C2)

    eps_elec_start3 = _elec_eps(hi_poly_start, D, D0, S)
    deps_start3 = jax.grad(lambda d: _elec_eps(d, D, D0, S))(hi_poly_start)
    dmin_elec = eiej * (C1 / (hi_poly_start * eps_elec_start3) - C2)
    dmin_elec_d_dist = -C1 * eiej * (eps_elec_start3 + hi_poly_start * deps_start3) / (
        (hi_poly_start * hi_poly_start) * (eps_elec_start3 * eps_elec_start3)
    )

    E3 = _interpolate_to_zero(dist, hi_poly_start, dmin_elec, dmin_elec_d_dist, hi_poly_end)

    E = jnp.where(
        region0,
        E0,
        jnp.where(region1, E1, jnp.where(region2, E2, jnp.where(region3, E3, 0.0))),
    )

    # Connectivity weights (per TMOL)
    w = _connectivity_weight(bonded_path_length)
    # Sum over unique pairs only
    tri_mask = jnp.triu(jnp.ones_like(base_mask), k=1)
    pair_mask_full = base_mask * tri_mask
    E_sum = (w * E * pair_mask_full).sum()
    return E_sum, {"elec": E_sum}


class TMOLEnergy(LossTerm):
    # This scaffolds TMOL parity. Additional terms (elec, hbond, etc.) to be added.
    lj_global: LJGlobalParams
    lj_type: LJTypeParams
    bonded_path_length: Float[Array, "A A"]
    atom_mask: Optional[Array] = None  # 1.0 for valid atoms, 0.0 for padded
    elec_global: Optional[ElecGlobalParams] = None
    elec_type: Optional[ElecTypeParams] = None
    # Hydrogen bond optional parameters (pair-prepared interface)
    hb_global: Optional["HBondGlobalParams"] = None
    hb_pairs: Optional["HBondPairs"] = None

    def __call__(self, *args, key, **kwds) -> tuple[float, dict]:
        # Accept Mosaic LossTerm calling convention: (sequence, output, *, key=...)
        sequence = kwds.get("sequence") if "sequence" in kwds else args[0]
        if "output" in kwds:
            output = cast(AbstractStructureOutput, kwds["output"])
        else:
            output = cast(AbstractStructureOutput, args[1])
        # Expect predictor to expose all-atom coordinates as `structure_coordinates`
        coords_all = getattr(output, "structure_coordinates", None)
        if coords_all is None:
            raise ValueError("Predictor output must expose structure_coordinates for TMOL energy.")

        # Optional mask to drop padded atoms
        if self.atom_mask is not None:
            keep = self.atom_mask > 0.5
            coords = coords_all[keep]
            keep_idx = jnp.where(keep)[0]
            bpl = self.bonded_path_length[jnp.ix_(keep_idx, keep_idx)]
            lj_type = LJTypeParams(
                lj_radius=self.lj_type.lj_radius[keep],
                lj_wdepth=self.lj_type.lj_wdepth[keep],
                is_acceptor=self.lj_type.is_acceptor[keep],
                is_donor=self.lj_type.is_donor[keep],
                is_hydroxyl=self.lj_type.is_hydroxyl[keep],
                is_polarh=self.lj_type.is_polarh[keep],
            )
        else:
            coords = coords_all
            bpl = self.bonded_path_length
            lj_type = self.lj_type

        E_lj, aux_lj = lj_energy_pairs(
            coords=coords,
            lj_type=lj_type,
            lj_global=self.lj_global,
            bonded_path_length=bpl,
        )
        total = E_lj
        aux = {**aux_lj}

        # Optional electrostatics term if params provided
        if self.elec_global is not None and self.elec_type is not None:
            E_elec, aux_e = elec_energy_pairs(
                coords=coords,
                elec_type=ElecTypeParams(
                    charges=self.elec_type.charges[self.atom_mask > 0.5]
                    if (self.atom_mask is not None)
                    else self.elec_type.charges
                ),
                elec_global=self.elec_global,
                bonded_path_length=bpl,
            )
            total = total + E_elec
            aux.update(aux_e)

        # Optional hydrogen bond term if pair list provided
        if self.hb_global is not None and self.hb_pairs is not None:
            # remap indices if atom_mask applied
            if self.atom_mask is not None:
                keep = (self.atom_mask > 0.5).astype(jnp.int32)
                # map old indices to new compacted indices
                old_to_new = -jnp.ones((keep.shape[0],), dtype=jnp.int32)
                new_idx = jnp.cumsum(keep) - 1
                old_to_new = jnp.where(keep == 1, new_idx, old_to_new)
                def remap(x):
                    return old_to_new[x]
                pairs = HBondPairs(
                    D_idx=remap(self.hb_pairs.D_idx),
                    H_idx=remap(self.hb_pairs.H_idx),
                    A_idx=remap(self.hb_pairs.A_idx),
                    B_idx=remap(self.hb_pairs.B_idx),
                    B0_idx=remap(self.hb_pairs.B0_idx),
                    acceptor_hybridization=self.hb_pairs.acceptor_hybridization,
                    acceptor_weight=self.hb_pairs.acceptor_weight,
                    donor_weight=self.hb_pairs.donor_weight,
                    AHdist_coeffs=self.hb_pairs.AHdist_coeffs,
                    AHdist_range=self.hb_pairs.AHdist_range,
                    AHdist_bound=self.hb_pairs.AHdist_bound,
                    cosAHD_coeffs=self.hb_pairs.cosAHD_coeffs,
                    cosAHD_range=self.hb_pairs.cosAHD_range,
                    cosAHD_bound=self.hb_pairs.cosAHD_bound,
                    cosBAH_coeffs=self.hb_pairs.cosBAH_coeffs,
                    cosBAH_range=self.hb_pairs.cosBAH_range,
                    cosBAH_bound=self.hb_pairs.cosBAH_bound,
                )
            else:
                pairs = self.hb_pairs

            E_hb = hbond_energy_pairs(
                coords=coords,
                pairs=pairs,
                hb_global=self.hb_global,
            )
            total = total + E_hb
            aux.update({"hbond": E_hb})

        return total, {"tmol": {**aux, "total": total}}


# --------------------------- Hydrogen bonds (JAX) ----------------------------


@dataclass
class HBondGlobalParams:
    hb_sp3_softmax_fade: float
    hb_sp2_BAH180_rise: float
    hb_sp2_range_span: float
    hb_sp2_outer_width: float
    max_ha_dis: float


@dataclass
class HBondPairs:
    # P preselected donor-H / acceptor pairs with required atom indices
    D_idx: Array
    H_idx: Array
    A_idx: Array
    B_idx: Array
    B0_idx: Array
    acceptor_hybridization: Array  # 1=sp2, 2=sp3, 3=ring
    acceptor_weight: Float[Array, "P"]
    donor_weight: Float[Array, "P"]
    # Per-pair polynomials (degree 11) and shared ranges/bounds
    AHdist_coeffs: Float[Array, "P 11"]
    AHdist_range: Float[Array, "2"]
    AHdist_bound: Float[Array, "2"]
    cosAHD_coeffs: Float[Array, "P 11"]
    cosAHD_range: Float[Array, "2"]
    cosAHD_bound: Float[Array, "2"]
    cosBAH_coeffs: Float[Array, "P 11"]
    cosBAH_range: Float[Array, "2"]
    cosBAH_bound: Float[Array, "2"]


def _bound_poly_eval(x: Array, coeffs: Array, rng: Array, bnd: Array) -> Array:
    # coeffs shape (..., 11)
    # Horner's method
    def horner(c, xx):
        v = c[..., 0]
        for i in range(1, c.shape[-1]):
            v = v * xx + c[..., i]
        return v
    below = x < rng[0]
    above = x > rng[1]
    val = horner(coeffs, x)
    val = jnp.where(below, bnd[0], jnp.where(above, bnd[1], val))
    return val


def _distance(a: Array, b: Array) -> Array:
    return jnp.linalg.norm(b - a + 1e-12, axis=-1)


def _interior_angle(a: Array, h: Array, d: Array) -> Array:
    # angle A-H-D
    v1 = a - h
    v2 = d - h
    v1n = v1 / jnp.clip(jnp.linalg.norm(v1, axis=-1, keepdims=True), a_min=1e-12)
    v2n = v2 / jnp.clip(jnp.linalg.norm(v2, axis=-1, keepdims=True), a_min=1e-12)
    cosang = (v1n * v2n).sum(axis=-1)
    cosang = jnp.clip(cosang, -1.0, 1.0)
    return jnp.arccos(cosang)


def _cos_interior_angle(u: Array, v: Array) -> Array:
    un = u / jnp.clip(jnp.linalg.norm(u, axis=-1, keepdims=True), a_min=1e-12)
    vn = v / jnp.clip(jnp.linalg.norm(v, axis=-1, keepdims=True), a_min=1e-12)
    return jnp.clip((un * vn).sum(axis=-1), -1.0, 1.0)


def _dihedral(a: Array, b: Array, c: Array, d: Array) -> Array:
    # dihedral angle between planes (a,b,c) and (b,c,d)
    b0 = a - b
    b1 = c - b
    b2 = d - c
    n0 = jnp.cross(b0, b1)
    n1 = jnp.cross(b1, b2)
    n0n = n0 / jnp.clip(jnp.linalg.norm(n0, axis=-1, keepdims=True), a_min=1e-12)
    n1n = n1 / jnp.clip(jnp.linalg.norm(n1, axis=-1, keepdims=True), a_min=1e-12)
    m1 = jnp.cross(n0n, b1 / jnp.clip(jnp.linalg.norm(b1, axis=-1, keepdims=True), a_min=1e-12))
    x = (n0n * n1n).sum(axis=-1)
    y = (m1 * n1n).sum(axis=-1)
    return jnp.arctan2(y, x)


def _BAH_angle_base_form(B: Array, A: Array, H: Array, coeffs: Array, rng: Array, bnd: Array) -> Array:
    AH = H - A
    BA = A - B
    cosT = _cos_interior_angle(AH, BA)
    return _bound_poly_eval(cosT, coeffs, rng, bnd)


def hbond_energy_pairs(
    coords: Float[Array, "A 3"],
    pairs: HBondPairs,
    hb_global: HBondGlobalParams,
) -> Array:
    if pairs.D_idx.shape[0] == 0:
        return jnp.array(0.0, dtype=jnp.float32)
    # Gather coordinates
    D = coords[pairs.D_idx]
    H = coords[pairs.H_idx]
    A = coords[pairs.A_idx]
    B = coords[pairs.B_idx]
    B0 = coords[pairs.B0_idx]

    # HA distance gating
    HA_dist = _distance(A, H)
    within = HA_dist < hb_global.max_ha_dis

    # A-H distance polynomial
    E_AH = _bound_poly_eval(HA_dist, pairs.AHdist_coeffs, pairs.AHdist_range, pairs.AHdist_bound)

    # AHD angle polynomial (non-cos space)
    AHD = _interior_angle(A, H, D)
    E_AHD = _bound_poly_eval(AHD, pairs.cosAHD_coeffs, pairs.cosAHD_range, pairs.cosAHD_bound)

    # BAH term depends on acceptor hybridization
    # sp2: base form with B
    E_BAH_sp2 = _BAH_angle_base_form(B, A, H, pairs.cosBAH_coeffs, pairs.cosBAH_range, pairs.cosBAH_bound)
    # ring: base form with Bm=(B+B0)/2
    Bm = 0.5 * (B + B0)
    E_BAH_ring = _BAH_angle_base_form(Bm, A, H, pairs.cosBAH_coeffs, pairs.cosBAH_range, pairs.cosBAH_bound)
    # sp3: softmax fade between B and B0 base forms
    E_BAH_B = _BAH_angle_base_form(B, A, H, pairs.cosBAH_coeffs, pairs.cosBAH_range, pairs.cosBAH_bound)
    E_BAH_B0 = _BAH_angle_base_form(B0, A, H, pairs.cosBAH_coeffs, pairs.cosBAH_range, pairs.cosBAH_bound)
    s = float(hb_global.hb_sp3_softmax_fade)
    # avoid overflow with logsumexp form
    m = jnp.maximum(E_BAH_B * s, E_BAH_B0 * s)
    E_BAH_sp3 = (jnp.log(jnp.exp(E_BAH_B * s - m) + jnp.exp(E_BAH_B0 * s - m)) + m) / s

    # Select per pair by hybridization: 1=sp2, 3=ring, 2=sp3; 0/other -> 0
    hyb = pairs.acceptor_hybridization
    E_BAH = jnp.where(hyb == 1, E_BAH_sp2, 0.0)
    E_BAH = jnp.where(hyb == 3, E_BAH_ring, E_BAH)
    E_BAH = jnp.where(hyb == 2, E_BAH_sp3, E_BAH)

    # B0BAH chi for sp2 only
    BAH = _interior_angle(B, A, H)
    B0BAH = _dihedral(B0, B, A, H)
    E_chi = sp2chi_energy(
        ang=BAH,
        chi=B0BAH,
        d=hb_global.hb_sp2_BAH180_rise,
        m=hb_global.hb_sp2_range_span,
        l=hb_global.hb_sp2_outer_width,
    )
    E_chi = jnp.where(hyb == 1, E_chi, 0.0)

    # Sum components
    E_pair = E_AH + E_AHD + E_BAH + E_chi

    # Donor/Acceptor weights
    ad_weight = pairs.acceptor_weight * pairs.donor_weight
    E_pair = E_pair * ad_weight

    # Truncate and fade
    E = jnp.where(E_pair > 0.1, 0.0, jnp.where(E_pair > -0.1, (-0.025 + 0.5 * E_pair - 2.5 * E_pair * E_pair), E_pair))

    # Zero out pairs beyond max_ha_dis
    E = jnp.where(within, E, 0.0)

    return E.sum()


def sp2chi_energy(ang: Array, chi: Array, d: float, m: float, l: float) -> Array:
    pi = jnp.pi
    H = 0.5 * (jnp.cos(2 * chi) + 1.0)
    # region 1: ang > 2pi/3
    E1 = H * (d / 2.0 * jnp.cos(3.0 * (pi - ang)) + d / 2.0 - 0.5) + (1 - H) * (d - 0.5)
    # region 2: ang >= 2pi/3 - l
    outer_rise = jnp.cos(pi - (pi * 2.0 / 3.0 - ang) / l)
    F = m / 2.0 * outer_rise + m / 2.0 - 0.5
    G = (m - d) / 2.0 * outer_rise + (m - d) / 2.0 + d - 0.5
    E2 = H * F + (1 - H) * G
    # region 3: else constant m - 0.5
    E3 = m - 0.5
    cond1 = ang > (2.0 * pi / 3.0)
    cond2 = (ang >= (2.0 * pi / 3.0 - l)) & (~cond1)
    return jnp.where(cond1, E1, jnp.where(cond2, E2, E3))


# ---------------- Composable Mosaic-style LossTerms for TMOL -----------------


class LJEnergy(LossTerm):
    lj_global: LJGlobalParams
    lj_type: LJTypeParams
    bonded_path_length: Float[Array, "A A"]
    atom_mask: Optional[Array] = None

    def __call__(self, *args, key, **kwds) -> tuple[float, dict]:
        sequence = kwds.get("sequence") if "sequence" in kwds else args[0]
        output = cast(AbstractStructureOutput, kwds.get("output") if "output" in kwds else args[1])
        coords_all = getattr(output, "structure_coordinates", None)
        if coords_all is None:
            raise ValueError("Predictor output must expose structure_coordinates for LJEnergy.")
        if self.atom_mask is not None:
            keep = self.atom_mask > 0.5
            coords = coords_all[keep]
            keep_idx = jnp.where(keep)[0]
            bpl = self.bonded_path_length[jnp.ix_(keep_idx, keep_idx)]
            lj_type = LJTypeParams(
                lj_radius=self.lj_type.lj_radius[keep],
                lj_wdepth=self.lj_type.lj_wdepth[keep],
                is_acceptor=self.lj_type.is_acceptor[keep],
                is_donor=self.lj_type.is_donor[keep],
                is_hydroxyl=self.lj_type.is_hydroxyl[keep],
                is_polarh=self.lj_type.is_polarh[keep],
            )
        else:
            coords = coords_all
            bpl = self.bonded_path_length
            lj_type = self.lj_type
        pair_mask = kwds.get("pair_mask") if "pair_mask" in kwds else None
        E_lj, aux = lj_energy_pairs(coords, lj_type, self.lj_global, bpl, pair_mask=pair_mask)
        return E_lj, aux


class LJAttractiveEnergy(LossTerm):
    lj_global: LJGlobalParams
    lj_type: LJTypeParams
    bonded_path_length: Float[Array, "A A"]
    atom_mask: Optional[Array] = None

    def __call__(self, *args, key, **kwds) -> tuple[float, dict]:
        sequence = kwds.get("sequence") if "sequence" in kwds else args[0]
        output = cast(AbstractStructureOutput, kwds.get("output") if "output" in kwds else args[1])
        coords_all = getattr(output, "structure_coordinates", None)
        if coords_all is None:
            raise ValueError("Predictor output must expose structure_coordinates for LJAttractiveEnergy.")
        if self.atom_mask is not None:
            keep = self.atom_mask > 0.5
            coords = coords_all[keep]
            keep_idx = jnp.where(keep)[0]
            bpl = self.bonded_path_length[jnp.ix_(keep_idx, keep_idx)]
            lj_type = LJTypeParams(
                lj_radius=self.lj_type.lj_radius[keep],
                lj_wdepth=self.lj_type.lj_wdepth[keep],
                is_acceptor=self.lj_type.is_acceptor[keep],
                is_donor=self.lj_type.is_donor[keep],
                is_hydroxyl=self.lj_type.is_hydroxyl[keep],
                is_polarh=self.lj_type.is_polarh[keep],
            )
        else:
            coords = coords_all
            bpl = self.bonded_path_length
            lj_type = self.lj_type
        # reuse pair code and extract components
        diff = coords[:, None, :] - coords[None, :, :]
        dist = jnp.linalg.norm(diff + 1e-12, axis=-1)
        sigma = _lj_sigma(
            lj_type.is_donor[:, None],
            lj_type.is_hydroxyl[:, None],
            lj_type.is_polarh[:, None],
            lj_type.is_acceptor[:, None],
            lj_type.is_donor[None, :],
            lj_type.is_hydroxyl[None, :],
            lj_type.is_polarh[None, :],
            lj_type.is_acceptor[None, :],
            lj_type.lj_radius[:, None],
            lj_type.lj_radius[None, :],
            self.lj_global,
        )
        epsilon = jnp.sqrt(lj_type.lj_wdepth[:, None] * lj_type.lj_wdepth[None, :])
        Vatr, _ = _lj_split_attractive_repulsive(dist, sigma, epsilon)
        w = _connectivity_weight(bpl)
        A = coords.shape[0]
        base_mask = 1.0 - jnp.eye(A)
        pair_mask = kwds.get("pair_mask") if "pair_mask" in kwds else None
        if pair_mask is not None:
            base_mask = base_mask * pair_mask
        tri_mask = jnp.triu(jnp.ones_like(base_mask), k=1)
        pair_mask_full = base_mask * tri_mask
        Eatr = (w * Vatr * pair_mask_full).sum() * 1.0
        return Eatr, {"lj_atr": Eatr}


class LJRepulsiveEnergy(LossTerm):
    lj_global: LJGlobalParams
    lj_type: LJTypeParams
    bonded_path_length: Float[Array, "A A"]
    atom_mask: Optional[Array] = None

    def __call__(self, *args, key, **kwds) -> tuple[float, dict]:
        sequence = kwds.get("sequence") if "sequence" in kwds else args[0]
        output = cast(AbstractStructureOutput, kwds.get("output") if "output" in kwds else args[1])
        coords_all = getattr(output, "structure_coordinates", None)
        if coords_all is None:
            raise ValueError("Predictor output must expose structure_coordinates for LJRepulsiveEnergy.")
        if self.atom_mask is not None:
            keep = self.atom_mask > 0.5
            coords = coords_all[keep]
            keep_idx = jnp.where(keep)[0]
            bpl = self.bonded_path_length[jnp.ix_(keep_idx, keep_idx)]
            lj_type = LJTypeParams(
                lj_radius=self.lj_type.lj_radius[keep],
                lj_wdepth=self.lj_type.lj_wdepth[keep],
                is_acceptor=self.lj_type.is_acceptor[keep],
                is_donor=self.lj_type.is_donor[keep],
                is_hydroxyl=self.lj_type.is_hydroxyl[keep],
                is_polarh=self.lj_type.is_polarh[keep],
            )
        else:
            coords = coords_all
            bpl = self.bonded_path_length
            lj_type = self.lj_type
        diff = coords[:, None, :] - coords[None, :, :]
        dist = jnp.linalg.norm(diff + 1e-12, axis=-1)
        sigma = _lj_sigma(
            lj_type.is_donor[:, None],
            lj_type.is_hydroxyl[:, None],
            lj_type.is_polarh[:, None],
            lj_type.is_acceptor[:, None],
            lj_type.is_donor[None, :],
            lj_type.is_hydroxyl[None, :],
            lj_type.is_polarh[None, :],
            lj_type.is_acceptor[None, :],
            lj_type.lj_radius[:, None],
            lj_type.lj_radius[None, :],
            self.lj_global,
        )
        epsilon = jnp.sqrt(lj_type.lj_wdepth[:, None] * lj_type.lj_wdepth[None, :])
        _, Vrep = _lj_split_attractive_repulsive(dist, sigma, epsilon)
        w = _connectivity_weight(bpl)
        A = coords.shape[0]
        base_mask = 1.0 - jnp.eye(A)
        pair_mask = kwds.get("pair_mask") if "pair_mask" in kwds else None
        if pair_mask is not None:
            base_mask = base_mask * pair_mask
        tri_mask = jnp.triu(jnp.ones_like(base_mask), k=1)
        pair_mask_full = base_mask * tri_mask
        Erep = (w * Vrep * pair_mask_full).sum() * 1.0
        return Erep, {"lj_rep": Erep}

class ElecEnergy(LossTerm):
    elec_global: ElecGlobalParams
    elec_type: ElecTypeParams
    bonded_path_length: Float[Array, "A A"]
    atom_mask: Optional[Array] = None

    def __call__(self, *args, key, **kwds) -> tuple[float, dict]:
        sequence = kwds.get("sequence") if "sequence" in kwds else args[0]
        output = cast(AbstractStructureOutput, kwds.get("output") if "output" in kwds else args[1])
        coords_all = getattr(output, "structure_coordinates", None)
        if coords_all is None:
            raise ValueError("Predictor output must expose structure_coordinates for ElecEnergy.")
        if self.atom_mask is not None:
            keep = self.atom_mask > 0.5
            coords = coords_all[keep]
            keep_idx = jnp.where(keep)[0]
            bpl = self.bonded_path_length[jnp.ix_(keep_idx, keep_idx)]
            elec_type = ElecTypeParams(charges=self.elec_type.charges[keep])
        else:
            coords = coords_all
            bpl = self.bonded_path_length
            elec_type = self.elec_type
        pair_mask = kwds.get("pair_mask") if "pair_mask" in kwds else None
        E_elec, aux = elec_energy_pairs(coords, elec_type, self.elec_global, bpl, pair_mask=pair_mask)
        return E_elec, aux


class HBondEnergy(LossTerm):
    hb_global: HBondGlobalParams
    hb_pairs: HBondPairs
    atom_mask: Optional[Array] = None

    def __call__(self, *args, key, **kwds) -> tuple[float, dict]:
        sequence = kwds.get("sequence") if "sequence" in kwds else args[0]
        output = cast(AbstractStructureOutput, kwds.get("output") if "output" in kwds else args[1])
        coords_all = getattr(output, "structure_coordinates", None)
        if coords_all is None:
            raise ValueError("Predictor output must expose structure_coordinates for HBondEnergy.")
        pair_mask = kwds.get("pair_mask") if "pair_mask" in kwds else None
        if self.atom_mask is not None:
            keep = (self.atom_mask > 0.5).astype(jnp.int32)
            old_to_new = -jnp.ones((keep.shape[0],), dtype=jnp.int32)
            new_idx = jnp.cumsum(keep) - 1
            old_to_new = jnp.where(keep == 1, new_idx, old_to_new)
            def remap(x):
                return old_to_new[x]
            pairs = HBondPairs(
                D_idx=remap(self.hb_pairs.D_idx),
                H_idx=remap(self.hb_pairs.H_idx),
                A_idx=remap(self.hb_pairs.A_idx),
                B_idx=remap(self.hb_pairs.B_idx),
                B0_idx=remap(self.hb_pairs.B0_idx),
                acceptor_hybridization=self.hb_pairs.acceptor_hybridization,
                acceptor_weight=self.hb_pairs.acceptor_weight,
                donor_weight=self.hb_pairs.donor_weight,
                AHdist_coeffs=self.hb_pairs.AHdist_coeffs,
                AHdist_range=self.hb_pairs.AHdist_range,
                AHdist_bound=self.hb_pairs.AHdist_bound,
                cosAHD_coeffs=self.hb_pairs.cosAHD_coeffs,
                cosAHD_range=self.hb_pairs.cosAHD_range,
                cosAHD_bound=self.hb_pairs.cosAHD_bound,
                cosBAH_coeffs=self.hb_pairs.cosBAH_coeffs,
                cosBAH_range=self.hb_pairs.cosBAH_range,
                cosBAH_bound=self.hb_pairs.cosBAH_bound,
            )
            coords = coords_all[keep]
        else:
            pairs = self.hb_pairs
            coords = coords_all
        # Optional cross-chain filtering using pair_mask: keep pairs where D-A are allowed
        if pair_mask is not None and pairs.D_idx.shape[0] > 0:
            # Clip indices within range
            A = coords.shape[0]
            didx = jnp.clip(pairs.D_idx, 0, A - 1)
            aidx = jnp.clip(pairs.A_idx, 0, A - 1)
            keep_pairs = (pair_mask[didx, aidx] > 0.5)
            if keep_pairs.ndim > 0:
                kp = keep_pairs.astype(bool)
                if kp.size > 0 and kp.sum() < kp.size:
                    def sel(x):
                        return x[kp]
                    pairs = HBondPairs(
                        D_idx=sel(pairs.D_idx),
                        H_idx=sel(pairs.H_idx),
                        A_idx=sel(pairs.A_idx),
                        B_idx=sel(pairs.B_idx),
                        B0_idx=sel(pairs.B0_idx),
                        acceptor_hybridization=sel(pairs.acceptor_hybridization),
                        acceptor_weight=sel(pairs.acceptor_weight),
                        donor_weight=sel(pairs.donor_weight),
                        AHdist_coeffs=sel(pairs.AHdist_coeffs),
                        AHdist_range=sel(pairs.AHdist_range),
                        AHdist_bound=sel(pairs.AHdist_bound),
                        cosAHD_coeffs=sel(pairs.cosAHD_coeffs),
                        cosAHD_range=sel(pairs.cosAHD_range),
                        cosAHD_bound=sel(pairs.cosAHD_bound),
                        cosBAH_coeffs=sel(pairs.cosBAH_coeffs),
                        cosBAH_range=sel(pairs.cosBAH_range),
                        cosBAH_bound=sel(pairs.cosBAH_bound),
                    )
        E_hb = hbond_energy_pairs(coords, pairs, self.hb_global)
        return E_hb, {"hbond": E_hb}


# ----------------------------- Factory helpers ------------------------------


def make_lj_energy(
    *,
    lj_radius,
    lj_wdepth,
    is_acceptor,
    is_donor,
    is_hydroxyl,
    is_polarh,
    lj_hbond_dis: float,
    lj_hbond_OH_donor_dis: float,
    lj_hbond_hdis: float,
    bonded_path_length,
    atom_mask=None,
):
    ljg = LJGlobalParams(
        lj_hbond_dis=float(lj_hbond_dis),
        lj_hbond_OH_donor_dis=float(lj_hbond_OH_donor_dis),
        lj_hbond_hdis=float(lj_hbond_hdis),
    )
    ljt = LJTypeParams(
        lj_radius=jnp.asarray(lj_radius, dtype=jnp.float32),
        lj_wdepth=jnp.asarray(lj_wdepth, dtype=jnp.float32),
        is_acceptor=jnp.asarray(is_acceptor, dtype=bool),
        is_donor=jnp.asarray(is_donor, dtype=bool),
        is_hydroxyl=jnp.asarray(is_hydroxyl, dtype=bool),
        is_polarh=jnp.asarray(is_polarh, dtype=bool),
    )
    bpl = jnp.asarray(bonded_path_length, dtype=jnp.int32)
    am = None if atom_mask is None else jnp.asarray(atom_mask, dtype=jnp.float32)
    return LJEnergy(lj_global=ljg, lj_type=ljt, bonded_path_length=bpl, atom_mask=am)


def make_elec_energy(
    *,
    charges,
    D: float,
    D0: float,
    S: float,
    min_dis: float,
    max_dis: float,
    bonded_path_length,
    atom_mask=None,
):
    eg = ElecGlobalParams(D=float(D), D0=float(D0), S=float(S), min_dis=float(min_dis), max_dis=float(max_dis))
    et = ElecTypeParams(charges=jnp.asarray(charges, dtype=jnp.float32))
    bpl = jnp.asarray(bonded_path_length, dtype=jnp.int32)
    am = None if atom_mask is None else jnp.asarray(atom_mask, dtype=jnp.float32)
    return ElecEnergy(elec_global=eg, elec_type=et, bonded_path_length=bpl, atom_mask=am)


def make_hbond_energy(
    *,
    D_idx,
    H_idx,
    A_idx,
    B_idx,
    B0_idx,
    acceptor_hybridization,
    acceptor_weight,
    donor_weight,
    AHdist_coeffs,
    AHdist_range,
    AHdist_bound,
    cosAHD_coeffs,
    cosAHD_range,
    cosAHD_bound,
    cosBAH_coeffs,
    cosBAH_range,
    cosBAH_bound,
    hb_sp3_softmax_fade: float,
    hb_sp2_BAH180_rise: float,
    hb_sp2_range_span: float,
    hb_sp2_outer_width: float,
    max_ha_dis: float,
    atom_mask=None,
):
    hg = HBondGlobalParams(
        hb_sp3_softmax_fade=float(hb_sp3_softmax_fade),
        hb_sp2_BAH180_rise=float(hb_sp2_BAH180_rise),
        hb_sp2_range_span=float(hb_sp2_range_span),
        hb_sp2_outer_width=float(hb_sp2_outer_width),
        max_ha_dis=float(max_ha_dis),
    )
    pairs = HBondPairs(
        D_idx=jnp.asarray(D_idx, dtype=jnp.int32),
        H_idx=jnp.asarray(H_idx, dtype=jnp.int32),
        A_idx=jnp.asarray(A_idx, dtype=jnp.int32),
        B_idx=jnp.asarray(B_idx, dtype=jnp.int32),
        B0_idx=jnp.asarray(B0_idx, dtype=jnp.int32),
        acceptor_hybridization=jnp.asarray(acceptor_hybridization, dtype=jnp.int32),
        acceptor_weight=jnp.asarray(acceptor_weight, dtype=jnp.float32),
        donor_weight=jnp.asarray(donor_weight, dtype=jnp.float32),
        AHdist_coeffs=jnp.asarray(AHdist_coeffs, dtype=jnp.float64),
        AHdist_range=jnp.asarray(AHdist_range, dtype=jnp.float64),
        AHdist_bound=jnp.asarray(AHdist_bound, dtype=jnp.float64),
        cosAHD_coeffs=jnp.asarray(cosAHD_coeffs, dtype=jnp.float64),
        cosAHD_range=jnp.asarray(cosAHD_range, dtype=jnp.float64),
        cosAHD_bound=jnp.asarray(cosAHD_bound, dtype=jnp.float64),
        cosBAH_coeffs=jnp.asarray(cosBAH_coeffs, dtype=jnp.float64),
        cosBAH_range=jnp.asarray(cosBAH_range, dtype=jnp.float64),
        cosBAH_bound=jnp.asarray(cosBAH_bound, dtype=jnp.float64),
    )
    am = None if atom_mask is None else jnp.asarray(atom_mask, dtype=jnp.float32)
    return HBondEnergy(hb_global=hg, hb_pairs=pairs, atom_mask=am)


def make_tmol_linear_combination(*, terms_with_weights: list[tuple[float, LossTerm]]):
    if not terms_with_weights:
        return LinearCombination(l=[], weights=jnp.asarray([], dtype=jnp.float32))
    losses = [t for (_, t) in terms_with_weights]
    weights = jnp.asarray([w for (w, _) in terms_with_weights], dtype=jnp.float32)
    return LinearCombination(l=losses, weights=weights)


def make_beta2016_subset_loss(*,
    lj_atr_term: LJAttractiveEnergy,
    lj_rep_term: LJRepulsiveEnergy,
    lk_iso_term: LossTerm,
    elec_term: ElecEnergy,
    hbond_term: HBondEnergy | None = None,
    ref_term: LossTerm | None = None,
):
    # match tmol.score._non_memoized_beta2016 subset weights for terms we implement here
    tw = []
    tw.append((1.0, lj_atr_term))         # fa_ljatr
    tw.append((0.55, lj_rep_term))        # fa_ljrep
    tw.append((1.0, lk_iso_term))         # fa_lk (our LJ-based isotropic desolvation)
    tw.append((1.0, elec_term))           # fa_elec
    if hbond_term is not None:
        tw.append((1.0, hbond_term))      # hbond
    if ref_term is not None:
        tw.append((1.0, ref_term))       # ref
    return make_tmol_linear_combination(terms_with_weights=tw)


def build_interchain_pair_mask(*, atom_chain_ids: Array) -> Array:
    """Return A x A mask for interchain pairs (1.0 for cross-chain, 0.0 otherwise)."""
    c = atom_chain_ids.astype(jnp.int32)
    mask = (c[:, None] != c[None, :]).astype(jnp.float32)
    # zero self
    A = atom_chain_ids.shape[0]
    mask = mask * (1.0 - jnp.eye(A))
    return mask


# --------------------------- LK Isotropic (JAX) ------------------------------


@dataclass
class LKTypeParams:
    lk_dgfree: Float[Array, "A"]
    lk_lambda: Float[Array, "A"]
    lk_volume: Float[Array, "A"]


def _lk_f_desolv(dist: Array, lj_radius_i: Array, lk_dgfree_i: Array, lk_lambda_i: Array, lk_volume_j: Array) -> Array:
    # -lk_volume_j * lk_dgfree_i / (2 * pi^(3/2) * lk_lambda_i) * 1/(dist^2) * exp(-(dist-lj_radius_i)^2 / lk_lambda_i^2)
    pi_pow1p5 = 5.56832799683
    exp_val = jnp.exp(-((dist - lj_radius_i) * (dist - lj_radius_i)) / (lk_lambda_i * lk_lambda_i + 1e-12))
    return (
        -lk_volume_j
        * lk_dgfree_i
        / (2.0 * pi_pow1p5 * jnp.clip(lk_lambda_i, a_min=1e-12))
        / (jnp.clip(dist, a_min=1e-12) * jnp.clip(dist, a_min=1e-12))
        * exp_val
    )

def _lk_f_desolv_elementwise(dist: float, lj_radius_i: float, lk_dgfree_i: float, lk_lambda_i: float, lk_volume_j: float) -> float:
    """Elementwise version for gradient computation"""
    pi_pow1p5 = 5.56832799683
    exp_val = jnp.exp(-((dist - lj_radius_i) * (dist - lj_radius_i)) / (lk_lambda_i * lk_lambda_i + 1e-12))
    return (
        -lk_volume_j
        * lk_dgfree_i
        / (2.0 * pi_pow1p5 * jnp.maximum(lk_lambda_i, 1e-12))
        / (jnp.maximum(dist, 1e-12) * jnp.maximum(dist, 1e-12))
        * exp_val
    )


def _lk_isotropic_pair(
    dist: Array,
    bonded_path_length: Array,
    lj_sigma_ij: Array,
    lj_radius_i: Array,
    lk_dgfree_i: Array,
    lk_lambda_i: Array,
    lk_volume_j: Array,
) -> Array:
    # Windows
    d_min = lj_sigma_ij * 0.89
    cpoly_close_dmin = jnp.sqrt(jnp.maximum(d_min * d_min - 1.45, 0.01))
    cpoly_close_dmax = jnp.sqrt(d_min * d_min + 1.05)
    cpoly_far_dmin = 4.5
    cpoly_far_dmax = 6.0

    # Regions
    region_far = dist > cpoly_far_dmax
    region_far_tail = (dist > cpoly_far_dmin) & (dist <= cpoly_far_dmax)
    region_mid = (dist > cpoly_close_dmax) & (dist <= cpoly_far_dmin)
    region_close = (dist > cpoly_close_dmin) & (dist <= cpoly_close_dmax)
    region_core = dist <= cpoly_close_dmin

    # Values at boundaries
    f_close_max_V = _lk_f_desolv(cpoly_close_dmax, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j)
    # Faithful slopes at boundaries via elementwise grad with broadcasted shapes
    # Broadcast scalars/vectors to [N, N]
    b_cpoly_close_dmax = jnp.broadcast_to(cpoly_close_dmax, dist.shape)
    b_lj_radius_i = jnp.broadcast_to(lj_radius_i, dist.shape)
    b_lk_dgfree_i = jnp.broadcast_to(lk_dgfree_i, dist.shape)
    b_lk_lambda_i = jnp.broadcast_to(lk_lambda_i, dist.shape)
    b_lk_volume_j = jnp.broadcast_to(lk_volume_j, dist.shape)
    df_close_max = jax.vmap(jax.vmap(jax.grad(_lk_f_desolv_elementwise)))(
        b_cpoly_close_dmax, b_lj_radius_i, b_lk_dgfree_i, b_lk_lambda_i, b_lk_volume_j
    )
    f_far_min_V = _lk_f_desolv(cpoly_far_dmin, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j)
    b_cpoly_far_dmin = jnp.broadcast_to(cpoly_far_dmin, dist.shape)
    df_far_min = jax.vmap(jax.vmap(jax.grad(_lk_f_desolv_elementwise)))(
        b_cpoly_far_dmin, b_lj_radius_i, b_lk_dgfree_i, b_lk_lambda_i, b_lk_volume_j
    )
    f_core = _lk_f_desolv(d_min, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j)

    # Piecewise assembly
    val_far = jnp.zeros_like(dist)
    val_far_tail = _interpolate_to_zero(dist, cpoly_far_dmin, f_far_min_V, df_far_min, cpoly_far_dmax)
    val_mid = _lk_f_desolv(dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j)
    val_close = _interpolate(dist, cpoly_close_dmin, f_core, 0.0, cpoly_close_dmax, f_close_max_V, df_close_max)
    val_core = f_core

    lk = jnp.where(
        region_far,
        val_far,
        jnp.where(
            region_far_tail,
            val_far_tail,
            jnp.where(region_mid, val_mid, jnp.where(region_close, val_close, val_core)),
        ),
    )

    w = _connectivity_weight(bonded_path_length)
    return w * lk


def lk_isotropic_matrix(
    coords: Float[Array, "A 3"],
    bonded_path_length: Array,
    lj_global: LJGlobalParams,
    lj_type: LJTypeParams,
    lk_type: LKTypeParams,
    pair_mask: Optional[Array] = None,
    heavy_mask: Optional[Array] = None,
) -> Array:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = jnp.linalg.norm(diff + 1e-12, axis=-1)

    # Sigma and parameters per pair
    sigma = _lj_sigma(
        lj_type.is_donor[:, None],
        lj_type.is_hydroxyl[:, None],
        lj_type.is_polarh[:, None],
        lj_type.is_acceptor[:, None],
        lj_type.is_donor[None, :],
        lj_type.is_hydroxyl[None, :],
        lj_type.is_polarh[None, :],
        lj_type.is_acceptor[None, :],
        lj_type.lj_radius[:, None],
        lj_type.lj_radius[None, :],
        lj_global,
    )

    # Build matrices for i->j contribution
    lj_radius_i = lj_type.lj_radius[:, None]
    dg_i = lk_type.lk_dgfree[:, None]
    lam_i = lk_type.lk_lambda[:, None]
    vol_j = lk_type.lk_volume[None, :]

    E_ij = _lk_isotropic_pair(dist, bonded_path_length, sigma, lj_radius_i, dg_i, lam_i, vol_j)

    # j->i contribution
    lj_radius_j = lj_type.lj_radius[None, :]
    dg_j = lk_type.lk_dgfree[None, :]
    lam_j = lk_type.lk_lambda[None, :]
    vol_i = lk_type.lk_volume[:, None]
    E_ji = _lk_isotropic_pair(dist, bonded_path_length, sigma, lj_radius_j, dg_j, lam_j, vol_i)

    E = E_ij + E_ji

    # Apply masks: no self, optional heavy-atom-only, optional pair_mask
    A = coords.shape[0]
    base = 1.0 - jnp.eye(A)
    if heavy_mask is not None:
        h = (heavy_mask > 0.5).astype(jnp.float32)
        base = base * (h[:, None] * h[None, :])
    if pair_mask is not None:
        base = base * pair_mask

    # Unique pairs only
    tri_mask = jnp.triu(jnp.ones_like(base), k=1)
    return (E * base * tri_mask).sum()


class LKEnergy(LossTerm):
    lj_global: LJGlobalParams
    lj_type: LJTypeParams
    lk_type: LKTypeParams
    bonded_path_length: Float[Array, "A A"]
    atom_mask: Optional[Array] = None
    heavy_mask: Optional[Array] = None

    def __call__(self, *args, key, **kwds) -> tuple[float, dict]:
        sequence = kwds.get("sequence") if "sequence" in kwds else args[0]
        output = cast(AbstractStructureOutput, kwds.get("output") if "output" in kwds else args[1])
        coords_all = getattr(output, "structure_coordinates", None)
        if coords_all is None:
            raise ValueError("Predictor output must expose structure_coordinates for LKEnergy.")
        if self.atom_mask is not None:
            keep = self.atom_mask > 0.5
            coords = coords_all[keep]
            keep_idx = jnp.where(keep)[0]
            bpl = self.bonded_path_length[jnp.ix_(keep_idx, keep_idx)]
            lj_type = LJTypeParams(
                lj_radius=self.lj_type.lj_radius[keep],
                lj_wdepth=self.lj_type.lj_wdepth[keep],
                is_acceptor=self.lj_type.is_acceptor[keep],
                is_donor=self.lj_type.is_donor[keep],
                is_hydroxyl=self.lj_type.is_hydroxyl[keep],
                is_polarh=self.lj_type.is_polarh[keep],
            )
            lk_type = LKTypeParams(
                lk_dgfree=self.lk_type.lk_dgfree[keep],
                lk_lambda=self.lk_type.lk_lambda[keep],
                lk_volume=self.lk_type.lk_volume[keep],
            )
            hv = self.heavy_mask[keep] if self.heavy_mask is not None else None
        else:
            coords = coords_all
            bpl = self.bonded_path_length
            lj_type = self.lj_type
            lk_type = self.lk_type
            hv = self.heavy_mask

        pair_mask = kwds.get("pair_mask") if "pair_mask" in kwds else None
        E_lk = lk_isotropic_matrix(
            coords=coords,
            bonded_path_length=bpl,
            lj_global=self.lj_global,
            lj_type=lj_type,
            lk_type=lk_type,
            pair_mask=pair_mask,
            heavy_mask=hv,
        )
        return E_lk, {"lk": E_lk}


def make_lk_energy(
    *,
    # LJ sigma context
    lj_radius,
    is_acceptor,
    is_donor,
    is_hydroxyl,
    is_polarh,
    lj_hbond_dis: float,
    lj_hbond_OH_donor_dis: float,
    lj_hbond_hdis: float,
    # LK params
    lk_dgfree,
    lk_lambda,
    lk_volume,
    bonded_path_length,
    atom_mask=None,
    heavy_mask=None,
):
    ljg = LJGlobalParams(
        lj_hbond_dis=float(lj_hbond_dis),
        lj_hbond_OH_donor_dis=float(lj_hbond_OH_donor_dis),
        lj_hbond_hdis=float(lj_hbond_hdis),
    )
    ljt = LJTypeParams(
        lj_radius=jnp.asarray(lj_radius, dtype=jnp.float32),
        lj_wdepth=jnp.zeros_like(jnp.asarray(lj_radius, dtype=jnp.float32)),
        is_acceptor=jnp.asarray(is_acceptor, dtype=bool),
        is_donor=jnp.asarray(is_donor, dtype=bool),
        is_hydroxyl=jnp.asarray(is_hydroxyl, dtype=bool),
        is_polarh=jnp.asarray(is_polarh, dtype=bool),
    )
    lkt = LKTypeParams(
        lk_dgfree=jnp.asarray(lk_dgfree, dtype=jnp.float32),
        lk_lambda=jnp.asarray(lk_lambda, dtype=jnp.float32),
        lk_volume=jnp.asarray(lk_volume, dtype=jnp.float32),
    )
    bpl = jnp.asarray(bonded_path_length, dtype=jnp.int32)
    am = None if atom_mask is None else jnp.asarray(atom_mask, dtype=jnp.float32)
    hv = None if heavy_mask is None else jnp.asarray(heavy_mask, dtype=jnp.float32)
    return LKEnergy(lj_global=ljg, lj_type=ljt, lk_type=lkt, bonded_path_length=bpl, atom_mask=am, heavy_mask=hv)


# ----------------------- TMOL -> arrays (builder API) ------------------------


def build_atom_type_index_map(packed_block_types) -> np.ndarray:
    # Returns A-length array of atom type indices per atom in the pose (single pose)
    bt_atom_types = packed_block_types.atom_types  # [n_types, max_n_atoms]
    # Expect a PoseStack to provide mapping of blocks to block types and block_coord_offset
    raise NotImplementedError("Pass a PoseStack to builders to construct per-atom arrays.")


def build_ljlk_arrays_from_tmol(*, pose_stack, packed_block_types, ljlk_param_resolver):
    import torch
    # Assume single pose
    assert pose_stack.n_poses == 1, "Only single-pose supported in this builder."
    device = pose_stack.device
    # block types per residue
    bt_inds = pose_stack.block_type_ind64[0].to(torch.int64)  # [max_n_blocks]
    real_blocks = bt_inds >= 0
    bt_inds = bt_inds[real_blocks]
    # number of atoms per block type
    bt_n_atoms = packed_block_types.n_atoms[bt_inds]  # [n_blocks]
    # offsets for atom flattening
    block_coord_offset = pose_stack.block_coord_offset64[0][real_blocks]
    total_atoms = int((bt_n_atoms).sum().item())

    # Atom type index per block type/atom index
    bt_atom_types = packed_block_types.atom_types  # [n_types, max_n_atoms]
    max_n_atoms = bt_atom_types.shape[1]
    # Gather per-atom type index
    atom_type_index = torch.full((total_atoms,), -1, dtype=torch.int64, device=device)
    cursor = 0
    for b, bt in enumerate(bt_inds.tolist()):
        n_atoms_b = int(packed_block_types.n_atoms[bt])
        types_b = bt_atom_types[bt, :n_atoms_b]
        atom_type_index[cursor:cursor + n_atoms_b] = types_b
        cursor += n_atoms_b

    # Resolve LJ/LK per-type params
    tparams = ljlk_param_resolver.type_params
    def tg(x):
        return x.detach().cpu().numpy()
    lj_radius = tg(tparams.lj_radius[atom_type_index])
    lj_wdepth = tg(tparams.lj_wdepth[atom_type_index])
    lk_dgfree = tg(tparams.lk_dgfree[atom_type_index])
    lk_lambda = tg(tparams.lk_lambda[atom_type_index])
    lk_volume = tg(tparams.lk_volume[atom_type_index])
    is_acceptor = tg(tparams.is_acceptor[atom_type_index])
    is_donor = tg(tparams.is_donor[atom_type_index])
    is_hydroxyl = tg(tparams.is_hydroxyl[atom_type_index])
    is_polarh = tg(tparams.is_polarh[atom_type_index])

    # LJ global params
    g = ljlk_param_resolver.global_params
    lj_globals = dict(
        lj_hbond_dis=float(g.lj_hbond_dis.item()),
        lj_hbond_OH_donor_dis=float(g.lj_hbond_OH_donor_dis.item()),
        lj_hbond_hdis=float(g.lj_hbond_hdis.item()),
    )

    return dict(
        lj_radius=lj_radius,
        lj_wdepth=lj_wdepth,
        is_acceptor=is_acceptor,
        is_donor=is_donor,
        is_hydroxyl=is_hydroxyl,
        is_polarh=is_polarh,
        lk_dgfree=lk_dgfree,
        lk_lambda=lk_lambda,
        lk_volume=lk_volume,
        lj_globals=lj_globals,
        block_coord_offset=np.asarray(block_coord_offset.detach().cpu().numpy(), dtype=np.int64),
    )


def build_charges_from_tmol(*, pose_stack, packed_block_types, elec_param_resolver) -> np.ndarray:
    import torch
    assert pose_stack.n_poses == 1
    device = pose_stack.device
    bt_inds = pose_stack.block_type_ind64[0]
    real = bt_inds >= 0
    bt_inds = bt_inds[real]
    block_coord_offset = pose_stack.block_coord_offset64[0][real]
    # Per block type partial charges [n_types, max_n_atoms]
    bt_partial = packed_block_types.elec_partial_charge  # may need setup via ElecEnergyTerm
    if not hasattr(packed_block_types, "elec_partial_charge"):
        # Fallback: ask resolver for each active block type
        n_types = packed_block_types.n_types
        max_n_atoms = packed_block_types.max_n_atoms
        bt_partial = torch.zeros((n_types, max_n_atoms), dtype=torch.float32, device=device)
        for i, bt in enumerate(packed_block_types.active_block_types):
            bt_partial[i, : bt.n_atoms] = torch.tensor(
                elec_param_resolver.get_partial_charges_for_block(bt), device=device, dtype=torch.float32
            )

    total_atoms = int(torch.sum(packed_block_types.n_atoms[bt_inds]).item())
    charges = torch.zeros((total_atoms,), dtype=torch.float32, device=device)
    cursor = 0
    for bt in bt_inds.tolist():
        n_atoms_b = int(packed_block_types.n_atoms[bt])
        charges[cursor:cursor + n_atoms_b] = bt_partial[bt, :n_atoms_b]
        cursor += n_atoms_b
    return charges.detach().cpu().numpy()


def build_bonded_path_length_from_tmol(*, pose_stack, packed_block_types) -> np.ndarray:
    import torch
    assert pose_stack.n_poses == 1
    device = pose_stack.device
    bt_inds = pose_stack.block_type_ind64[0]
    real = bt_inds >= 0
    bt_inds = bt_inds[real]
    n_blocks = int(real.sum().item())

    # block_type tensors
    bt_path_distance = packed_block_types.bond_separation  # [n_types, max_n_atoms, max_n_atoms]
    bt_n_atoms = packed_block_types.n_atoms
    bt_conn_atoms = packed_block_types.conn_atom  # [n_types, max_n_conn]
    inter_bsep = pose_stack.inter_block_bondsep64[0]  # [max_n_blocks, max_n_blocks, max_n_conn, max_n_conn]

    # flatten atoms total
    total_atoms = int(torch.sum(bt_n_atoms[bt_inds]).item())
    B = np.full((total_atoms, total_atoms), 100, dtype=np.int32)
    # Intra-res pairs from path_distance
    cursor = 0
    for bi, bt in enumerate(bt_inds.tolist()):
        nA = int(bt_n_atoms[bt])
        pd = bt_path_distance[bt, :nA, :nA].detach().cpu().numpy().astype(np.int32)
        B[cursor:cursor + nA, cursor:cursor + nA] = pd
        cursor += nA

    # Inter-res pairs via min over connection atoms per block pair
    # Precompute atom ranges per block
    starts = np.zeros((n_blocks,), dtype=np.int64)
    lens = np.zeros((n_blocks,), dtype=np.int64)
    s = 0
    for bi, bt in enumerate(bt_inds.tolist()):
        starts[bi] = s
        nA = int(bt_n_atoms[bt])
        lens[bi] = nA
        s += nA

    max_n_conn = bt_conn_atoms.shape[1]
    bt_conn_atoms_np = bt_conn_atoms.detach().cpu().numpy().astype(np.int64)
    inter_bsep_np = inter_bsep.detach().cpu().numpy().astype(np.int32)
    bt_inds_np = bt_inds.detach().cpu().numpy().astype(np.int64)

    for bi in range(n_blocks):
        for bj in range(n_blocks):
            if bi == bj:
                continue
            bt_i = bt_inds_np[bi]
            bt_j = bt_inds_np[bj]
            nAi, nAj = int(lens[bi]), int(lens[bj])
            si, sj = int(starts[bi]), int(starts[bj])
            # connection atom indices (may be -1 if fewer connections)
            conn_i = bt_conn_atoms_np[bt_i]
            conn_j = bt_conn_atoms_np[bt_j]
            # valid connections are those != -1
            valid_i = conn_i >= 0
            valid_j = conn_j >= 0
            idx_i = conn_i[valid_i]
            idx_j = conn_j[valid_j]
            ni = idx_i.shape[0]
            nj = idx_j.shape[0]
            if ni == 0 or nj == 0:
                continue
            # per-atom to connection distances within blocks
            pd_i = bt_path_distance[bt_i, :nAi, :nAi].detach().cpu().numpy().astype(np.int32)
            pd_j = bt_path_distance[bt_j, :nAj, :nAj].detach().cpu().numpy().astype(np.int32)
            # inter-block connection separations for these blocks
            # pose_stack.inter_block_bondsep indexed by block indices within pose ordering
            inter_ij = inter_bsep_np[bi, bj]
            # For each atom a in i and b in j, compute min over connections c,d: pd_i[a,c] + inter_ij[c', d'] + pd_j[b,d]
            # where c' and d' are indices into max_n_conn; idx_i and idx_j give atom indices; we need their conn indices positions
            # Approximating by enumerating valid connection positions
            conn_pos_i = np.where(valid_i)[0]
            conn_pos_j = np.where(valid_j)[0]
            # Preextract inter separations for the valid connection positions
            inter_sel = inter_ij[np.ix_(conn_pos_i, conn_pos_j)]
            # Compute for each atom pair
            for a in range(nAi):
                ai = si + a
                di = pd_i[a, idx_i][:, None]  # shape [ni, 1]
                for b in range(nAj):
                    bj2 = sj + b
                    dj = pd_j[b, idx_j][None, :]  # shape [1, nj]
                    sep = di + inter_sel + dj
                    B[ai, bj2] = int(np.minimum(B[ai, bj2], int(sep.min())))

    return B


def build_elec_repr_bpl_from_tmol(*, pose_stack, packed_block_types) -> np.ndarray:
    import torch
    assert pose_stack.n_poses == 1
    device = pose_stack.device
    bt_inds = pose_stack.block_type_ind64[0]
    real = bt_inds >= 0
    bt_inds = bt_inds[real]
    n_blocks = int(real.sum().item())

    # Required tensors prepared by TMOL ElecEnergyTerm.setup_packed_block_types
    if not (hasattr(packed_block_types, "elec_inter_repr_path_distance") and hasattr(packed_block_types, "elec_intra_repr_path_distance")):
        raise RuntimeError("PackedBlockTypes missing elec representative path distance tensors. Ensure ElecEnergyTerm.setup_packed_block_types was called.")

    bt_inter_rpd = packed_block_types.elec_inter_repr_path_distance  # [n_types, max_n_atoms, max_n_atoms]
    bt_intra_rpd = packed_block_types.elec_intra_repr_path_distance  # [n_types, max_n_atoms, max_n_atoms]
    bt_n_atoms = packed_block_types.n_atoms
    bt_conn_atoms = packed_block_types.conn_atom  # [n_types, max_n_conn]
    inter_sep = pose_stack.inter_block_bondsep64[0]  # [max_n_blocks, max_n_blocks, max_n_conn, max_n_conn]

    # Flatten atoms across blocks
    total_atoms = int(torch.sum(bt_n_atoms[bt_inds]).item())
    R = np.full((total_atoms, total_atoms), 100, dtype=np.int32)

    # Intra-block representative distances
    cursor = 0
    for bi, bt in enumerate(bt_inds.tolist()):
        nA = int(bt_n_atoms[bt])
        intra = bt_intra_rpd[bt, :nA, :nA].detach().cpu().numpy().astype(np.int32)
        R[cursor:cursor + nA, cursor:cursor + nA] = intra
        cursor += nA

    # Precompute atom ranges per block
    starts = np.zeros((n_blocks,), dtype=np.int64)
    lens = np.zeros((n_blocks,), dtype=np.int64)
    c = 0
    for i, bt in enumerate(bt_inds.tolist()):
        nA = int(bt_n_atoms[bt])
        starts[i] = c
        lens[i] = nA
        c += nA

    # Inter-block: min over connection atoms of inter_rpd_k[ck, i] + inter_rpd_l[cl, j] + sep(ck, cl)
    for bi in range(n_blocks):
        bti = int(bt_inds[bi])
        ni = int(lens[bi])
        start_i = int(starts[bi])
        conn_i = bt_conn_atoms[bti].detach().cpu().numpy().astype(np.int32)
        valid_conn_i = conn_i[conn_i >= 0]
        for bj in range(bi + 1, n_blocks):
            btj = int(bt_inds[bj])
            nj = int(lens[bj])
            start_j = int(starts[bj])
            conn_j = bt_conn_atoms[btj].detach().cpu().numpy().astype(np.int32)
            valid_conn_j = conn_j[conn_j >= 0]

            # Prepare arrays
            inter_i = bt_inter_rpd[bti, :ni, :ni].detach().cpu().numpy().astype(np.int32)
            inter_j = bt_inter_rpd[btj, :nj, :nj].detach().cpu().numpy().astype(np.int32)
            # We need inter_rpd_k[ck, i] so index first dim by ck
            # Build per-atom arrays
            rblock = np.full((ni, nj), 100, dtype=np.int32)
            for idx_ck, ck in enumerate(valid_conn_i.tolist()):
                for idx_cl, cl in enumerate(valid_conn_j.tolist()):
                    sep = int(inter_sep[bi, bj, idx_ck, idx_cl].item())
                    # inter_rpd_k: rows=atoms (a), cols=rep(b); we want [ck, i]
                    term_i = inter_i[ck, :ni][:, None]  # shape [ni,1]
                    term_j = inter_j[cl, :nj][None, :]  # shape [1,nj]
                    cand = term_i + term_j + sep
                    rblock = np.minimum(rblock, cand)

            # Write symmetric blocks
            R[start_i:start_i + ni, start_j:start_j + nj] = rblock
            R[start_j:start_j + nj, start_i:start_i + ni] = rblock.T

    return R

def build_hbond_pairs_from_tmol(*, pose_stack, packed_block_types, hbond_energy_term) -> dict:
    import torch
    assert pose_stack.n_poses == 1
    device = pose_stack.device
    pbt = packed_block_types
    # Ensure HBondEnergyTerm has setup the packed_block_types so hbpbt_params exist
    if not hasattr(pbt, "hbpbt_params"):
        hbond_energy_term.setup_packed_block_types(pbt)
    # Extract tiling tables; we will flatten tile layout to per-atom indices
    tile_n_donH = pbt.hbpbt_params.tile_n_donH    # [n_types, n_tiles]
    tile_n_acc = pbt.hbpbt_params.tile_n_acc      # [n_types, n_tiles]
    tile_donH_inds = pbt.hbpbt_params.tile_donH_inds  # [n_types, n_tiles, TILE]
    tile_acc_inds = pbt.hbpbt_params.tile_acc_inds    # [n_types, n_tiles, TILE]
    tile_donor_type = pbt.hbpbt_params.tile_donorH_type
    tile_acceptor_type = pbt.hbpbt_params.tile_acceptor_type
    tile_acc_hybrid = pbt.hbpbt_params.tile_acceptor_hybridization
    is_hydrogen = pbt.hbpbt_params.is_hydrogen  # [n_types, max_n_atoms]

    # Build global params and pair tables from HBondEnergyTerm’s db
    pair_params = hbond_energy_term.hb_param_db.pair_param_table  # [n_don_types, n_acc_types]
    pair_polys = hbond_energy_term.hb_param_db.pair_poly_table
    g = hbond_energy_term.hb_param_db.global_param_table

    # TMOL packs HBond global params into a parameter vector: [hb_sp2_range_span, hb_sp2_BAH180_rise,
    # hb_sp2_outer_width, hb_sp3_softmax_fade, threshold_distance, max_ahdis]
    # Map by index to preserve fidelity
    gv = g  # torch.nn.Parameter with shape [1, 6]
    hb_globals = dict(
        hb_sp2_range_span=float(gv[0, 0].item()),
        hb_sp2_BAH180_rise=float(gv[0, 1].item()),
        hb_sp2_outer_width=float(gv[0, 2].item()),
        hb_sp3_softmax_fade=float(gv[0, 3].item()),
        max_ha_dis=float(gv[0, 5].item()),
    )

    # Pose info
    bt_inds = pose_stack.block_type_ind64[0]  # [max_n_blocks]
    real_blocks = (bt_inds >= 0)
    bt_inds = bt_inds[real_blocks]
    block_coord_offset = pose_stack.block_coord_offset64[0][real_blocks]
    n_blocks = int(bt_inds.shape[0])

    # Per-block number of atoms
    bt_n_atoms = pbt.n_atoms  # [n_types]

    # Bond graph per block type
    n_all_bonds = pbt.n_all_bonds  # [n_types]
    all_bonds = pbt.all_bonds      # [n_types, max_n_bonds, 2]
    atom_all_bond_ranges = pbt.atom_all_bond_ranges  # [n_types, max_n_atoms, 2]

    tile_size = hbond_energy_term.tile_size
    max_tiles = tile_n_donH.shape[1]

    # Collect donors and acceptors per block: list of tuples
    donors = []  # list of dicts with H_local, D_local, dt, block_index
    acceptors = []  # list of dicts with A_local, B_local, B0_local, at, hyb, block_index

    def first_heavy_neighbor(bt, atom_local):
        rng = atom_all_bond_ranges[bt, atom_local]
        start = int(rng[0].item())
        end = int(rng[1].item())
        for k in range(start, end):
            a = int(all_bonds[bt, k, 0].item())
            b = int(all_bonds[bt, k, 1].item())
            other = b if a == atom_local else (a if b == atom_local else -1)
            if other >= 0 and not bool(is_hydrogen[bt, other].item()):
                return other
        return -1

    def two_heavy_neighbors(bt, atom_local):
        rng = atom_all_bond_ranges[bt, atom_local]
        start = int(rng[0].item())
        end = int(rng[1].item())
        res = []
        for k in range(start, end):
            a = int(all_bonds[bt, k, 0].item())
            b = int(all_bonds[bt, k, 1].item())
            other = b if a == atom_local else (a if b == atom_local else -1)
            if other >= 0 and not bool(is_hydrogen[bt, other].item()):
                res.append(other)
            if len(res) == 2:
                break
        if len(res) == 0:
            return -1, -1
        if len(res) == 1:
            return res[0], res[0]
        return res[0], res[1]

    for bi in range(n_blocks):
        bt = int(bt_inds[bi].item())
        offset = int(block_coord_offset[bi].item())
        n_atoms_b = int(bt_n_atoms[bt].item())
        n_tiles_b = min(max_tiles, int((n_atoms_b + tile_size - 1) // tile_size))
        for t in range(n_tiles_b):
            start_atom = t * tile_size
            # donors
            nd = int(tile_n_donH[bt, t].item())
            if nd > 0:
                for k in range(nd):
                    h_in_tile = int(tile_donH_inds[bt, t, k].item())
                    if h_in_tile < 0:
                        continue
                    h_local = start_atom + h_in_tile
                    if h_local >= n_atoms_b:
                        continue
                    dt = int(tile_donor_type[bt, t, k].item())
                    d_local = first_heavy_neighbor(bt, h_local)
                    if d_local < 0:
                        continue
                    donors.append({"block": bi, "H": h_local, "D": d_local, "dt": dt, "offset": offset})
            # acceptors
            na = int(tile_n_acc[bt, t].item())
            if na > 0:
                for k in range(na):
                    a_in_tile = int(tile_acc_inds[bt, t, k].item())
                    if a_in_tile < 0:
                        continue
                    a_local = start_atom + a_in_tile
                    if a_local >= n_atoms_b:
                        continue
                    at = int(tile_acceptor_type[bt, t, k].item())
                    hyb = int(tile_acc_hybrid[bt, t, k].item())
                    b_local, b0_local = two_heavy_neighbors(bt, a_local)
                    if b_local < 0:
                        continue
                    acceptors.append({
                        "block": bi,
                        "A": a_local,
                        "B": b_local,
                        "B0": b0_local if b0_local >= 0 else b_local,
                        "at": at,
                        "hyb": hyb,
                        "offset": offset,
                    })

    # Enumerate pairs: intra and inter
    D_idx = []
    H_idx = []
    A_idx = []
    B_idx = []
    B0_idx = []
    acc_hyb = []
    acc_w = []
    don_w = []
    AH_coeffs = []
    cosAHD_coeffs = []
    cosBAH_coeffs = []

    # Shared ranges/bounds (assume consistent across types) – derive from packed vector
    vec0 = pair_polys[0, 0, :].detach().cpu().numpy()
    L = vec0.shape[0]
    assert L % 3 == 0, "Unexpected HBond pair_poly_table packing"
    block_len = L // 3
    coeff_len = block_len - 5  # [coeffs][pad(1)][range(2)][bound(2)]
    # AH block
    AH_range = vec0[coeff_len + 1 : coeff_len + 3]
    AH_bound = vec0[coeff_len + 3 : coeff_len + 5]
    # cosBAH block
    BAH_start = block_len
    BAH_range = vec0[BAH_start + coeff_len + 1 : BAH_start + coeff_len + 3]
    BAH_bound = vec0[BAH_start + coeff_len + 3 : BAH_start + coeff_len + 5]
    # cosAHD block
    AHD_start = 2 * block_len
    AHD_range = vec0[AHD_start + coeff_len + 1 : AHD_start + coeff_len + 3]
    AHD_bound = vec0[AHD_start + coeff_len + 3 : AHD_start + coeff_len + 5]

    def add_pair(don, acc):
        dt = don["dt"]; at = acc["at"]
        pp = pair_params[dt][at]
        pol_vec = pair_polys[dt, at, :].detach().cpu().numpy()
        Lp = pol_vec.shape[0]
        # Each of three blocks packs: [coeffs][pad(1)][range(2)][bound(2)] -> +5
        # If coefficients are absent for this pair, Lp may be 15 and cl=0; skip
        cl = (Lp - 15) // 3 if Lp >= 15 else -1
        if cl is None or cl <= 0:
            return
        bl = cl + 5
        D_idx.append(int(don["offset"]) + int(don["D"]))
        H_idx.append(int(don["offset"]) + int(don["H"]))
        A_idx.append(int(acc["offset"]) + int(acc["A"]))
        B_idx.append(int(acc["offset"]) + int(acc["B"]))
        B0_idx.append(int(acc["offset"]) + int(acc["B0"]))
        acc_hyb.append(int(pp.acceptor_hybridization.item()))
        acc_w.append(float(pp.acceptor_weight.item()))
        don_w.append(float(pp.donor_weight.item()))
        # Coefficients slices
        AH_coeffs.append(pol_vec[0 : cl])
        cosBAH_coeffs.append(pol_vec[bl : bl + cl])
        cosAHD_coeffs.append(pol_vec[2 * bl : 2 * bl + cl])

    # Optional distance precheck using pose coordinates
    coords_t = pose_stack.coords[0]  # [N, 3]
    max_ha = hb_globals["max_ha_dis"]
    for don in donors:
        H_glob = int(don["offset"]) + int(don["H"])
        H_xyz = coords_t[H_glob].detach().cpu().numpy()
        for acc in acceptors:
            A_glob = int(acc["offset"]) + int(acc["A"])
            A_xyz = coords_t[A_glob].detach().cpu().numpy()
            d = float(np.linalg.norm(H_xyz - A_xyz))
            if not np.isfinite(d) or d > max_ha:
                continue
            add_pair(don, acc)

    return dict(
        hb_globals=hb_globals,
        D_idx=np.asarray(D_idx, dtype=np.int32),
        H_idx=np.asarray(H_idx, dtype=np.int32),
        A_idx=np.asarray(A_idx, dtype=np.int32),
        B_idx=np.asarray(B_idx, dtype=np.int32),
        B0_idx=np.asarray(B0_idx, dtype=np.int32),
        acceptor_hybridization=np.asarray(acc_hyb, dtype=np.int32),
        acceptor_weight=np.asarray(acc_w, dtype=np.float32),
        donor_weight=np.asarray(don_w, dtype=np.float32),
        AHdist_coeffs=np.asarray(AH_coeffs, dtype=np.float64),
        AHdist_range=AH_range.astype(np.float64),
        AHdist_bound=AH_bound.astype(np.float64),
        cosAHD_coeffs=np.asarray(cosAHD_coeffs, dtype=np.float64),
        cosAHD_range=AHD_range.astype(np.float64),
        cosAHD_bound=AHD_bound.astype(np.float64),
        cosBAH_coeffs=np.asarray(cosBAH_coeffs, dtype=np.float64),
        cosBAH_range=BAH_range.astype(np.float64),
        cosBAH_bound=BAH_bound.astype(np.float64),
    )


