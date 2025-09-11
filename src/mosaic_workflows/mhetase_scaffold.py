import numpy as np
from typing import cast
import jax
import jax.numpy as jnp
import copy
import functools
import os
from pathlib import Path

from mosaic.common import LossTerm
from mosaic_workflows.design import run_workflow
from mosaic_workflows.optimizers import adamw_logits_adapter as adamw_logits, sgd_logits_adapter as sgd_logits
from mosaic_workflows.transforms import (
    temperature_on_logits,
    e_soft_on_logits,
    gradient_normalizer,
    per_position_allowed_tokens,
    position_mask,
)
# Lazy import boltz2 pieces at use-time to avoid heavy deps during simple unit tests
import mosaic.losses.structure_prediction as sp
from mosaic.losses.transformations import ClippedGradient, NormedGradient
# from mosaic.losses.wrappers import RiskWrappedLoss  - iffy implementation atm

"""MHETase motif scaffolding workflow (clean, mosaic-style).

Phases:
- warmup: hallucination entropy + pLDDT (stabilize backbone)
- anneal: full objective (motif geometry + contacts + pLDDT) with temperature decay

Step counts are assigned by the caller (e.g., Modal app) and can be overridden externally.
"""

@functools.lru_cache(maxsize=1)
def _get_mpnn():
    # Lazy import to avoid heavy deps during light tests
    from mosaic.proteinmpnn.mpnn import ProteinMPNN as _ProteinMPNN
    return _ProteinMPNN()


class CatalyticMotifLoss(LossTerm):
    def __init__(
        self,
        *,
        ser_pos: int,
        his_pos: int,
        asp_pos: int,
        oxyanion_positions: tuple[int, ...] | None = None,
        allowed_ser: tuple[str, ...] = ("S",),
        allowed_his: tuple[str, ...] = ("H",),
        allowed_asp: tuple[str, ...] = ("D", "E"),
        oxyanion_allowed: tuple[str, ...] = ("G", "S", "A"),
        w_identity_ser: float = 1.0,
        w_identity_his: float = 1.0,
        w_identity_asp: float = 1.0,
        w_identity_oxyanion: float = 0.25,
    ):
        object.__setattr__(self, "ser_pos", int(ser_pos))
        object.__setattr__(self, "his_pos", int(his_pos))
        object.__setattr__(self, "asp_pos", int(asp_pos))
        object.__setattr__(self, "oxyanion_positions", oxyanion_positions)
        object.__setattr__(self, "allowed_ser", allowed_ser)
        object.__setattr__(self, "allowed_his", allowed_his)
        object.__setattr__(self, "allowed_asp", allowed_asp)
        object.__setattr__(self, "oxyanion_allowed", oxyanion_allowed)
        object.__setattr__(self, "w_identity_ser", float(w_identity_ser))
        object.__setattr__(self, "w_identity_his", float(w_identity_his))
        object.__setattr__(self, "w_identity_asp", float(w_identity_asp))
        object.__setattr__(self, "w_identity_oxyanion", float(w_identity_oxyanion))

    def __call__(self, x, key, **kwds):
        vocab = "ARNDCQEGHILKMFPSTWYV"
        idx_map = {a: i for i, a in enumerate(vocab)}

        def pos_penalty(pos: int, allowed: tuple[str, ...]):
            ai = jnp.array([idx_map[a] for a in allowed], dtype=jnp.int32)
            return 1.0 - jnp.clip(jnp.take(x[int(pos)], ai).sum(), 0.0, 1.0)

        p_ser = pos_penalty(self.ser_pos, self.allowed_ser)
        p_his = pos_penalty(self.his_pos, self.allowed_his)
        p_asp = pos_penalty(self.asp_pos, self.allowed_asp)

        loss = (
            self.w_identity_ser * p_ser
            + self.w_identity_his * p_his
            + self.w_identity_asp * p_asp
        )

        if self.oxyanion_positions is not None and len(self.oxyanion_positions) > 0:
            o_pen = 0.0
            for pos in self.oxyanion_positions:
                o_pen = o_pen + pos_penalty(int(pos), self.oxyanion_allowed)
            o_pen = o_pen / float(len(self.oxyanion_positions))
            loss = loss + self.w_identity_oxyanion * o_pen

        aux = {
            "mhetase_motif": {
                "p_ser": p_ser,
                "p_his": p_his,
                "p_asp": p_asp,
            }
        }
        return loss, aux


class MotifStructureRMSD(LossTerm):
    def __init__(
        self,
        *,
        motif_positions: tuple[int, ...],
        target_ca: np.ndarray | None = None,
        target_backbone: np.ndarray | None = None,
    ):
        """Motif RMSD after superposition.

        Uses backbone (N, CA, C) RMSD if target_backbone is provided (shape [K,3,3]).
        Otherwise falls back to CA-only RMSD with target_ca of shape [K,3].
        """
        assert (target_backbone is not None) or (target_ca is not None), "Provide target_backbone or target_ca"
        if target_backbone is not None:
            assert isinstance(target_backbone, np.ndarray)
            assert target_backbone.shape[-2:] == (3, 3)
            K = int(target_backbone.shape[0])
        else:
            assert isinstance(target_ca, np.ndarray)
            K = int(target_ca.shape[0])
        assert len(motif_positions) == K, "motif_positions and target motif size mismatch"
        object.__setattr__(self, "motif_positions", tuple(int(p) for p in motif_positions))
        object.__setattr__(self, "target_ca", None if target_ca is None else jnp.asarray(target_ca, dtype=jnp.float32))
        object.__setattr__(self, "target_backbone", None if target_backbone is None else jnp.asarray(target_backbone, dtype=jnp.float32))

    @staticmethod
    def _kabsch_align(P: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
        """Return aligned P given P, Q of shape [K,3]."""
        P_cent = P - P.mean(axis=0, keepdims=True)
        Q_cent = Q - Q.mean(axis=0, keepdims=True)
        H = P_cent.T @ Q_cent
        U, S, Vt = jnp.linalg.svd(H)
        R = Vt.T @ U.T
        # Ensure right-handed rotation
        det = jnp.linalg.det(R)
        R = jnp.where(det < 0, Vt.T @ (jnp.diag(jnp.array([1.0, 1.0, -1.0])) @ U.T), R)
        P_aln = (P_cent @ R) + Q.mean(axis=0, keepdims=True)
        return P_aln

    def __call__(self, sequence, output, key):
        # Extract backbone coords for the binder
        bb = getattr(output, "backbone_coordinates")  # [N, 4, 3] in N,CA,C,O order
        binder_len = sequence.shape[0]
        sel = jnp.asarray(self.motif_positions, dtype=jnp.int32)
        safe_sel = jnp.clip(sel, 0, jnp.maximum(binder_len - 1, 0))

        if self.target_backbone is not None:
            # Use N,CA,C points per residue → flatten to [K*3,3]
            nca = bb[:binder_len, [0, 1, 2], :]  # [L,3,3]
            P_pts = nca[safe_sel].reshape(-1, 3)
            Q_pts = self.target_backbone.reshape(-1, 3)
        else:
            # Fallback to CA-only
            ca = bb[:binder_len, 1, :]
            P_pts = ca[safe_sel]
            Q_pts = self.target_ca

        # Weighted/Masked Kabsch without boolean indexing
        finite = jnp.all(jnp.isfinite(P_pts), axis=-1)
        w = finite.astype(jnp.float32)
        sum_w = w.sum()
        P = jnp.nan_to_num(P_pts, nan=0.0, posinf=0.0, neginf=0.0)

        def _weighted_kabsch(_):
            denom = sum_w + 1e-8
            P_mean = (P * w[:, None]).sum(axis=0) / denom
            Q_mean = (Q_pts * w[:, None]).sum(axis=0) / denom
            P_cent = P - P_mean
            Q_cent = Q_pts - Q_mean
            H = P_cent.T @ (Q_cent * w[:, None])
            U, S, Vt = jnp.linalg.svd(H, full_matrices=False)
            R = Vt.T @ U.T
            det = jnp.linalg.det(R)
            R = jnp.where(det < 0, Vt.T @ (jnp.diag(jnp.array([1.0, 1.0, -1.0])) @ U.T), R)
            P_aln = (P_cent @ R) + Q_mean
            sq = jnp.sum((P_aln - Q_pts) ** 2, axis=-1)
            rmsd_val = jnp.sqrt((w * sq).sum() / denom + 1e-8)
            return rmsd_val

        rmsd = jax.lax.cond(sum_w >= 3.0, _weighted_kabsch, lambda _: jnp.asarray(0.0, dtype=jnp.float32), operand=None)
        return rmsd, {"motif_rmsd": rmsd, "motif_finite": sum_w}


class MotifPseudoCBRMSD(LossTerm):
    def __init__(self, *, motif_positions: tuple[int, ...], target_backbone: np.ndarray):
        """Pseudo-CB RMSD after superposition using N,CA,C to construct CB.

        target_backbone: [K,3,3] for residues in motif (N,CA,C order)
        """
        assert isinstance(target_backbone, np.ndarray)
        assert target_backbone.shape[-2:] == (3, 3)
        K = int(target_backbone.shape[0])
        assert len(motif_positions) == K
        object.__setattr__(self, "motif_positions", tuple(int(p) for p in motif_positions))
        object.__setattr__(self, "target_backbone", jnp.asarray(target_backbone, dtype=jnp.float32))

    @staticmethod
    def _pseudo_cb(n: jnp.ndarray, ca: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        # Construct virtual CB from backbone following idealized geometry
        # Based on common reconstruction: unit vectors and their sum with out-of-plane component
        b1 = n - ca
        b2 = c - ca
        b1 = b1 / (jnp.linalg.norm(b1) + 1e-8)
        b2 = b2 / (jnp.linalg.norm(b2) + 1e-8)
        # out-of-plane component
        b3 = jnp.cross(b1, b2)
        b3 = b3 / (jnp.linalg.norm(b3) + 1e-8)
        # coefficients approximate tetrahedral geometry
        # scale to typical CA-CB distance ~1.522 Å
        dir_vec = -0.58273431 * b1 - 0.43526509 * b2 + 0.68819615 * b3
        dir_vec = dir_vec / (jnp.linalg.norm(dir_vec) + 1e-8)
        return ca + 1.522 * dir_vec

    @staticmethod
    def _pseudo_cb_batch(bb_ncac: jnp.ndarray) -> jnp.ndarray:
        # bb_ncac: [...,3,3] (N,CA,C)
        n = bb_ncac[..., 0, :]
        ca = bb_ncac[..., 1, :]
        c = bb_ncac[..., 2, :]
        fn = lambda N, CA, C: MotifPseudoCBRMSD._pseudo_cb(N, CA, C)
        return jax.vmap(fn)(n, ca, c)

    def __call__(self, sequence, output, key):
        bb = getattr(output, "backbone_coordinates")  # [N,4,3]
        binder_len = sequence.shape[0]
        sel = jnp.asarray(self.motif_positions, dtype=jnp.int32)
        safe_sel = jnp.clip(sel, 0, jnp.maximum(binder_len - 1, 0))

        # Build [K,3,3] for predicted N,CA,C
        nca = bb[:binder_len, [0, 1, 2], :]  # [L,3,3]
        P_ncac = nca[safe_sel]
        Q_ncac = self.target_backbone.astype(jnp.float32)

        # Compute CBs
        P_cb = self._pseudo_cb_batch(P_ncac)
        Q_cb = self._pseudo_cb_batch(Q_ncac)

        # Weighted Kabsch alignment on CBs
        finite = jnp.all(jnp.isfinite(P_cb), axis=-1)
        w = finite.astype(jnp.float32)
        sum_w = w.sum()
        P = jnp.nan_to_num(P_cb, nan=0.0, posinf=0.0, neginf=0.0)

        def _weighted_kabsch(_):
            denom = sum_w + 1e-8
            P_mean = (P * w[:, None]).sum(axis=0) / denom
            Q_mean = (Q_cb * w[:, None]).sum(axis=0) / denom
            P_cent = P - P_mean
            Q_cent = Q_cb - Q_mean
            H = P_cent.T @ (Q_cent * w[:, None])
            U, S, Vt = jnp.linalg.svd(H, full_matrices=False)
            R = Vt.T @ U.T
            det = jnp.linalg.det(R)
            R = jnp.where(det < 0, Vt.T @ (jnp.diag(jnp.array([1.0, 1.0, -1.0])) @ U.T), R)
            P_aln = (P_cent @ R) + Q_mean
            sq = jnp.sum((P_aln - Q_cb) ** 2, axis=-1)
            rmsd_val = jnp.sqrt((w * sq).sum() / denom + 1e-8)
            return rmsd_val

        rmsd = jax.lax.cond(sum_w >= 1.0, _weighted_kabsch, lambda _: jnp.asarray(0.0, dtype=jnp.float32), operand=None)
        return rmsd, {"motif_cb_rmsd": rmsd, "motif_cb_finite": sum_w}

class MotifDistogramCCE(LossTerm):
    def __init__(self, *, motif_positions: tuple[int, ...], motif_template_ca: np.ndarray, max_pair_distance: float = 20.0):
        object.__setattr__(self, "motif_positions", tuple(int(p) for p in motif_positions))
        object.__setattr__(self, "motif_template_ca", jnp.asarray(motif_template_ca, dtype=jnp.float32))
        object.__setattr__(self, "max_pair_distance", float(max_pair_distance))

    def __call__(self, sequence, output, key):
        binder_len = sequence.shape[0]
        logits = output.distogram_logits[:binder_len, :binder_len]  # [N,N,B]
        logits = jnp.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        bins = output.distogram_bins  # [B]
        # Map template CA distances to nearest bin indices
        idx = jnp.asarray(self.motif_positions, dtype=jnp.int32)
        Q = self.motif_template_ca  # [K,3]
        # build pair list over K x K excluding i==j
        K = Q.shape[0]
        # distances and mask within threshold
        dmat = jnp.sqrt(jnp.sum((Q[:, None, :] - Q[None, :, :]) ** 2, axis=-1))
        mask_pairs = (dmat <= self.max_pair_distance) & (~jnp.eye(K, dtype=bool))  # [K,K] bool
        # Sub-select motif logits: [K,K,B]
        logits_sub = logits[idx[:, None], idx[None, :], :]
        logp = jax.nn.log_softmax(logits_sub, axis=-1)  # [K,K,B]
        # Compute target bin index per pair: [K,K]
        # nearest bin by absolute difference
        # Expand dims to broadcast: dmat[...,None] vs bins[None,None,:]
        t_idx = jnp.argmin(jnp.abs(dmat[..., None] - bins[None, None, :]), axis=-1)  # [K,K]
        # One-hot over bins
        oh = jax.nn.one_hot(t_idx, logp.shape[-1], dtype=logp.dtype)  # [K,K,B]
        nll = -(oh * logp).sum(axis=-1)  # [K,K]
        # Reduce over masked pairs
        mask_f = mask_pairs.astype(nll.dtype)
        denom = mask_f.sum()
        loss = jnp.where(denom > 0, (nll * mask_f).sum() / denom, 0.0)
        return loss, {"motif_cce": loss}


class HallucinationEntropyDist(LossTerm):
    def __init__(self, *, excluded_positions: tuple[int, ...] = (), beta: float = 10.0, max_contact_bin: float = 5.0):
        object.__setattr__(self, "excluded_positions", tuple(int(p) for p in excluded_positions))
        object.__setattr__(self, "beta", float(beta))
        object.__setattr__(self, "max_contact_bin", float(max_contact_bin))

    def __call__(self, sequence, output, key):
        binder_len = sequence.shape[0]
        logits = output.distogram_logits[:binder_len, :binder_len]  # [N,N,B]
        logits = jnp.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        bins = output.distogram_bins  # [B]
        # build mask excluding motif positions and diagonal
        N = binder_len
        all_idx = jnp.arange(N)
        excl = jnp.zeros((N,), dtype=bool).at[jnp.asarray(self.excluded_positions, dtype=jnp.int32)].set(True)
        pair_mask = (~excl[:, None]) & (~excl[None, :]) & (~jnp.eye(N, dtype=bool))
        # select bins up to max_contact_bin and exclude the last/no-contact bin by construction
        bin_mask = bins <= self.max_contact_bin
        # renormalize with beta scaling
        logp = jax.nn.log_softmax(logits, axis=-1)
        logp_sel = jnp.where(bin_mask[None, None, :], logp, -1e9)
        logp_scaled = self.beta * logp_sel
        p_hat = jax.nn.softmax(logp_scaled, axis=-1)
        # entropy = -sum p log p over selected bins
        entropy = -(p_hat * jnp.where(bin_mask[None, None, :], jnp.log(jnp.clip(p_hat, 1e-12)), 0.0)).sum(-1)
        # average over allowed pairs
        loss = jnp.where(pair_mask.sum() > 0, (entropy * pair_mask).sum() / pair_mask.sum(), 0.0)
        return loss, {"halluc_entropy": loss}


class HallucinationEntropyLeaky(LossTerm):
    def __init__(self, *, excluded_positions: tuple[int, ...] = (), beta: float = 10.0, max_contact_bin: float | None = 5.0):
        object.__setattr__(self, "excluded_positions", tuple(int(p) for p in excluded_positions))
        object.__setattr__(self, "beta", float(beta))
        object.__setattr__(self, "max_contact_bin", None if max_contact_bin is None else float(max_contact_bin))

    def __call__(self, sequence, output, key):
        binder_len = sequence.shape[0]
        logits = output.distogram_logits[:binder_len, :binder_len]  # [N,N,B]
        bins = output.distogram_bins  # [B]

        N = binder_len
        excl = jnp.zeros((N,), dtype=bool).at[jnp.asarray(self.excluded_positions, dtype=jnp.int32)].set(True)
        pair_mask = (~excl[:, None]) & (~excl[None, :]) & (~jnp.eye(N, dtype=bool))

        # Masks over bins: exclude last bin; optionally restrict to <= max_contact_bin for the sum
        B = logits.shape[-1]
        last_idx = B - 1
        allow_bins = jnp.ones((B,), dtype=bool).at[last_idx].set(False)
        if self.max_contact_bin is not None:
            allow_bins = allow_bins & (bins <= self.max_contact_bin)

        # p over all bins; q_all is softmax over all bins with beta scaling as in supplement
        logp = jax.nn.log_softmax(logits, axis=-1)
        q_all = jax.nn.softmax(self.beta * logp, axis=-1)  # [N,N,B]

        # p_hat excludes the last bin, renormalized over allowed bins
        mask_allow = allow_bins[None, None, :]
        logp_masked = jnp.where(mask_allow, logp, -1e9)
        p_hat = jax.nn.softmax(self.beta * logp_masked, axis=-1)

        # Compute leaky entropy: -sum_{k in allowed, k != last} p_hat_k * log q_all_k
        contrib = -(p_hat * jnp.log(jnp.clip(q_all, 1e-12))).sum(axis=-1)

        # Exclude bins not allowed from summation by subtracting their contributions (since p_hat there ~0 it is fine)
        loss_mat = jnp.where(pair_mask, contrib, 0.0)
        denom = pair_mask.sum()
        loss = jnp.where(denom > 0, loss_mat.sum() / denom, 0.0)
        return loss, {"halluc_entropy_leaky": loss}


class SurfaceNonPolarLoss(LossTerm):
    def __init__(self, *, n0: float = 2.5, m: float = 1.0, a: float = 0.5, b: float = 2.0):
        object.__setattr__(self, "n0", float(n0))
        object.__setattr__(self, "m", float(m))
        object.__setattr__(self, "a", float(a))
        object.__setattr__(self, "b", float(b))

    @staticmethod
    def _pseudo_cb_from_backbone(bb_ncac: jnp.ndarray) -> jnp.ndarray:
        # bb_ncac: [L,3,3] (N,CA,C)
        return MotifPseudoCBRMSD._pseudo_cb_batch(bb_ncac)

    def __call__(self, sequence, output, key):
        # sequence: [L,20] probabilities (after masking); output carries backbone coords
        probs = jnp.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        L = probs.shape[0]
        bb = getattr(output, "backbone_coordinates")  # [N,4,3] in N,CA,C,O order
        nca = jnp.nan_to_num(bb[:L, [0, 1, 2], :], nan=0.0, posinf=0.0, neginf=0.0)  # [L,3,3]
        cb = self._pseudo_cb_from_backbone(nca)  # [L,3]
        ca = nca[:, 1, :]  # [L,3]

        # Direction vectors CA->CB per residue
        v = cb - ca
        v = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

        # Pairwise vectors and distances
        diff = cb[:, None, :] - cb[None, :, :]  # [L,L,3]
        d = jnp.linalg.norm(diff, axis=-1)  # [L,L]
        # Angle between CA->CB (at i) and vector to neighbor j (CB_j - CB_i)
        vi = v[:, None, :]  # [L,1,3]
        vij = diff / (jnp.linalg.norm(diff, axis=-1, keepdims=True) + 1e-8)  # [L,L,3]
        cos_phi = jnp.clip(jnp.sum(vi * vij, axis=-1), -1.0, 1.0)  # [L,L]
        # Use cos(pi - phi) = -cos(phi)
        cone = (-cos_phi + self.a) / (1.0 + self.a)
        cone = jnp.power(jnp.maximum(cone, 0.0), self.b)

        # Distance weighting 1/(1 + exp(d - m)) on CB-CB distance
        w_d = 1.0 / (1.0 + jnp.exp(d - self.m))

        # Exclude self-pairs
        mask = ~jnp.eye(L, dtype=bool)
        neigh = jnp.where(mask, w_d * cone, 0.0).sum(axis=-1)  # [L]

        # Surface indicator ~ 1 - sigmoid(n_i - n0)
        surf = 1.0 - jax.nn.sigmoid(neigh - self.n0)

        # Expected non-polar probability at each residue from soft sequence
        vocab = "ARNDCQEGHILKMFPSTWYV"
        nonpolar = ("V", "I", "L", "M", "W", "F")
        idxs = jnp.array([vocab.index(x) for x in nonpolar], dtype=jnp.int32)
        p_nonpolar = probs[:, idxs].sum(axis=-1)  # [L]

        # Loss = average p_nonpolar over surface residues
        denom = surf.sum()
        num = (p_nonpolar * surf).sum()
        loss = jnp.where(denom > 0, num / denom, jnp.asarray(0.0, dtype=jnp.float32))
        return loss, {"surface_nonpolar": loss, "n_surface_mean": surf.mean()}


class NetChargeLoss(LossTerm):
    def __init__(self, *, target_max_charge: float = -5.0, normalize_by_length: bool = True):
        object.__setattr__(self, "target_max_charge", float(target_max_charge))
        object.__setattr__(self, "normalize_by_length", bool(normalize_by_length))

    def __call__(self, sequence, output, key):
        probs = sequence  # [L,20]
        L = probs.shape[0]
        vocab = "ARNDCQEGHILKMFPSTWYV"
        def idx(a):
            return jnp.asarray(vocab.index(a), dtype=jnp.int32)
        pos = probs[:, idx("K")] + probs[:, idx("R")]
        neg = probs[:, idx("D")] + probs[:, idx("E")]
        net = (pos - neg).sum()
        if self.normalize_by_length:
            net = net / (jnp.maximum(jnp.asarray(L, dtype=jnp.float32), 1.0))
        # Hinge at target_max_charge (e.g., -5 or per-length normalized value)
        penalty = jnp.maximum(net - self.target_max_charge, 0.0)
        return penalty, {"net_charge": net, "net_charge_penalty": penalty}


## Removed BinderLigandContact to simplify the objective

def _greedy_autoplace_motif(logp_all: jnp.ndarray, bins: jnp.ndarray, motif_template_ca: jnp.ndarray, max_pair_distance: float = 20.0) -> jnp.ndarray:
    """Greedy auto-placement of K motif residues onto N positions using distogram CCE.

    Args:
        logp_all: [N,N,B] log-softmax over distance bins
        bins: [B] bin centers (Å)
        motif_template_ca: [K,3] template CA coords
    Returns:
        idx: [K] int32 positions
    """
    N = logp_all.shape[0]
    K = motif_template_ca.shape[0]
    # pairwise target distances and mask
    Q = motif_template_ca
    dmat = jnp.sqrt(jnp.sum((Q[:, None, :] - Q[None, :, :]) ** 2, axis=-1))  # [K,K]
    mask_pairs = (dmat <= max_pair_distance) & (~jnp.eye(K, dtype=bool))

    # helper to build NLL matrix for a target distance d
    def nll_for_d(d: jnp.ndarray) -> jnp.ndarray:
        bidx = jnp.argmin(jnp.abs(bins - d))
        return -logp_all[:, :, bidx]

    # initialize with best (i,j) for first two motif indices 0,1
    nll01 = nll_for_d(dmat[0, 1])
    # mask diagonal
    nll01 = nll01 + 1e6 * jnp.eye(N)
    i0j1 = jnp.unravel_index(jnp.argmin(nll01), nll01.shape)
    placed = [i0j1[0], i0j1[1]]

    def place_next(k, placed_list):
        used = jnp.array(placed_list, dtype=jnp.int32)
        # candidate mask over positions not in used
        mask_free = jnp.ones((N,), dtype=bool).at[used].set(False)
        # accumulate per-candidate costs against all placed indices
        cost = jnp.zeros((N,), dtype=logp_all.dtype) + 1e9
        def body(pos, acc):
            # skip used positions later by masking
            per = 0.0
            for uu in range(used.shape[0]):
                d = dmat[k, uu]
                C = nll_for_d(d)  # [N,N]
                per = per + C[pos, used[uu]] + C[used[uu], pos]
            return acc.at[pos].set(per)
        cost = jax.lax.fori_loop(0, N, body, cost)
        cost = jnp.where(mask_free, cost, 1e9)
        p = jnp.argmin(cost)
        return int(p)

    for k in range(2, K):
        pk = place_next(k, placed)
        placed.append(pk)

    return jnp.asarray(placed, dtype=jnp.int32)

def _build_mhetase_yaml(*, binder_len: int, enzyme_chain: str = "A", ligand_chain: str = "L", ligand_ccd: str | None = None, ligand_smiles: str | None = None, bond_constraints: list[dict] | None = None, binder_sequence: str | None = None) -> str:
    if not (ligand_ccd or ligand_smiles):
        raise ValueError("Provide ligand_ccd or ligand_smiles")
    # Build sequences block
    lines = ["version: 1", "sequences:"]
    seq = binder_sequence if (binder_sequence is not None and len(binder_sequence) == binder_len) else ("X" * binder_len)
    lines.append(f"  - protein:\n      id: {enzyme_chain}\n      sequence: {seq}\n      msa: empty")
    if ligand_ccd:
        lines.append(f"  - ligand:\n      id: {ligand_chain}\n      ccd: {ligand_ccd}")
    else:
        lines.append(f"  - ligand:\n      id: {ligand_chain}\n      smiles: '{ligand_smiles}'")
    # Constraints block (optional)
    if bond_constraints:
        lines.append("constraints:")
        for bc in bond_constraints:
            a1 = bc["atom1"]; a2 = bc["atom2"]
            lines.append("  - bond:")
            lines.append(f"      atom1: [{a1[0]}, {a1[1]}, {a1[2]}]")
            lines.append(f"      atom2: [{a2[0]}, {a2[1]}, {a2[2]}]")
    return "\n".join(lines)


def build_boltz2_predict_fn_mhetase(*, binder_len: int, enzyme_chain: str, ligand_chain: str, ligand_ccd: str | None = None, ligand_smiles: str | None = None, num_sampling_steps: int = 200, recycling_steps: int = 3):
    # Lazy import here to avoid heavy deps when only testing losses locally
    try:
        from mosaic.losses.boltz2 import (
            load_boltz2,
            load_features_and_structure_writer as load_boltz2_features,
            set_binder_sequence as set_boltz2_binder_sequence,
            Boltz2Output,
        )
    except Exception as e:
        raise RuntimeError("Boltz2 dependencies are not available. Use a mock predict_fn for local tests.") from e
    joltz2 = load_boltz2()

    es_yaml = _build_mhetase_yaml(
        binder_len=binder_len,
        enzyme_chain=enzyme_chain,
        ligand_chain=ligand_chain,
        ligand_ccd=ligand_ccd,
        ligand_smiles=ligand_smiles,
        bond_constraints=None,
    )
    es_features, _ = load_boltz2_features(es_yaml, cache=Path(os.environ.get("BOLTZ_CACHE", "/root/.boltz")).expanduser())
    esstar_cache = {}

    def predict_fn(probs, *, key, state: dict | None = None):
        state = state or {}
        forced = tuple(state.get("forced_bonds", ()))
        if forced:
            bc = [
                {"atom1": forced_pair[0], "atom2": forced_pair[1]}
                for forced_pair in forced
            ]
            if forced not in esstar_cache:
                esstar_yaml = _build_mhetase_yaml(
                    binder_len=binder_len,
                    enzyme_chain=enzyme_chain,
                    ligand_chain=ligand_chain,
                    ligand_ccd=ligand_ccd,
                    ligand_smiles=ligand_smiles,
                    bond_constraints=bc,
                )
                esstar_features, _ = load_boltz2_features(esstar_yaml, cache=Path(os.environ.get("BOLTZ_CACHE", "/root/.boltz")).expanduser())
                esstar_cache[forced] = esstar_features
            features = esstar_cache[forced]
        else:
            features = es_features

        # Use soft probabilities directly; straight-through is handled at the optimizer/transform level
        features_local = copy.deepcopy(features)
        feats = set_boltz2_binder_sequence(probs, features_local)
        out = Boltz2Output(
            joltz2=joltz2,
            features=feats,
            deterministic=False,
            key=key if key is not None else jax.random.key(0),
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
        )
        return out

    return predict_fn


def build_tmol_energy_fn(tmol_context, predict_fn):
    """Return a predictor-only energy function.

    This workflow uses structure-prediction-driven losses (motif CCE + RMSD, hallucination entropy, contacts, pLDDT).
    TMOL terms are omitted to keep the objective focused and stable.

    Steps:
    1) predict structure from sequence probabilities via predict_fn (Boltz2 by default)
    2) returns a minimal aux tying through pLDDT if available
    """
    def energy_fn(probs, key, state):
        output = predict_fn(probs, key=key, state=state)
        aux = {}
        try:
            plddt = getattr(output, "plddt", None)
            if plddt is not None:
                aux = {"structure_confidence": {"plddt_per_residue": plddt}}
        except Exception:
            pass
        # Zero baseline; real loss is composed at build_loss level
        return jnp.asarray(0.0, dtype=jnp.float32), aux

    return energy_fn


def make_workflow(*, binder_len: int, motif_positions: dict, tmol_context: dict, predict_fn=None, es_star_forced_bonds: tuple[tuple[int, int], ...] | None = None, motif_template_ca: np.ndarray | None = None, motif_template_backbone: np.ndarray | None = None, avoid_residues: tuple[str, ...] = ("C",), ligand_metric: str = "iptm", pae_on: bool = True, helix_weight: float = -0.3, mpnn_weight: float = 0.0, use_leaky_entropy_soft: bool = False, use_leaky_entropy_anneal: bool = True, surface_nonpolar_weight_soft: float = 0.0, surface_nonpolar_weight_anneal: float = 0.5, net_charge_weight_soft: float = 0.0, net_charge_weight_anneal: float = 0.05, net_charge_target: float = -5.0, snp_n0: float = 2.5, entropy_beta: float = 10.0, entropy_max_contact_bin: float = 5.0, anneal_min_temp: float = 0.01, include_struct_priors_soft: bool = True, include_struct_priors_anneal: bool = True):
    ser_val = motif_positions.get("ser")
    his_val = motif_positions.get("his")
    asp_val = motif_positions.get("asp")
    ser = int(ser_val) if ser_val is not None else 0
    his = int(his_val) if his_val is not None else 0
    asp = int(asp_val) if asp_val is not None else 0
    oxy_list = motif_positions.get("oxyanion", []) or []
    oxy = tuple(int(x) for x in oxy_list)

    motif_loss = lambda: CatalyticMotifLoss(
        ser_pos=ser, his_pos=his, asp_pos=asp, oxyanion_positions=oxy if oxy else None
    )
    motif_positions_tuple = None
    positions_tuple = (ser, his, asp) + (tuple(oxy) if oxy else tuple())
    if motif_template_backbone is not None and hasattr(motif_template_backbone, "shape") and int(motif_template_backbone.shape[0]) == len(positions_tuple):
        motif_positions_tuple = positions_tuple
        motif_struct = lambda: MotifStructureRMSD(
            motif_positions=positions_tuple,
            target_backbone=motif_template_backbone.astype(np.float32),
        )
    elif motif_template_ca is not None and len(motif_template_ca) == len(positions_tuple):
        motif_positions_tuple = positions_tuple
        motif_struct = lambda: MotifStructureRMSD(
            motif_positions=positions_tuple,
            target_ca=motif_template_ca.astype(np.float32),
        )
    else:
        motif_struct = None

    if predict_fn is None:
        def _default_predict(probs, *, key, state):
            class _Out:
                def __init__(self, coords):
                    self.structure_coordinates = tmol_context["coords"]
                    self.plddt_per_residue = None
            return _Out(coords=None)
        predict = _default_predict
    else:
        predict = predict_fn

    energy_fn = build_tmol_energy_fn(tmol_context, predict)

    # Build per-position allowed token mask following ColabDesign-style hard constraints
    vocab = "ARNDCQEGHILKMFPSTWYV"
    def _allowed_matrix() -> np.ndarray:
        allowed = np.ones((binder_len, 20), dtype=np.float32)
        # Avoid globally
        for r in avoid_residues:
            if r in vocab:
                allowed[:, vocab.index(r)] = 0.0
        # Motif fixed identities/sets
        # ser/his/asp and optional oxyanion positions
        if ser >= 0:
            allowed[ser, :] = 0.0; allowed[ser, vocab.index("S")] = 1.0
        if his >= 0:
            allowed[his, :] = 0.0; allowed[his, vocab.index("H")] = 1.0
        if asp >= 0:
            # Allow D (and E if asp in allowed set)
            allowed[asp, :] = 0.0
            allowed[asp, vocab.index("D")] = 1.0
        for ox in oxy:
            allowed[ox, :] = 0.0
            for aa in ("G", "S", "A"):
                allowed[ox, vocab.index(aa)] = 1.0
        return allowed

    allowed_tokens = _allowed_matrix()
    fixed_mask = (allowed_tokens.sum(-1) == 1.0).astype(np.float32)  # positions fully fixed to one AA

    # Precompute greedy auto-placed motif indices once if not provided
    autoplace_idx: tuple[int, ...] | None = None
    if motif_template_ca is not None and (ser_val is None or his_val is None or asp_val is None):
        # uniform over allowed tokens
        probs0 = jnp.asarray(allowed_tokens, dtype=jnp.float32)
        probs0 = probs0 / (jnp.sum(probs0, axis=-1, keepdims=True) + 1e-8)
        out0 = predict(probs0, key=jax.random.key(0), state={})
        logp_b = jax.nn.log_softmax(out0.distogram_logits[:binder_len, :binder_len], axis=-1)
        idx_auto = _greedy_autoplace_motif(logp_b, jnp.asarray(out0.distogram_bins, dtype=jnp.float32), jnp.asarray(motif_template_ca, dtype=jnp.float32), max_pair_distance=20.0)
        autoplace_idx = tuple(int(x) for x in list(np.array(idx_auto)))

    def _build_motif_geo_loss():
        geo = None
        if motif_template_ca is not None:
            pos = motif_positions_tuple if motif_positions_tuple is not None else (autoplace_idx if autoplace_idx is not None else None)
            if pos is not None:
                geo = 2.0 * MotifDistogramCCE(
                    motif_positions=pos,
                    motif_template_ca=motif_template_ca.astype(np.float32),
                    max_pair_distance=20.0,
                )
        if motif_struct is not None:
            geo_term = 1.0 * motif_struct()
            geo = geo + geo_term if geo is not None else geo_term
        if geo is None:
            # fallback to zero-loss
            class _Zero(LossTerm):
                def __call__(self, *a, **k):
                    return jnp.asarray(0.0, dtype=jnp.float32), {}
            geo = _Zero()
        # Stabilize motif geometry with normalized gradients only
        geo = NormedGradient(loss=geo)
        return geo

    # Single-stage design loss configuration (clean, mosaic-style)
    def build_loss_single(*, state, use_leaky_entropy=False, surface_nonpolar_weight=0.0, net_charge_weight=0.0, net_charge_target=-5.0, snp_n0=2.5, entropy_beta_local: float | None = None, entropy_max_contact_bin_local: float | None = None, include_struct_priors: bool = True):
        def _fn(probs, key=None):
            # Enforce per-position allowed identities directly on probs before any prediction/loss
            mask = jnp.asarray(allowed_tokens, dtype=jnp.float32)
            probs_masked = probs * mask
            probs_masked = probs_masked / (jnp.sum(probs_masked, axis=-1, keepdims=True) + 1e-8)
            probs_masked = jnp.nan_to_num(probs_masked, nan=1.0/20.0, posinf=0.0, neginf=0.0)

            output = predict(probs_masked, key=key, state=state)
            # Complete objective following paper: w_M * ℒ_M + w_H * ℒ_H + ℒ_aux (w_M = w_H = 1)
            # ℒ_M = motif geometric loss, ℒ_H = hallucination entropy loss, ℒ_aux = structure losses

            # Hallucination loss ℒ_H (entropy of renormalized predictions)
            beta_val = float(entropy_beta if entropy_beta_local is None else entropy_beta_local)
            max_bin_val = float(entropy_max_contact_bin if entropy_max_contact_bin_local is None else entropy_max_contact_bin_local)
            if use_leaky_entropy:
                hallucination_loss = 1.0 * HallucinationEntropyLeaky(
                    excluded_positions=motif_positions_tuple if motif_positions_tuple else (),
                    beta=beta_val,
                    max_contact_bin=max_bin_val,
                )
            else:
                hallucination_loss = 1.0 * HallucinationEntropyDist(
                    excluded_positions=motif_positions_tuple if motif_positions_tuple else (),
                    beta=beta_val,
                    max_contact_bin=max_bin_val,
                )

            # Auxiliary structural priors (toggleable)
            aux_loss = None
            if bool(include_struct_priors):
                aux_loss = (
                    1.0 * sp.WithinBinderContact(
                        max_contact_distance=14.0,
                        num_contacts_per_residue=4,
                        min_sequence_separation=8,
                    )
                    + 0.1 * sp.PLDDTLoss()
                )
                if bool(pae_on):
                    aux_loss = aux_loss + 0.2 * sp.WithinBinderPAE()
                if float(helix_weight) != 0.0:
                    aux_loss = aux_loss + float(helix_weight) * sp.HelixLoss()

            # Surface nonpolar penalty and net charge (optional)
            if float(surface_nonpolar_weight) > 0.0:
                base = aux_loss if aux_loss is not None else (0 * hallucination_loss)
                aux_loss = base + float(surface_nonpolar_weight) * SurfaceNonPolarLoss(n0=float(snp_n0))
            if float(net_charge_weight) > 0.0:
                base = aux_loss if aux_loss is not None else (0 * hallucination_loss)
                aux_loss = base + float(net_charge_weight) * NetChargeLoss(target_max_charge=float(net_charge_target), normalize_by_length=True)

            # Total objective: start from hallucination; add optional aux; add motif terms
            motif_geo = _build_motif_geo_loss()
            struct = hallucination_loss + (aux_loss if aux_loss is not None else 0 * hallucination_loss) + motif_geo

            # Loss-internal gradient shaping
            struct = NormedGradient(loss=struct)
            struct = ClippedGradient(loss=struct, max_norm=1.0)

            v_struct, aux_struct = struct(probs_masked, output=output, key=key)

            # Optional ProteinMPNN prior
            if float(mpnn_weight) > 0.0:
                try:
                    from mosaic.losses.protein_mpnn import ProteinMPNNLoss as _ProteinMPNNLoss
                    mpnn_loss = _ProteinMPNNLoss(mpnn=_get_mpnn(), num_samples=4, stop_grad=True)
                    v_mpnn, aux_mpnn = mpnn_loss(probs_masked, output, key)
                    v_struct = v_struct + float(mpnn_weight) * v_mpnn
                    aux_struct = {**aux_struct, "mpnn": aux_mpnn}
                except Exception:
                    pass
            if not isinstance(aux_struct, dict):
                aux_struct = {"components": aux_struct}

            # Attach motif identity metrics (no identity loss added)
            try:
                _val_motif, aux_motif = motif_loss()(probs_masked, key=key)
                aux = {"struct": aux_struct, "motif": aux_motif}
            except Exception:
                aux = {"struct": aux_struct}
            return v_struct, aux
        return _fn

    # Two-phase schedules
    def sched_warmup(g, p):
        return {"lr": base_lr * inv_sqrt_L, "temperature": 1.0, "e_soft": 1.0}

    inv_sqrt_L = float(1.0 / max(1.0, np.sqrt(binder_len)))
    base_lr = 0.05  # paper-typical constant LR

    # Default internal values; external runners typically override `steps` per phase
    phase_warmup_steps = 40
    phase_anneal_steps = 240

    def sched_anneal(g, p):
        T = max(float(anneal_min_temp), 1.0 - (1.0 - float(anneal_min_temp)) * (p / max(1, phase_anneal_steps)) ** 2)
        # constant LR scaled by 1/sqrt(L); temperature anneals only
        return {"lr": base_lr * inv_sqrt_L, "temperature": T}

    def build_loss_warmup():
        def _fn(probs, key=None):
            mask = jnp.asarray(allowed_tokens, dtype=jnp.float32)
            probs_masked = probs * mask
            probs_masked = probs_masked / (jnp.sum(probs_masked, axis=-1, keepdims=True) + 1e-8)
            output = predict(probs_masked, key=key, state={})
            losses = (
                1.0 * HallucinationEntropyDist(
                    excluded_positions=positions_tuple if positions_tuple else (),
                    beta=10.0,
                    max_contact_bin=5.0,
                )
                + 0.1 * sp.PLDDTLoss()
            )
            # Loss-internal gradient shaping for stability in warmup
            # losses = ClippedGradient(loss=losses, max_norm=1.0)
            v, aux = losses(probs_masked, output=output, key=key)
            return v, {"struct": aux}
        return _fn

    # Three phases: motif_lock (geometry only), soft (full at T=1, e_soft=0.8), anneal (temperature decay)
    def build_loss_motif_lock():
        def _fn(probs, key=None):
            mask = jnp.asarray(allowed_tokens, dtype=jnp.float32)
            probs_masked = probs * mask
            probs_masked = probs_masked / (jnp.sum(probs_masked, axis=-1, keepdims=True) + 1e-8)
            output = predict(probs_masked, key=key, state={})
            geo = _build_motif_geo_loss()
            geo = ClippedGradient(loss=geo, max_norm=1.0)
            v, aux = geo(probs_masked, output=output, key=key)
            return v, {"struct": aux}
        return _fn

    def sched_soft(g, p):
        return {"lr": base_lr * inv_sqrt_L, "temperature": 1.0, "e_soft": 0.8}

    phases = [
        {
            "name": "motif_lock",
            "build_loss": build_loss_motif_lock,
            "optimizer": sgd_logits,
            "steps": max(1, int(0.2 * (phase_warmup_steps + phase_anneal_steps))),  # default; runner may override
            "schedule": sched_warmup,
            "transforms": {
                "pre_logits": [temperature_on_logits()],
                "post_logits": [per_position_allowed_tokens(allowed_tokens)],
                "grad": [position_mask(1.0 - fixed_mask), gradient_normalizer(mode="l2_effL")],
            },
            "analyzers": [],
            "analyze_every": 1,
        },
        {
            "name": "soft",
            "build_loss": lambda: build_loss_single(
                state={},
                use_leaky_entropy=bool(use_leaky_entropy_soft),
                surface_nonpolar_weight=float(surface_nonpolar_weight_soft),
                net_charge_weight=float(net_charge_weight_soft),
                net_charge_target=float(net_charge_target),
                snp_n0=float(snp_n0),
                entropy_beta_local=float(entropy_beta),
                entropy_max_contact_bin_local=float(entropy_max_contact_bin),
                include_struct_priors=bool(include_struct_priors_soft),
            ),
            "optimizer": sgd_logits,
            "steps": phase_warmup_steps,  # default; runner may override
            "schedule": sched_soft,
            "transforms": {
                "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
                "post_logits": [per_position_allowed_tokens(allowed_tokens)],
                "grad": [position_mask(1.0 - fixed_mask), gradient_normalizer(mode="l2_effL")],
            },
            "analyzers": [],
            "analyze_every": 1,
        },
        {
            "name": "anneal",
            "build_loss": lambda: build_loss_single(
                state={},
                use_leaky_entropy=bool(use_leaky_entropy_anneal),
                surface_nonpolar_weight=float(surface_nonpolar_weight_anneal),
                net_charge_weight=float(net_charge_weight_anneal),
                net_charge_target=float(net_charge_target),
                snp_n0=float(snp_n0),
                entropy_beta_local=float(entropy_beta),
                entropy_max_contact_bin_local=float(entropy_max_contact_bin),
                include_struct_priors=bool(include_struct_priors_anneal),
            ),
            "optimizer": sgd_logits,
            "steps": phase_anneal_steps,
            "schedule": sched_anneal,
            "transforms": {
                "pre_logits": [temperature_on_logits()],
                "post_logits": [per_position_allowed_tokens(allowed_tokens)],
                "grad": [position_mask(1.0 - fixed_mask), gradient_normalizer(mode="l2_effL")],
            },
            "analyzers": [],
            "analyze_every": 1,
        },
    ]

    return {"phases": phases, "binder_len": int(binder_len), "seed": 0}


def run(binder_len: int, motif_positions: dict, tmol_context: dict, initial_x: np.ndarray | None = None):
    # Build a default Boltz2 predictor without forced bonds (ligand must be provided via tmol_context extras)
    ligand = tmol_context.get("ligand", {})
    predict_fn = build_boltz2_predict_fn_mhetase(
        binder_len=binder_len,
        enzyme_chain=ligand.get("enzyme_chain", "A"),
        ligand_chain=ligand.get("ligand_chain", "L"),
        ligand_ccd=ligand.get("ccd"),
        ligand_smiles=ligand.get("smiles"),
    )
    wf = make_workflow(binder_len=binder_len, motif_positions=motif_positions, tmol_context=tmol_context, predict_fn=predict_fn)
    if initial_x is not None:
        wf["initial_x"] = initial_x
    else:
        # Initialize logits with motif residues fixed (ColabDesign-style)
        vocab = "ARNDCQEGHILKMFPSTWYV"
        x0 = np.random.randn(binder_len, 20).astype(np.float32) * 0.1
        ser = int(motif_positions.get("ser", 0)); his = int(motif_positions.get("his", 0)); asp = int(motif_positions.get("asp", 0))
        def set_pos(pos, aa):
            if 0 <= pos < binder_len:
                x0[pos, :] = -10.0
                x0[pos, vocab.index(aa)] = 10.0
        set_pos(ser, "S"); set_pos(his, "H"); set_pos(asp, "D")
        for ox in motif_positions.get("oxyanion", []) or []:
            if 0 <= int(ox) < binder_len:
                x0[int(ox), :] = -10.0
                for aa in ["G", "S", "A"]:
                    x0[int(ox), vocab.index(aa)] = 10.0
        wf["initial_x"] = x0
    return run_workflow(wf)


