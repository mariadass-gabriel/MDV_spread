"""Per-step training diagnostics for CBOpt optimizers."""

from __future__ import annotations

from typing import Dict


def step_metrics(optimizer) -> Dict[str, float]:
    """Return diagnostic scalars from optimizer diag_state, or empty dict."""
    ds = getattr(optimizer, "diag_state", None)
    if not ds:
        return {}
    return {
        "iso": _iso_metrics,
        "diag": _diag_metrics,
        "lrd": _lrd_metrics,
        "block": _block_metrics}[ds["structure"]](ds)


# Log uncapped H_s > 0 positivity check and Picard contraction constant L_Phi
def _iso_metrics(ds: dict) -> Dict[str, float]:
    a_min = ds["eps_rel"] * ds["hd_min"]
    lphi_bound = 2.0 * ds["d"] * ds["grad_sq"] / max(ds["u_sq"] ** 2 * a_min ** 3, 1e-30)
    return {
        "hs_pos_raw": ds["hs_pos_raw"],
        "hs_margin_raw": ds["hs_margin_raw"],
        "hs_pos_raw_cf": ds["hs_pos_raw_cf"],
        "hs_margin_raw_cf": ds["hs_margin_raw_cf"],
        "lphi_real": ds["lphi"],
        "lphi_cf": ds["lphi_cf"],
        "lphi_bound_real": lphi_bound,
        "vartheta": ds["vartheta"],
        "vartheta_cf": ds["vartheta_cf"]}


# Log coordinate-wise uncapped H_s > 0 positivity check
def _diag_metrics(ds: dict) -> Dict[str, float]:
    return {
        "hs_pos_raw": ds["hs_pos_raw"],
        "hs_margin_raw": ds["hs_margin_raw"]}


# Track s=-1 feasibility, sign-flip rate, and uncapped positivity for LRD
def _lrd_metrics(ds: dict) -> Dict[str, float]:
    return {
        "feasible_sminus": float(ds["feasible"]),
        "infeasible_sminus": float(not ds["feasible"]),
        "flip_to_splus": float(ds["flipped"]),
        "n_sminus": float(ds["n_sminus"]),
        "n_sminus_infeasible": float(ds["n_sminus_infeasible"]),
        "n_splus": float(ds["n_splus"]),
        "psi": ds["psi"],
        "u_sq": ds["u_sq"],
        "hs_pos_raw_lrd": ds["hs_pos_raw_lrd"]}


# Aggregate per-block feasibility, sign flips, and uncapped positivity
def _block_metrics(ds: dict) -> Dict[str, float]:
    n = max(ds["n_blocks"], 1)
    return {
        "n_blocks": float(ds["n_blocks"]),
        "flip_to_splus_frac": ds["n_flip"] / n,
        "infeasible_frac": ds["n_infeasible"] / n,
        "n_sminus": float(ds["n_sminus"]),
        "n_sminus_infeasible": float(ds["n_sminus_infeasible"]),
        "n_splus": float(ds["n_splus"]),
        "hs_pos_raw_block": ds["n_hs_pos_raw"] / n,
        "psi_mean": ds["psi_mean"]
        }
