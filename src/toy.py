"""Closed-form isotropic quadratic toy problem: existence, penalty, Picard map, and SAM comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToyQuadratic:
    """Diagonal quadratic with an L2 prior of precision delta."""

    a: np.ndarray
    mu: np.ndarray
    delta: float

    @property
    def dim(self) -> int:
        return int(self.a.shape[0])

    @property
    def hd(self) -> np.ndarray:
        """Diagonal of nabla^2 lhat = a + delta (the boundary of H_s > 0 per coordinate)."""
        return self.a + self.delta

    def grad_l(self, eta: np.ndarray) -> np.ndarray:
        """nabla l(eta) = A (eta - mu)."""
        return self.a * (eta - self.mu)

    def grad_lhat(self, eta: np.ndarray) -> np.ndarray:
        """nabla lhat(eta) = A (eta - mu) + delta eta."""
        return self.a * (eta - self.mu) + self.delta * eta

    def l(self, theta: np.ndarray) -> float:
        return float(0.5 * np.sum(self.a * (theta - self.mu) ** 2))

    def lhat(self, theta: np.ndarray) -> float:
        return self.l(theta) + 0.5 * self.delta * float(np.sum(theta ** 2))


# Signed Newton step u_s^iso = -s (diag(a+delta) - vartheta I)^-1 nabla lhat(eta) (Prop iso_diag).
def u_iso(toy: ToyQuadratic, eta: np.ndarray, vartheta: float, s: int) -> np.ndarray:
    return -s * toy.grad_lhat(eta) / (toy.hd - vartheta)


def sign_condition(toy: ToyQuadratic, vartheta: float, s: int) -> bool:
    """s=+1 needs vartheta < min(a+delta), while s=-1 needs vartheta > max(a+delta)."""
    if s == 1:
        return bool(vartheta < np.min(toy.hd))
    return bool(vartheta > np.max(toy.hd))


# Raw signed objective CBO_s(eta, vartheta) = ext_s { -lhat(theta) + vartheta/2 ||theta-eta||^2 }, before the penalty.
def cbo_iso(toy: ToyQuadratic, eta: np.ndarray, vartheta: float, s: int) -> float:
    if not sign_condition(toy, vartheta, s):
        return float("inf") if s == 1 else float("-inf")
    u_c = -toy.grad_lhat(eta) / (toy.hd - vartheta)
    theta_c = eta + u_c
    return -toy.lhat(theta_c) + 0.5 * vartheta * float(np.sum(u_c ** 2))


# Penalized objective PCBO_s(eta, vartheta) = CBO_s - zeta log vartheta, zeta = d/2 (Prop one_penalty).
def pcbo_iso(toy: ToyQuadratic, eta: np.ndarray, vartheta: float, s: int,
             zeta: float | None = None) -> float:
    if zeta is None:
        zeta = toy.dim / 2.0
    return cbo_iso(toy, eta, vartheta, s) - zeta * float(np.log(vartheta))


# Prop breaks_monotone : d/dvartheta PCBO_s = 1/2 ||u_s(vartheta)||^2 - zeta / vartheta
def dpcbo_dvartheta(toy: ToyQuadratic, eta: np.ndarray, vartheta: float, s: int,
                    zeta: float | None = None) -> float:
    if zeta is None:
        zeta = toy.dim / 2.0
    u_c = toy.grad_lhat(eta) / (toy.hd - vartheta)
    return 0.5 * float(np.sum(u_c ** 2)) - zeta / vartheta


# Exact uncapped stationary point of PCBO_s in vartheta, by bisection (Prop breaks_monotone).
def stationary_vartheta_iso(toy: ToyQuadratic, eta: np.ndarray, s: int,
                            zeta: float | None = None, tol: float = 1e-12) -> float:
    m, M = float(np.min(toy.hd)), float(np.max(toy.hd))
    if s == 1:
        lo, hi = 1e-12 * m, m * (1.0 - 1e-9)
    else:
        lo, hi = M * (1.0 + 1e-9), M * 1e6
    f = lambda v: dpcbo_dvartheta(toy, eta, v, s, zeta)
    f_lo, f_hi = f(lo), f(hi)
    assert f_lo * f_hi < 0.0, "no sign change on the feasible interval"
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) * f_lo > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(hi, 1.0):
            break
    return 0.5 * (lo + hi)


# SAM (ssec:sam). Eq sam_pert : eps*_SAM = rho nabla l(eta) / ||nabla l(eta)||.
def sam_perturbation(toy: ToyQuadratic, eta: np.ndarray, rho: float) -> np.ndarray:
    g = toy.grad_l(eta)
    return rho * g / float(np.linalg.norm(g))


# Eq gauss_pert : u*_soft = vartheta^-1 nabla l(eta) (same direction as eps*_SAM, different norm).
def u_soft(toy: ToyQuadratic, eta: np.ndarray, vartheta: float) -> np.ndarray:
    return toy.grad_l(eta) / vartheta


# Eq adaptive_radius / Eq level_set at alpha = e^{-1/2} : rho_alpha = vartheta^{-1/2}.
def induced_radius(vartheta: float) -> float:
    return float(vartheta ** -0.5)
