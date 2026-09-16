"""CBOpt optimizers with learned precision (iso, diag, LRD, block) and SAM."""

from __future__ import annotations
import math
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

ClosureType = Callable[[], Tensor]


def _assign_flat(params, flat: Tensor) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        p.add_(flat[offset:offset + n].view_as(p))
        offset += n


class _CBOptBase(Optimizer):
    """Shared EMA bookkeeping for the CBOpt family (gradient / squared-gradient)."""

    def _ema_step(self, group) -> tuple:
        b1, b2 = group["beta1"], group["beta2"]
        h0 = group["hess_init"]
        params, grads, m_list, v_list, steps = [], [], [], [], []
        for p in group["params"]:
            if p is None or p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["m"] = torch.zeros_like(p)
                state["v"] = torch.full_like(p, float(h0))
                state["vartheta"] = torch.full_like(p, float("nan"))
            state["step"] += 1
            params.append(p)
            grads.append(p.grad)
            m_list.append(state["m"])
            v_list.append(state["v"])
            steps.append(state["step"])
        if not params:
            return None
        sq = torch._foreach_mul(grads, grads)
        torch._foreach_mul_(m_list, b1)
        torch._foreach_add_(m_list, grads, alpha=1.0 - b1)
        torch._foreach_mul_(v_list, b2)
        torch._foreach_add_(v_list, sq, alpha=1.0 - b2)
        t = steps[0]
        m_bar = torch._foreach_div(m_list, [1.0 - (b1 ** s) for s in steps])
        return params, m_bar, v_list, t


# Isotropic precision update via damped Picard iterations with safety cap.
class CBOptIso(_CBOptBase):
    """Isotropic CBOpt: scalar precision vartheta learned by a damped Picard fixed point."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        s: int = 1,
        weight_decay: float = 2e-3,
        hess_init: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        eps_rel: float = 0.1,
        r: float = 1e-2,
        omega: float = 0.5,
        picard_steps: int = 5,
        eps: float = 1e-12):
        assert s in (1, -1), "s must be +1 or -1"
        assert 0.0 < eps_rel < 1.0 and 0.0 < r < 1.0 and 0.0 < omega <= 1.0
        defaults = dict(lr=lr, s=s, weight_decay=weight_decay, hess_init=hess_init,
                        beta1=beta1, beta2=beta2, eps_rel=eps_rel, r=r, omega=omega,
                        picard_steps=picard_steps, eps=eps)
        super().__init__(params, defaults)
        self.diag_state = {}

    @torch.no_grad()
    def step(self, closure: Optional[ClosureType] = None) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            packed = self._ema_step(group)
            if packed is None:
                continue
            params, m_bar, v_list, t = packed
            s = group["s"]
            wd = group["weight_decay"]
            eps_rel, r, omega = group["eps_rel"], group["r"], group["omega"]
            delta = wd

            numer = torch._foreach_add(m_bar, params, alpha=wd)
            hd_list = torch._foreach_add(v_list, delta)
            hd_min = min(float(x.min()) for x in hd_list)
            hd_max = max(float(x.max()) for x in hd_list)
            d = float(sum(p.numel() for p in params))

            vth0 = group.get("vartheta")
            if vth0 is None or not math.isfinite(vth0):
                bound = hd_min if s == 1 else hd_max
                group["vartheta"] = bound * (1.0 - s * eps_rel)

            k, eps = group["picard_steps"], group["eps"]
            vartheta = self._picard(group["vartheta"], s, numer, hd_list, hd_min, hd_max,
                                    d, eps_rel, r, omega, k, eps)
            group["vartheta"] = vartheta

            denom = torch._foreach_sub(hd_list, vartheta)
            u_list = torch._foreach_div(numer, denom)
            torch._foreach_mul_(u_list, -float(s))
            torch._foreach_add_(params, u_list, alpha=group["lr"])
            u_sq = float(sum(float(x.pow(2).sum()) for x in u_list))

            # counterfactual precision for the opposite sign (runs_plan, principe (i))
            s_cf = -s
            bound_cf = hd_min if s_cf == 1 else hd_max
            vth_cf = self._picard(bound_cf * (1.0 - s_cf * eps_rel), s_cf, numer, hd_list,
                                  hd_min, hd_max, d, eps_rel, r, omega, k, eps)
            u_cf_sq = self._u_sq(numer, hd_list, vth_cf)

            grad_sq = float(sum(float(x.pow(2).sum()) for x in numer))
            diag = dict(structure="iso", s=s, step=t, delta=delta, d=d, eps_rel=eps_rel,
                        hd_min=hd_min, hd_max=hd_max, vartheta=vartheta, vartheta_cf=vth_cf,
                        u_sq=u_sq, u_sq_cf=u_cf_sq, grad_sq=grad_sq)
            diag.update(self._positivity_and_lphi(numer, hd_list, d, s, u_sq, "", hd_min, hd_max))
            diag.update(self._positivity_and_lphi(numer, hd_list, d, s_cf, u_cf_sq, "_cf",
                                                  hd_min, hd_max))
            self.diag_state = diag
        return loss

    @staticmethod
    def _u_sq(numer, hd_list, vartheta) -> float:
        u_list = torch._foreach_div(numer, torch._foreach_sub(hd_list, vartheta))
        return float(sum(float(x.pow(2).sum()) for x in u_list))

    @staticmethod
    def _positivity_and_lphi(numer, hd_list, d, sign, u_sq, suffix, hd_min, hd_max) -> dict:
        # Uncapped positivity check H_s > 0 and exact Picard contraction constant L_Phi.
        vstar_raw = d / max(u_sq, 1e-30)
        margin = (hd_min - vstar_raw) if sign == 1 else (vstar_raw - hd_max)
        a_list = torch._foreach_sub(hd_list, vstar_raw)
        g2a3 = float(sum(float((n.pow(2) / a.pow(3)).sum()) for n, a in zip(numer, a_list)))
        u_raw_sq = float(sum(float((n.pow(2) / a.pow(2)).sum()) for n, a in zip(numer, a_list)))
        lphi = abs(-(2.0 * d / max(u_raw_sq, 1e-30) ** 2) * g2a3)
        return {"hs_margin_raw" + suffix: margin, "hs_pos_raw" + suffix: float(margin > 0.0),
                "lphi" + suffix: lphi, "vartheta_raw" + suffix: vstar_raw}

    @staticmethod
    def _picard(vartheta, s, numer, hd_list, hd_min, hd_max, d, eps_rel, r, omega, k, eps):
        for _ in range(k):
            denom = torch._foreach_sub(hd_list, vartheta)
            u_list = torch._foreach_div(numer, denom)
            u_sq = float(sum(float(x.pow(2).sum()) for x in u_list))
            vstar = d / max(u_sq, eps)
            if s == 1:
                vstar = max(min(vstar, hd_min / r, hd_min * (1.0 - eps_rel)), eps_rel * hd_min)
            else:
                vstar = max(min(vstar, hd_max / r), hd_max * (1.0 + eps_rel))
            vartheta = (1.0 - omega) * vartheta + omega * vstar
        return vartheta


# Diagonal precision update via coordinate-wise Picard iterations with safety cap.
class CBOptDiag(_CBOptBase):
    """Diagonal CBOpt: per-coordinate precision, damped Picard applied coordinatewise."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        s: int = 1,
        weight_decay: float = 2e-3,
        hess_init: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        eps_rel: float = 0.1,
        r: float = 1e-2,
        omega: float = 0.5,
        picard_steps: int = 5,
        eps: float = 1e-12):
        assert s in (1, -1), "s must be +1 or -1"
        assert 0.0 < eps_rel < 1.0 and 0.0 < r < 1.0 and 0.0 < omega <= 1.0
        defaults = dict(lr=lr, s=s, weight_decay=weight_decay, hess_init=hess_init,
                        beta1=beta1, beta2=beta2, eps_rel=eps_rel, r=r, omega=omega,
                        picard_steps=picard_steps, eps=eps)
        super().__init__(params, defaults)
        self.diag_state = {}

    @torch.no_grad()
    def step(self, closure: Optional[ClosureType] = None) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            packed = self._ema_step(group)
            if packed is None:
                continue
            params, m_bar, v_list, t = packed
            s = group["s"]
            wd = group["weight_decay"]
            eps_rel, r, omega = group["eps_rel"], group["r"], group["omega"]
            delta = wd
            k = group["picard_steps"]
            eps = group["eps"]

            numer = torch._foreach_add(m_bar, params, alpha=wd)
            hd_list = torch._foreach_add(v_list, delta)

            raw_margins = []
            for p, num_j, hd_j in zip(params, numer, hd_list):
                st = self.state[p]
                vth = st["vartheta"]
                if bool(torch.isnan(vth).any()):
                    vth = hd_j * (1.0 - s * eps_rel)
                for _ in range(k):
                    u_j = -float(s) * num_j / (hd_j - vth)
                    vstar = 1.0 / (u_j * u_j).clamp_min(eps)
                    if s == 1:
                        vstar = torch.maximum(
                            torch.minimum(torch.minimum(vstar, hd_j / r),
                                          hd_j * (1.0 - eps_rel)), eps_rel * hd_j)
                    else:
                        vstar = torch.maximum(torch.minimum(vstar, hd_j / r),
                                              hd_j * (1.0 + eps_rel))
                    vth = (1.0 - omega) * vth + omega * vstar
                st["vartheta"] = vth
                u_j = -float(s) * num_j / (hd_j - vth)
                p.add_(u_j, alpha=group["lr"])
                # Uncapped coordinate-wise positivity check H_s > 0 with target vartheta*_j = 1 / (u_j)^2.
                vstar_raw = 1.0 / (u_j * u_j).clamp_min(eps)
                raw_margins.append(float((s * (hd_j - vstar_raw)).min()))

            hs_margin_raw = min(raw_margins) if raw_margins else float("nan")
            self.diag_state = dict(structure="diag", s=s, delta=delta, step=t,
                                   hs_margin_raw=hs_margin_raw,
                                   hs_pos_raw=float(hs_margin_raw > 0.0),
                                   eps_rel=eps_rel, r=r, omega=omega, picard_steps=k)
        return loss


# LRD regularization: rank-one target (Prop full_spread), sign-asymmetric safeguards (Thm pd_plus/pd_minus).
class CBOptLRD(_CBOptBase):
    """Low-rank-plus-diagonal CBOpt with sign-asymmetric positivity safeguards (a: Thm pd_plus margin, r: psi floor)."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        s: int = 1,
        weight_decay: float = 2e-3,
        hess_init: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        a: float = 1.0,
        r: float = 1e-3,
        xi: float = 0.1,
        varphi: float = 2.0,
        eps: float = 1e-12):
        assert s in (1, -1), "s must be +1 or -1"
        assert a > 0.0 and r > 0.0 and 0.0 < xi < 1.0 and varphi > 1.0
        defaults = dict(lr=lr, s=s, weight_decay=weight_decay, hess_init=hess_init,
                        beta1=beta1, beta2=beta2, a=a, r=r, xi=xi, varphi=varphi, eps=eps)
        super().__init__(params, defaults)
        self.diag_state = {}
        for group in self.param_groups:
            group["u_S"] = None
            group["psi"] = 1.0 / group["weight_decay"]
            group["s_eff"] = group["s"]

    @torch.no_grad()
    def step(self, closure: Optional[ClosureType] = None) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            packed = self._ema_step(group)
            if packed is None:
                continue
            params, m_bar, v_list, t = packed
            s = group["s"]
            delta = group["weight_decay"]
            a, r, xi, varphi = group["a"], group["r"], group["xi"], group["varphi"]

            rhs = torch.cat([(mb + delta * p.detach()).reshape(-1)
                             for mb, p in zip(m_bar, params)])
            hd = torch.cat([(v + delta).reshape(-1) for v in v_list])
            m, M = float(hd.min()), float(hd.max())
            u_max_sq = (1.0 - xi) / M

            u_S = group["u_S"]
            psi_S = group["psi"]
            u, u_sq = _lrd_solve(hd, rhs, u_S, psi_S, group["s_eff"], group["eps"])

            infeasible = u_sq >= 1.0 / M # Eq feasibility, before the regularizer
            flipped = False
            if s == -1:
                if u_sq <= u_max_sq:
                    group["s_eff"] = -1
                else:
                    group["s_eff"] = 1
                    flipped = True
            s_eff = group["s_eff"]

            n_sminus = 1 if s_eff == -1 else 0
            n_splus = 1 if s_eff == 1 else 0
            n_sminus_infeasible = 1 if (s_eff == -1 and infeasible) else 0

            if s_eff == 1:
                psi = max(r, 1.0 / m + a) # Thm pd_plus
            else:
                psi = max(r, (1.0 / M - u_sq) / varphi) # Thm pd_minus

            # H_s > 0 for the raw (psi -> 0) S^-1 = u u^T, by the nominal sign s (Prop full_spread).
            hs_pos_raw_lrd = float(not infeasible) if s == -1 else 0.0

            group["u_S"] = u.clone()
            group["psi"] = psi
            _assign_flat(params, group["lr"] * u)

            self.diag_state = dict(structure="lrd", s=s, s_eff=s_eff, step=t,
                                   u_sq=u_sq, psi=psi, m=m, M=M,
                                   feasible=not infeasible, flipped=flipped, delta=delta,
                                   n_sminus=n_sminus, n_splus=n_splus,
                                   n_sminus_infeasible=n_sminus_infeasible,
                                   hs_pos_raw_lrd=hs_pos_raw_lrd)
        return loss


# Block-diagonal: one block per parameter tensor, per-block positivity and flip (Thm block).
class CBOptBlock(_CBOptBase):
    """Block-diagonal CBOpt: the LRD construction applied per parameter tensor."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        s: int = 1,
        weight_decay: float = 2e-3,
        hess_init: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        a: float = 1.0,
        r: float = 1e-3,
        xi: float = 0.1,
        varphi: float = 2.0,
        eps: float = 1e-12):
        assert s in (1, -1), "s must be +1 or -1"
        assert a > 0.0 and r > 0.0 and 0.0 < xi < 1.0 and varphi > 1.0
        defaults = dict(lr=lr, s=s, weight_decay=weight_decay, hess_init=hess_init,
                        beta1=beta1, beta2=beta2, a=a, r=r, xi=xi, varphi=varphi, eps=eps)
        super().__init__(params, defaults)
        self.diag_state = {}

    @torch.no_grad()
    def step(self, closure: Optional[ClosureType] = None) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            packed = self._ema_step(group)
            if packed is None:
                continue
            params, m_bar, v_list, t = packed
            s = group["s"]
            delta = group["weight_decay"]
            a, r, xi, varphi = group["a"], group["r"], group["xi"], group["varphi"]

            n_flip, n_infeasible, n_hs_pos_raw = 0, 0, 0
            n_sminus, n_splus, n_sminus_infeasible = 0, 0, 0
            psi_sum = 0.0
            for p, mb, v in zip(params, m_bar, v_list):
                st = self.state[p]
                if "u_S_bd" not in st:
                    st["u_S_bd"] = None
                    st["psi_bd"] = 1.0 / delta
                    st["s_eff_bd"] = s

                rhs = (mb + delta * p.detach()).reshape(-1)
                hd = (v + delta).reshape(-1)
                m_l, M_l = float(hd.min()), float(hd.max())
                u_max_sq = (1.0 - xi) / M_l

                u, u_sq = _lrd_solve(hd, rhs, st["u_S_bd"], st["psi_bd"],
                                     st["s_eff_bd"], group["eps"])
                infeasible = u_sq >= 1.0 / M_l # Eq feasibility, before the regularizer
                if infeasible:
                    n_infeasible += 1
                # H_s > 0 for this block's raw (psi -> 0) S_l^-1 = u_l u_l^T (Prop full_spread).
                if s == -1 and not infeasible:
                    n_hs_pos_raw += 1
                if s == -1:
                    if u_sq <= u_max_sq:
                        st["s_eff_bd"] = -1
                    else:
                        st["s_eff_bd"] = 1
                        n_flip += 1

                se = st["s_eff_bd"]
                if se == -1:
                    n_sminus += 1
                    if infeasible:
                        n_sminus_infeasible += 1
                else:
                    n_splus += 1

                if se == 1:
                    psi = max(r, 1.0 / m_l + a)
                else:
                    psi = max(r, (1.0 / M_l - u_sq) / varphi)

                st["u_S_bd"] = u.clone()
                st["psi_bd"] = psi
                psi_sum += psi
                p.add_(u.view_as(p), alpha=group["lr"])

            n = max(len(params), 1)
            self.diag_state = dict(structure="block", s=s, step=t, n_blocks=len(params),
                                   n_flip=n_flip, n_infeasible=n_infeasible,
                                   n_sminus=n_sminus, n_splus=n_splus,
                                   n_sminus_infeasible=n_sminus_infeasible, delta=delta,
                                   n_hs_pos_raw=n_hs_pos_raw, psi_mean=psi_sum / n)
        return loss


def _lrd_solve(hd: Tensor, rhs: Tensor, u_S, psi_S: float, s_eff: int, eps: float) -> tuple:
    """Solve u = -s_eff (diag(hd) - S)^-1 rhs, S^-1 = u_S u_S^T + psi_S I, via Sherman-Morrison (Prop complexity)."""
    if u_S is None:
        u_S = torch.zeros_like(hd)
    u_norm_sq = float(u_S.pow(2).sum())
    b = 1.0 / (psi_S * (psi_S + u_norm_sq)) if (psi_S + u_norm_sq) > 0 else 0.0
    p_diag = hd - 1.0 / psi_S
    p_diag = torch.where(p_diag.abs() < eps, torch.full_like(p_diag, eps), p_diag)
    pinv_rhs = rhs / p_diag
    if b > 0.0:
        pinv_u = u_S / p_diag
        denom = 1.0 + b * float((u_S * pinv_u).sum())
        x = pinv_rhs - b * pinv_u * (float((u_S * pinv_rhs).sum()) / denom)
    else:
        x = pinv_rhs
    u = -float(s_eff) * x
    return u, float(u.pow(2).sum())


# eps*_SAM = rho grad / ||grad|| (Eq sam_pert), the lower CBO of the indicator candidate (Prop sam_indicator).
class SAM(Optimizer):
    """Sharpness-Aware Minimization with a fixed radius rho over an SGD base step."""

    is_sam = True

    def __init__(
        self,
        params,
        lr: float = 5e-3,
        rho: float = 0.05,
        momentum: float = 0.9,
        weight_decay: float = 1e-5,
        eps: float = 1e-12):
        assert rho > 0.0
        defaults = dict(lr=lr, rho=rho, momentum=momentum, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)
        self.base = torch.optim.SGD(self.param_groups, lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)

    def _grad_norm(self) -> Tensor:
        devices = [p.grad for g in self.param_groups for p in g["params"] if p.grad is not None]
        return torch.norm(torch.stack([g.norm(p=2) for g in devices]), p=2)

    @torch.no_grad()
    def first_step(self) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + group["eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base.step()

    @torch.no_grad()
    def step(self, closure: Optional[ClosureType] = None) -> Optional[Tensor]:
        assert closure is not None, "SAM requires a closure that re-evaluates the gradient"
        with torch.enable_grad():
            loss = closure()
        self.first_step()
        with torch.enable_grad():
            closure()
        self.second_step()
        return loss


def build_optimizer(name: str, params, **kwargs):
    """Factory: cbopt_iso, cbopt_diag, cbopt_lrd, cbopt_block, sam."""
    name = name.lower()
    if name in ("cbopt_iso", "cbopt_diag"):
        cls = CBOptIso if name == "cbopt_iso" else CBOptDiag
        return cls(
            params,
            lr=kwargs.get("lr", 1e-2),
            s=kwargs.get("s", 1),
            weight_decay=kwargs.get("weight_decay", 2e-3),
            hess_init=kwargs.get("hess_init", 0.05),
            beta1=kwargs.get("beta1", 0.9),
            beta2=kwargs.get("beta2", 0.99999),
            eps_rel=kwargs.get("eps_rel", 0.1),
            r=kwargs.get("r", 1e-2),
            omega=kwargs.get("omega", 0.5),
            picard_steps=kwargs.get("picard_steps", 5))
    elif name in ("cbopt_lrd", "cbopt_block"):
        cls = CBOptLRD if name == "cbopt_lrd" else CBOptBlock
        return cls(
            params,
            lr=kwargs.get("lr", 1e-2),
            s=kwargs.get("s", 1),
            weight_decay=kwargs.get("weight_decay", 2e-3),
            hess_init=kwargs.get("hess_init", 0.05),
            beta1=kwargs.get("beta1", 0.9),
            beta2=kwargs.get("beta2", 0.99999),
            a=kwargs.get("a", 1.0),
            r=kwargs.get("r", 1e-3),
            xi=kwargs.get("xi", 0.1),
            varphi=kwargs.get("varphi", 2.0))
    elif name == "sam":
        return SAM(
            params,
            lr=kwargs.get("lr", 5e-3),
            rho=kwargs.get("rho", 0.05),
            momentum=kwargs.get("momentum", 0.9),
            weight_decay=kwargs.get("weight_decay", 1e-5))
    else:
        raise ValueError(f"Unknown optimizer: {name}")
