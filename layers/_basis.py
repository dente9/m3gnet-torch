# m3gnet/layers/_basis.py (Final Fixed Version 2)

"""Basis function expansions for distances and angles."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import math

# associated_legendre_polynomials, SphericalHarmonicsBasis, GaussianBasis, SphericalBesselBasis are unchanged...
def associated_legendre_polynomials(costheta, max_l=3):
    """
    Computes associated Legendre polynomials P_l^m(cos(theta)).
    """
    x = costheta.unsqueeze(-1)
    sintheta_sq = torch.clamp(1 - x * x, min=1e-10)
    sintheta = sintheta_sq.sqrt()
    
    num_p = sum(l + 1 for l in range(max_l + 1))
    p = torch.zeros(x.shape[0], num_p, device=x.device)
    
    p[:, 0] = 1.0
    if max_l < 1: return p
    p[:, 1] = x.squeeze(-1)
    p[:, 2] = -sintheta.squeeze(-1)
    if max_l < 2: return p
    p[:, 3] = 0.5 * (3 * x * x - 1).squeeze(-1)
    p[:, 4] = -3 * x.squeeze(-1) * sintheta.squeeze(-1)
    p[:, 5] = 3 * sintheta_sq.squeeze(-1)
    if max_l < 3: return p
    p[:, 6] = 0.5 * (5 * x**3 - 3 * x).squeeze(-1)
    p[:, 7] = -1.5 * (5 * x**2 - 1).squeeze(-1) * sintheta.squeeze(-1)
    p[:, 8] = 15 * x.squeeze(-1) * sintheta_sq.squeeze(-1)
    p[:, 9] = -15 * (sintheta_sq**1.5).squeeze(-1)
    return p

class SphericalHarmonicsBasis(nn.Module):
    """Spherical harmonics basis function expansion (pure torch)."""
    def __init__(self, max_l: int):
        super().__init__()
        if max_l > 3:
            raise NotImplementedError("This implementation only supports max_l <= 3")
        self.max_l = max_l
        
        norm_consts = []
        for l in range(max_l + 1):
            for m in range(l + 1):
                # Calculate normalization constant for P_l^m
                val = (2 * l + 1) / (4 * torch.pi) * (math.factorial(l - m) / math.factorial(l + m))
                norm_consts.append(torch.sqrt(torch.tensor(val, dtype=torch.float32)))
        
        # Register as a buffer, converting the list of tensors to a single tensor
        self.register_buffer("norm_consts", torch.stack(norm_consts))

    def forward(self, costheta: torch.Tensor, phi: Optional[torch.Tensor] = None) -> torch.Tensor:
        if phi is None:
            phi = torch.zeros_like(costheta)
        
        p_lm = associated_legendre_polynomials(costheta, self.max_l)
        
        sh_basis = []
        p_idx = 0
        norm_idx = 0
        for l in range(self.max_l + 1):
            sh_basis.append(self.norm_consts[norm_idx] * p_lm[:, p_idx])
            p_idx += 1
            norm_idx += 1
            for m in range(1, l + 1):
                prefactor = np.sqrt(2) * self.norm_consts[norm_idx]
                sh_basis.append(prefactor * p_lm[:, p_idx] * torch.cos(m * phi))
                sh_basis.append(prefactor * p_lm[:, p_idx] * torch.sin(m * phi))
                p_idx += 1
                norm_idx += 1
                
        return torch.stack(sh_basis, dim=1)

class GaussianBasis(nn.Module):
    def __init__(self, centers: np.ndarray, width: float, trainable: bool = False):
        super().__init__()
        centers = torch.tensor(centers, dtype=torch.float32).view(1, -1)
        if trainable:
            self.centers = nn.Parameter(centers, requires_grad=True)
        else:
            self.register_buffer("centers", centers)
        self.width = width

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((r.view(-1, 1) - self.centers)**2) / self.width)

class SphericalBesselBasis(nn.Module):
    def __init__(self, max_l: int, max_n: int, cutoff: float):
        super().__init__()
        self.max_l = max_l
        self.max_n = max_n
        self.cutoff = cutoff
        
        # Pre-calculate zeros and normalization factor and register as buffers
        j0_zeros = torch.arange(1, max_n + 1, dtype=torch.float32) * torch.pi
        self.register_buffer("j0_zeros", j0_zeros)
        
        norm_factor = torch.sqrt(torch.tensor(2.0 / self.cutoff, dtype=torch.float32))
        self.register_buffer("norm_factor", norm_factor)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r = r.view(-1, 1)
        # Use the pre-calculated zeros and normalization factor
        n = self.j0_zeros / self.cutoff
        # Add a small epsilon to r in the denominator for numerical stability
        return self.norm_factor * torch.sin(n * r) / (r + 1e-8)

# <<<<<<<<<<<<<<<<<<<< FIX IS HERE <<<<<<<<<<<<<<<<<<<<
class SphericalBesselWithHarmonics(nn.Module):
    """Combines Spherical Bessel and Spherical Harmonics."""
    def __init__(self, max_n: int, max_l: int, cutoff: float):
        super().__init__()
        self.sbf = SphericalBesselBasis(max_l, max_n, cutoff)
        self.shf = SphericalHarmonicsBasis(max_l)
        self.max_n = max_n
        self.max_l = max_l

    def forward(self, r: torch.Tensor, costheta: torch.Tensor, phi: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Ensure costheta is always 1D before passing to shf
        costheta_1d = costheta.squeeze(-1) if costheta.dim() == 2 else costheta
        
        sbf = self.sbf(r)
        shf = self.shf(costheta_1d, phi)
        
        n_sbf = sbf.shape[-1]
        n_shf = shf.shape[-1]

        sbf_view = sbf.view(-1, 1, n_sbf).expand(-1, n_shf, -1)
        shf_view = shf.view(-1, n_shf, 1).expand(-1, -1, n_sbf)
        
        return (sbf_view * shf_view).view(-1, n_sbf * n_shf)