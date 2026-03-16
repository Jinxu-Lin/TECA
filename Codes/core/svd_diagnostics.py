"""SVD projection diagnostics: assess spectral confound risk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class SVDDiagnosticResult:
    """Result of SVD projection analysis for one layer."""

    layer_idx: int
    top_k: int
    delta_projection_ratio: float  # fraction of ||Δθ|| in top-k SV subspace
    gradient_projection_ratio: float  # fraction of ||g|| in top-k SV subspace
    spectral_risk: str  # "low" / "medium" / "high"
    singular_values: List[float]  # top-k singular values (for logging)


def svd_projection_diagnostic(
    weight_matrix: torch.Tensor,
    delta_weight: torch.Tensor,
    gradient: torch.Tensor,
    top_k: int = 10,
) -> SVDDiagnosticResult:
    """Compute how much of Δθ and g live in the top-k singular vector subspace of W.

    If both vectors have >80% projection onto top-k SVs, the spectral confound
    risk is high: any vector related to this layer's computation will naturally
    align with the dominant singular directions.

    Args:
        weight_matrix: The weight matrix W at the target layer.
        delta_weight: ROME edit delta Δθ (same shape as W).
        gradient: TDA gradient g (same shape as W).
        top_k: Number of top singular vectors to consider.

    Returns:
        SVDDiagnosticResult with projection ratios and risk assessment.
    """
    W = weight_matrix.float().cpu()
    delta = delta_weight.float().cpu()
    grad = gradient.float().cpu()

    # Flatten to 2D if needed (already should be for MLP weight)
    if W.dim() == 1:
        W = W.unsqueeze(0)

    # SVD of the weight matrix
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    # Top-k right singular vectors (columns of V = rows of Vh)
    Vk = Vh[:top_k, :]  # [top_k, d_ff]

    # Project Δθ onto top-k subspace
    delta_flat = delta.reshape(W.shape)
    delta_proj = _projection_ratio(delta_flat, U, S, Vh, top_k)

    # Project g onto top-k subspace
    grad_flat = grad.reshape(W.shape)
    grad_proj = _projection_ratio(grad_flat, U, S, Vh, top_k)

    # Risk assessment
    if delta_proj > 0.8 and grad_proj > 0.8:
        risk = "high"
    elif delta_proj > 0.5 or grad_proj > 0.5:
        risk = "medium"
    else:
        risk = "low"

    return SVDDiagnosticResult(
        layer_idx=-1,  # caller should set this
        top_k=top_k,
        delta_projection_ratio=delta_proj,
        gradient_projection_ratio=grad_proj,
        spectral_risk=risk,
        singular_values=S[:top_k].tolist(),
    )


def _projection_ratio(
    matrix: torch.Tensor,
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
    top_k: int,
) -> float:
    """Compute ||P_k(M)||_F / ||M||_F where P_k projects onto top-k SV subspace.

    For a matrix M and SVD of reference W = U S V^T:
    P_k(M) = U_k U_k^T M V_k V_k^T  (project both left and right)

    Simplified: use Frobenius norm ratio.
    """
    M = matrix.float()
    total_norm = torch.norm(M, p="fro").item()
    if total_norm < 1e-12:
        return 0.0

    Uk = U[:, :top_k]  # [d_model, top_k]
    Vk = Vh[:top_k, :]  # [top_k, d_ff]

    # Project: P_k(M) = Uk @ (Uk^T @ M @ Vk^T) @ Vk
    projected = Uk @ (Uk.T @ M @ Vk.T) @ Vk
    proj_norm = torch.norm(projected, p="fro").item()

    return proj_norm / total_norm
