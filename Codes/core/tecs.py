# Component: TECS Core Metric
# Source: research/method-design.md §2
# Ablation config key: N/A (always on)

"""TECS (TDA-Editing Consistency Score) computation and null baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class TECSResult:
    """Result of a single TECS computation for one fact."""

    fact_id: int
    tecs_real: float              # cos(Δθ_E, g_M) at edit layer
    tecs_null_a: List[float]      # TECS with unrelated edit directions
    tecs_placebo: dict            # {layer_offset: tecs_value} for Null-B
    edit_success: bool
    angular_variance: float       # mean pairwise cosine among top-k gradients
    metadata: dict                # additional info


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors after flattening."""
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()

    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def compute_tecs(
    delta_weight: torch.Tensor,
    aggregated_gradient: torch.Tensor,
) -> float:
    """Compute TECS = cos(Δθ_E, g_M) for a single (fact, layer) pair.

    Args:
        delta_weight: ROME rank-1 update Δθ at the target layer.
        aggregated_gradient: Aggregated TDA gradient g_M at the same layer.

    Returns:
        Cosine similarity (scalar).
    """
    return cosine_similarity_flat(delta_weight, aggregated_gradient)


def compute_null_a(
    aggregated_gradient: torch.Tensor,
    unrelated_deltas: List[torch.Tensor],
) -> List[float]:
    """Compute Null-A: TECS with unrelated facts' edit directions.

    Args:
        aggregated_gradient: g_M for the target fact.
        unrelated_deltas: List of Δθ_E from unrelated facts.

    Returns:
        List of null TECS values.
    """
    return [cosine_similarity_flat(d, aggregated_gradient) for d in unrelated_deltas]


def compute_mean_pairwise_cosine(gradients: List[torch.Tensor]) -> float:
    """Compute mean pairwise cosine similarity among gradient vectors.

    A value near 0 means gradients point in essentially random directions.
    A value near 1 means they are highly aligned (strong directional signal).

    In very high-dimensional spaces (~10M parameters for GPT-2-XL MLP weights),
    random vectors have expected cosine similarity ~0 due to concentration of
    measure. Therefore, even small positive values (e.g., 0.001) indicate
    meaningful directional alignment. The kill threshold should be much lower
    than in low-dimensional settings.

    Used as Kill Gate 1: if mean pairwise cosine < threshold, the gradient
    signal is too noisy for TECS to be meaningful.
    """
    if len(gradients) < 2:
        return 0.0

    flat = [g.reshape(-1).float() for g in gradients]
    n = len(flat)
    total_cos = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            cos = F.cosine_similarity(flat[i].unsqueeze(0), flat[j].unsqueeze(0)).item()
            total_cos += cos
            count += 1

    return total_cos / count if count > 0 else 0.0


# Backward-compatible alias
compute_angular_variance = compute_mean_pairwise_cosine
