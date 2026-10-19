"""Greedy k-center coreset selection.

Implements the coreset subsampling from PatchCore (Roth et al., CVPR 2022):
random Gaussian projection to lower dimension for distance computation,
then greedy k-center to select representative patch embeddings.
"""

from __future__ import annotations

import torch


def random_projection_matrix(
    d_in: int,
    d_out: int,
    seed: int = 42,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate a random Gaussian projection matrix.

    Projects from d_in to d_out dimensions. Used to speed up
    distance computations during coreset selection.

    Args:
        d_in: Input dimension.
        d_out: Output (projected) dimension.
        seed: Random seed for reproducibility.
        device: Target device.

    Returns:
        Projection matrix of shape (d_out, d_in).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Random Gaussian matrix, scaled for approximate distance preservation
    P: torch.Tensor = torch.randn(d_out, d_in, generator=gen, device="cpu") / float(d_out**0.5)
    if device is not None and device.type != "cpu":
        P = P.to(device)
    return P


def greedy_kcenter(
    embeddings: torch.Tensor,
    n_select: int,
    projection_dim: int = 128,
    seed: int = 42,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Greedy k-center coreset selection.

    Selects n_select points from embeddings such that the maximum
    distance from any point to its nearest selected point is minimized.

    Args:
        embeddings: Patch embeddings of shape (N, D).
        n_select: Number of coreset points to select.
        projection_dim: Dimension for random projection (speeds up distances).
        seed: Random seed.
        batch_size: Batch size for distance computation (controls memory).

    Returns:
        Indices of selected coreset points, shape (n_select,).
    """
    N, D = embeddings.shape
    if n_select >= N:
        return torch.arange(N, device=embeddings.device)

    device = embeddings.device

    # Project to lower dimension for fast distance computation
    if projection_dim < D:
        P = random_projection_matrix(D, projection_dim, seed=seed, device=device)
        projected = embeddings @ P.T  # (N, projection_dim)
    else:
        projected = embeddings

    # Initialize: pick a random starting point
    gen = torch.Generator(device="cpu").manual_seed(seed)
    start_idx = int(torch.randint(N, (1,), generator=gen).item())

    selected_indices = torch.zeros(n_select, dtype=torch.long, device=device)
    selected_indices[0] = start_idx

    # min_distances[i] = min distance from point i to any selected point
    min_distances = torch.full((N,), float("inf"), device=device)

    # Update distances from the first selected point
    _update_min_distances_batched(
        projected, projected[start_idx : start_idx + 1], min_distances, batch_size
    )

    for step in range(1, n_select):
        # Select the point with the largest minimum distance
        new_idx = int(torch.argmax(min_distances).item())
        selected_indices[step] = new_idx

        # Update min distances
        _update_min_distances_batched(
            projected, projected[new_idx : new_idx + 1], min_distances, batch_size
        )

        # Set distance of selected point to 0 so it won't be picked again
        min_distances[new_idx] = 0.0

    return selected_indices


def _update_min_distances_batched(
    all_points: torch.Tensor,
    new_center: torch.Tensor,
    min_distances: torch.Tensor,
    batch_size: int,
) -> None:
    """Update minimum distances in batches to control memory.

    Args:
        all_points: All projected points, shape (N, d).
        new_center: Newly selected center, shape (1, d).
        min_distances: Current minimum distances, shape (N,). Modified in-place.
        batch_size: Number of points per batch.
    """
    N = all_points.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = all_points[start:end]
        dists = torch.cdist(batch, new_center).squeeze(-1)  # (batch_size,)
        min_distances[start:end] = torch.minimum(min_distances[start:end], dists)
