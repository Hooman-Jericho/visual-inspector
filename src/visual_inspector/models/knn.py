"""k-Nearest Neighbors with faiss-cpu backend and pure-torch fallback.

Provides a unified interface so the rest of the codebase doesn't need
to know which backend is being used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class KNNBackend(ABC):
    """Abstract base class for kNN backends."""

    @abstractmethod
    def fit(self, memory_bank: torch.Tensor) -> None:
        """Build the index from memory bank embeddings.

        Args:
            memory_bank: Tensor of shape (M, D).
        """

    @abstractmethod
    def query(self, queries: torch.Tensor, k: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Find k nearest neighbors.

        Args:
            queries: Tensor of shape (Q, D).
            k: Number of neighbors.

        Returns:
            distances: Tensor of shape (Q, k).
            indices: Tensor of shape (Q, k).
        """


class FaissKNN(KNNBackend):
    """kNN using faiss-cpu for fast exact search."""

    def __init__(self) -> None:
        import faiss

        self._faiss = faiss
        self._index: faiss.IndexFlatL2 | None = None

    def fit(self, memory_bank: torch.Tensor) -> None:
        data = memory_bank.cpu().numpy().astype("float32")
        d = data.shape[1]
        self._index = self._faiss.IndexFlatL2(d)
        self._index.add(data)

    def query(self, queries: torch.Tensor, k: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._index is not None, "Must call fit() before query()"
        q = queries.cpu().numpy().astype("float32")
        distances, indices = self._index.search(q, k)
        # Faiss IndexFlatL2 returns squared Euclidean distance; take sqrt to match TorchKNN
        dists = torch.from_numpy(distances).to(queries.device)
        dists = torch.sqrt(torch.clamp(dists, min=0.0))
        return (
            dists,
            torch.from_numpy(indices).to(queries.device),
        )


class TorchKNN(KNNBackend):
    """kNN using pure PyTorch cdist. Slower but always available."""

    def __init__(self) -> None:
        self._memory_bank: torch.Tensor | None = None

    def fit(self, memory_bank: torch.Tensor) -> None:
        self._memory_bank = memory_bank

    def query(self, queries: torch.Tensor, k: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._memory_bank is not None, "Must call fit() before query()"
        # Move to same device
        mb = self._memory_bank.to(queries.device)
        # Compute distances in batches to manage memory
        batch_size = 2048
        all_distances = []
        all_indices = []

        for start in range(0, queries.shape[0], batch_size):
            end = min(start + batch_size, queries.shape[0])
            batch = queries[start:end]
            dists = torch.cdist(batch, mb)  # (batch, M)
            topk_dists, topk_indices = torch.topk(dists, k, dim=1, largest=False)
            all_distances.append(topk_dists)
            all_indices.append(topk_indices)

        return torch.cat(all_distances, dim=0), torch.cat(all_indices, dim=0)


def get_knn_backend(prefer_faiss: bool = True) -> KNNBackend:
    """Get the best available kNN backend.

    Args:
        prefer_faiss: If True, try faiss-cpu first.

    Returns:
        A KNNBackend instance.
    """
    if prefer_faiss:
        try:
            import faiss  # noqa: F401

            return FaissKNN()
        except ImportError:
            pass
    return TorchKNN()
