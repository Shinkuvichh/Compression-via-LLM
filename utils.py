"""Small shared helpers."""

import numpy as np


def safe_normalize_probs(probs: np.ndarray) -> np.ndarray:
    if not np.isfinite(probs).all():
        return np.full_like(probs, 1.0 / len(probs), dtype=np.float64)
    total = float(probs.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(probs, 1.0 / len(probs), dtype=np.float64)
    return probs / total


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"
