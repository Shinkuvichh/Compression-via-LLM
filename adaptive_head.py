"""Adaptive head reused from Nacrith."""

import numpy as np


class AdaptiveHead:
    def __init__(self, vocab_size: int = 49152, lr: float = 0.001):
        self.vocab_size = vocab_size
        self.lr = lr
        self.bias = np.zeros(vocab_size, dtype=np.float64)
        self._log_buf = np.zeros(vocab_size, dtype=np.float64)
        self._grad_buf = np.zeros(vocab_size, dtype=np.float64)

    def reset(self):
        self.bias[:] = 0

    def adjust(self, probs: np.ndarray) -> np.ndarray:
        log_buf = self._log_buf
        np.log(probs + 1e-10, out=log_buf)
        log_buf += self.bias
        log_buf -= log_buf.max()
        np.exp(log_buf, out=log_buf)
        log_buf /= log_buf.sum()
        return log_buf

    def update(self, actual_token: int, adjusted_probs: np.ndarray):
        grad = self._grad_buf
        np.copyto(grad, adjusted_probs)
        grad[actual_token] -= 1.0
        self.bias -= self.lr * grad
