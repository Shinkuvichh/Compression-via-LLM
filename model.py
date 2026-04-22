"""Base model runner and n-gram model abstractions."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModelRunner(ABC):
    """Common interface for neural LM runner used by codec.

    Contract
    --------
    - ``init_batch_state(B)`` allocates InferenceParams, feeds BOS for all B streams,
      and returns a state whose ``.probs`` is ``[B, V]`` numpy float32 — the LM
      distribution for predicting ``tokens[:, 0]``.
    - ``step_batch(tokens_b, state)`` feeds ``tokens_b [B]`` (the token just
      encoded/decoded) into the backbone, advances the recurrent state **in-place**,
      and returns ``(probs_bv [B, V] float32 numpy, updated_state)``.  The returned
      probs predict the **next** token.
    - Both methods must be usable under ``torch.inference_mode()``.
    """

    tokenizer: Any
    vocab_size: int
    bos_token_id: int

    @abstractmethod
    def init_batch_state(self, batch_size: int) -> Any:
        """Allocate state + BOS forward. Returns state with .probs [B, V]."""

    @abstractmethod
    def step_batch(self, tokens_b: np.ndarray, state: Any) -> tuple[np.ndarray, Any]:
        """Feed tokens_b [B int]. Return (probs_bv [B, V] float32, new_state)."""


class BaseNgramModel(ABC):
    """Interface for the parallel n-gram model used in confidence-skip mode.

    Each codec batch creates B independent n-gram contexts (one per chunk).
    Contexts are reset to empty at the start of every chunk (``init_states``).

    Contract
    --------
    - ``init_states(B)`` → opaque state object for B parallel streams.
    - ``step(state, prev_tokens_b)`` → ``(logits_bv [B, V] float32, new_state)``.
      ``prev_tokens_b`` is the token observed at the previous time step (or BOS at
      ``t=0``).  The codec applies softmax internally; return raw logits.
    """

    vocab_size: int

    @abstractmethod
    def init_states(self, batch_size: int) -> Any:
        """Return initial state for B parallel empty contexts."""

    @abstractmethod
    def step(self, state: Any, prev_tokens_b: np.ndarray) -> tuple[np.ndarray, Any]:
        """Update B contexts with prev_tokens_b, return (logits_bv [B,V], new_state)."""
