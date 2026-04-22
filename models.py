"""Mamba model runner for codec (local checkpoint + SentencePiece)."""

from dataclasses import dataclass
import os

import numpy as np
import sentencepiece as spm

# Must be set before importing torch for deterministic CUDA kernels.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["USE_HUB_KERNELS"] = "0"

import torch

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams

from config import CodecConfig
from model import BaseModelRunner


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base, path))


@dataclass
class MambaBatchState:
    """Mutable recurrent state for B parallel streams.

    ``ip`` is an InferenceParams whose key_value_memory_dict tensors are
    updated **in-place** by each call to ``step_batch``.  Do not share a
    MambaBatchState between two codec passes simultaneously.
    ``probs`` is the LM distribution for the **next** token: shape ``[B, V]``
    float32 numpy.
    """
    ip: InferenceParams
    probs: np.ndarray  # [B, V]


class SPMTokenizer:
    """SentencePiece wrapper."""

    def __init__(self, model_path: str):
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(model_path)
        bid = self._sp.bos_id()
        uid = self._sp.unk_id()
        self.bos_token_id: int = bid if bid >= 0 else 0
        self.unk_id: int = uid if uid >= 0 else 1

    @property
    def sp(self) -> spm.SentencePieceProcessor:
        return self._sp

    def encode(self, text: str) -> list[int]:
        return self._sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self._sp.decode(ids)


class MambaModelRunner(BaseModelRunner):
    def __init__(self, config: CodecConfig):
        self.config = config
        self.device = torch.device(config.device)

        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        tok_path = _resolve_path(config.tokenizer_path)
        self.tokenizer = SPMTokenizer(tok_path)

        ckpt_path = _resolve_path(config.checkpoint_path)
        map_loc = self.device if self.device.type == "cuda" else "cpu"
        ckpt = torch.load(ckpt_path, map_location=map_loc, weights_only=False)
        cfg = MambaConfig(**ckpt["cfg"])
        self.model = MambaLMHeadModel(cfg, device=str(self.device), dtype=config.torch_dtype)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.vocab_size: int = len(self.tokenizer.sp)
        self.model_vocab_size: int = int(self.model.config.vocab_size)
        self.bos_token_id: int = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else config.bos_token_id
        )
        # max_seqlen upper bound: BOS + chunk_len tokens
        self._max_seqlen = config.normalized_chunk_len() + 2

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _allocate_ip(self, batch_size: int) -> InferenceParams:
        ip = InferenceParams(max_seqlen=self._max_seqlen, max_batch_size=batch_size)
        ip.key_value_memory_dict = self.model.backbone.allocate_inference_cache(
            batch_size, self._max_seqlen, dtype=self.config.torch_dtype
        )
        ip.seqlen_offset = 0
        return ip

    def _forward_step(
        self,
        input_ids: torch.Tensor,   # [B, 1]
        ip: InferenceParams,
    ) -> torch.Tensor:
        """Single-token forward; updates ip.key_value_memory_dict in-place."""
        with torch.inference_mode():
            out = self.model(input_ids, inference_params=ip, num_last_tokens=1)
        return out.logits.squeeze(1)  # [B, V]

    def _logits_to_probs(self, logits: torch.Tensor) -> np.ndarray:
        if logits.shape[-1] != self.vocab_size:
            logits = logits[..., : self.vocab_size]
        probs = torch.softmax(logits.float(), dim=-1)
        return probs.detach().cpu().numpy().astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # BaseModelRunner interface
    # ------------------------------------------------------------------

    def init_batch_state(self, batch_size: int) -> MambaBatchState:
        """Allocate InferenceParams, feed BOS, return state predicting tokens[:,0]."""
        ip = self._allocate_ip(batch_size)
        bos = torch.full(
            (batch_size, 1), self.bos_token_id, dtype=torch.long, device=self.device
        )
        logits = self._forward_step(bos, ip)
        ip.seqlen_offset += 1
        return MambaBatchState(ip=ip, probs=self._logits_to_probs(logits))

    def step_batch(
        self,
        tokens_b: np.ndarray,          # [B] int — token just encoded/decoded
        state: MambaBatchState,
    ) -> tuple[np.ndarray, MambaBatchState]:
        """Feed tokens_b, advance state in-place. Return (probs [B,V], state)."""
        input_ids = (
            torch.as_tensor(tokens_b, dtype=torch.long, device=self.device)
            .view(-1, 1)
        )
        logits = self._forward_step(input_ids, state.ip)
        state.ip.seqlen_offset += 1
        state.probs = self._logits_to_probs(logits)
        return state.probs, state
