"""
Standalone benchmark: Mamba2 forward only, NLL (bits) → bpb, timing.
No ZoloZip imports — only torch, mamba_ssm, sentencepiece on the cluster.

Place next to this script: checkpoint .pt, tokenizer.model, data file (see paths below).
"""

from __future__ import annotations

import math
import os
import time

import sentencepiece as spm
import torch
import torch.nn.functional as F
from tqdm import tqdm

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# os.environ.setdefault("USE_HUB_KERNELS", "0")

# --- same directory as this script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "enwik8")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "best.pt")
TOKENIZER_PATH = os.path.join(SCRIPT_DIR, "tokenizer.model")

CHUNK_LEN = 2048
BATCH_SIZE = 2048
DEVICE = "cuda"
TORCH_DTYPE = torch.float32
SEED = 67
BOS_FALLBACK = 2


class SPMTokenizer:
    def __init__(self, model_path: str):
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(model_path)
        bid = self._sp.bos_id()
        uid = self._sp.unk_id()
        self.bos_token_id: int = bid if bid >= 0 else BOS_FALLBACK
        self.unk_id: int = uid if uid >= 0 else 1

    @property
    def sp(self) -> spm.SentencePieceProcessor:
        return self._sp

    def encode(self, text: str) -> list[int]:
        return self._sp.encode(text, out_type=int)


def spm_filtered_ids_and_oov_stats(
    text: str, sp: spm.SentencePieceProcessor
) -> tuple[list[int], dict[str, int | float]]:
    """LM token stream with UNK pieces removed (same order as ZoloZip oov_utils). OOV stats on the side."""
    proto = sp.encode(text, out_type="immutable_proto")
    pieces = [p for p in proto.pieces if p.begin != p.end]
    unk_id = sp.unk_id()
    n_pieces = len(pieces)
    n_bytes = len(text.encode("utf-8"))
    filtered: list[int] = []
    if unk_id < 0:
        filtered = [p.id for p in pieces]
        stats: dict[str, int | float] = {
            "n_pieces": n_pieces,
            "n_unk": 0,
            "n_kept": n_pieces,
            "bytes_oov": 0,
            "n_bytes": n_bytes,
            "frac_pieces": 0.0,
            "frac_bytes": 0.0,
        }
        return filtered, stats
    n_unk = 0
    bytes_oov = 0
    for p in pieces:
        if p.id == unk_id:
            n_unk += 1
            bytes_oov += len(p.surface.encode("utf-8"))
        else:
            filtered.append(p.id)
    n_kept = len(filtered)
    stats = {
        "n_pieces": n_pieces,
        "n_unk": n_unk,
        "n_kept": n_kept,
        "bytes_oov": bytes_oov,
        "n_bytes": n_bytes,
        "frac_pieces": (n_unk / n_pieces) if n_pieces else 0.0,
        "frac_bytes": (bytes_oov / n_bytes) if n_bytes else 0.0,
    }
    return filtered, stats


def load_mamba(
    checkpoint_path: str, tokenizer_path: str, device_str: str, dtype: torch.dtype
) -> tuple[MambaLMHeadModel, SPMTokenizer, int, int, int, torch.device, torch.dtype]:
    device = torch.device(device_str)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tok = SPMTokenizer(tokenizer_path)
    map_loc = device if device.type == "cuda" else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=map_loc, weights_only=False)
    cfg = MambaConfig(**ckpt["cfg"])
    model = MambaLMHeadModel(cfg, device=str(device), dtype=dtype)
    model.load_state_dict(ckpt["model"])
    model.eval()

    vocab_size = len(tok.sp)
    model_vocab = int(model.config.vocab_size)
    return model, tok, vocab_size, model_vocab, tok.bos_token_id, device, dtype


def _nll_bits_sum(logits: torch.Tensor, target_ids: torch.Tensor, vocab_limit: int) -> torch.Tensor:
    """Scalar: sum over batch of CE in bits (-log2 p(true)). logits [B, V], target [B] long."""
    logits = logits[:, :vocab_limit].float()
    logp = F.log_softmax(logits, dim=-1)
    ce_nats = F.nll_loss(logp, target_ids, reduction="sum")
    return ce_nats / math.log(2.0)


@torch.inference_mode()
def batch_forward_nll_bits(
    model: MambaLMHeadModel,
    device: torch.device,
    dtype: torch.dtype,
    batch_chunks: list[list[int]],
    chunk_len: int,
    vocab_size: int,
    bos_token_id: int,
) -> float:
    """
    Same step schedule as ZoloZip MambaModelRunner.predict_batch_chunks, but only [B, V] logits
    per step — no [B, T, V] probability tensor on GPU.
    """
    if not batch_chunks:
        return 0.0
    if any(len(c) != chunk_len for c in batch_chunks):
        raise ValueError("equal chunk lengths required")
    bsz = len(batch_chunks)
    tokens = torch.as_tensor(batch_chunks, dtype=torch.long, device=device)
    max_seqlen = chunk_len + 2
    ip = InferenceParams(max_seqlen=max_seqlen, max_batch_size=bsz)
    ip.key_value_memory_dict = model.backbone.allocate_inference_cache(
        bsz, max_seqlen, dtype=dtype
    )
    ip.seqlen_offset = 0

    total_bits = torch.zeros((), device=device, dtype=torch.float64)

    bos = torch.full((bsz, 1), bos_token_id, dtype=torch.long, device=device)
    hidden = model.backbone(bos, inference_params=ip)
    logits = model.lm_head(hidden[:, -1, :])
    total_bits = total_bits + _nll_bits_sum(logits, tokens[:, 0], vocab_size)
    ip.seqlen_offset += 1

    for t in range(chunk_len - 1):
        hidden = model.backbone(tokens[:, t : t + 1], inference_params=ip)
        logits = model.lm_head(hidden[:, -1, :])
        total_bits = total_bits + _nll_bits_sum(logits, tokens[:, t + 1], vocab_size)
        ip.seqlen_offset += 1

    return float(total_bits.item())


def _build_chunk_batches(
    token_ids: list[int], chunk_len: int, batch_size: int
) -> list[list[list[int]]]:
    n = len(token_ids)
    n_full = (n // chunk_len) * chunk_len
    if n_full == 0:
        return []
    flat = token_ids[:n_full]
    chunks: list[list[int]] = [
        flat[i : i + chunk_len] for i in range(0, n_full, chunk_len)
    ]
    batches: list[list[list[int]]] = []
    for i in range(0, len(chunks), batch_size):
        block = chunks[i : i + batch_size]
        if len(block) < batch_size:
            break
        batches.append(block)
    return batches


def main() -> None:
    model, tok, vocab_size, model_vocab, bos_id, device, dtype = load_mamba(
        CHECKPOINT_PATH, TOKENIZER_PATH, DEVICE, TORCH_DTYPE
    )
    if model_vocab < vocab_size:
        print(
            f"warning: model vocab {model_vocab} < SPM vocab {vocab_size}; "
            "logits are sliced to SPM size."
        )

    with open(DATA_PATH, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    token_ids, oov = spm_filtered_ids_and_oov_stats(text, tok.sp)
    n_bytes = int(oov["n_bytes"])
    n_tokens = len(token_ids)
    # bit/token → bit/byte of whole file (OOV bytes not predicted by LM; side channel separate)
    bpt_to_bpb = n_tokens / n_bytes if n_bytes else 0.0
    bytes_covered = n_bytes - int(oov["bytes_oov"])
    bpt_to_bpb_covered = (n_tokens / bytes_covered) if bytes_covered > 0 else 0.0

    batches = _build_chunk_batches(token_ids, CHUNK_LEN, BATCH_SIZE)
    if not batches:
        print("No full batches; increase data or lower chunk_len / batch_size.")
        return

    first = batches[0]

    # Warmup (not in main wall clock)
    batch_forward_nll_bits(
        model, device, dtype, first, CHUNK_LEN, vocab_size, bos_id
    )

    t0_ref = time.perf_counter()
    batch_forward_nll_bits(
        model, device, dtype, first, CHUNK_LEN, vocab_size, bos_id
    )
    t_ref = time.perf_counter() - t0_ref

    sum_nll_bits = 0.0
    n_positions = 0

    t0 = time.perf_counter()
    for batch in tqdm(batches, desc="batches", unit="batch"):
        sum_nll_bits += batch_forward_nll_bits(
            model, device, dtype, batch, CHUNK_LEN, vocab_size, bos_id
        )
        n_positions += len(batch) * CHUNK_LEN
    t1 = time.perf_counter()

    wall = t1 - t0
    wall_adj = max(wall - t_ref, 1e-9)
    n_batches = len(batches)

    mean_nll_bpt = sum_nll_bits / n_positions if n_positions else 0.0
    mean_bpb = mean_nll_bpt * bpt_to_bpb
    mean_bpb_covered = mean_nll_bpt * bpt_to_bpb_covered

    print(f"data: {DATA_PATH}")
    print(
        f"OOV/UNK: pieces {oov['n_unk']:,} / {oov['n_pieces']:,} "
        f"({100.0 * float(oov['frac_pieces']):.4f}%); "
        f"utf8 bytes in unk surfaces {oov['bytes_oov']:,} / {n_bytes:,} "
        f"({100.0 * float(oov['frac_bytes']):.4f}%)"
    )
    print(
        f"LM stream: {n_tokens:,} ids (unk stripped)  utf8_bytes file: {n_bytes:,}  "
        f"bpt→bpb @file: {bpt_to_bpb:.6f}  @bytes w/o oov utf8: {bpt_to_bpb_covered:.6f}"
    )
    print(f"chunk_len={CHUNK_LEN}  batch_size={BATCH_SIZE}  full_batches={n_batches}")
    print(f"pred_positions: {n_positions:,}")
    print(f"mean NLL (true token): {mean_nll_bpt:.4f} bit/token")
    print(f"mean bpb (× file bytes): {mean_bpb:.4f}  (× bytes minus oov utf8): {mean_bpb_covered:.4f}")
    print(f"wall_batches: {wall:.3f}s  minus_ref_first_batch: {t_ref:.3f}s  => adj: {wall_adj:.3f}s")
    print(f"avg wall / batch (adj): {wall_adj / n_batches:.4f}s")
    print(f"throughput (adj): {n_positions / wall_adj:.1f} tok/s")


if __name__ == "__main__":
    main()
