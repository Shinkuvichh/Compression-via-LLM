"""Text codec: neural LM + optional n-gram confidence skip + arithmetic coding."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from arithmetic_coder import ArithmeticDecoder, ArithmeticEncoder, CdfConverter
from config import CodecConfig
from io_format import (
    FLAG_CONFIDENCE_SKIP,
    FLAG_OOV_BYPASS,
    ArchiveMetadata,
    ArchiveWriter,
    ChunkRecord,
    OovRecord,
    pack_archive,
    unpack_archive,
)
from oov_utils import extract_oov_from_text, merge_text_from_filtered
from model import BaseModelRunner, BaseNgramModel

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass(slots=True)
class CodecRunStats:
    num_tokens: int = 0
    encode_seconds: float = 0.0
    tokens_per_second: float = 0.0
    skipped_tokens: int = 0


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax. [B, V] → [B, V] float32."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_l = np.exp(shifted, dtype=np.float32)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


def _entropy_bits_batch(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits for each row. [B, V] → [B] float32."""
    log2p = np.log2(np.maximum(probs, 1e-30, dtype=np.float32))
    return -(probs * log2p).sum(axis=1)


# ---------------------------------------------------------------------------
# ZoloArithmeticCodec
# ---------------------------------------------------------------------------

class ZoloArithmeticCodec:
    def __init__(
        self,
        config: CodecConfig,
        runner: BaseModelRunner,
        ngram: "BaseNgramModel | None" = None,
    ):
        self.config = config
        self.runner = runner
        # ngram is required when use_confidence_skip=True but may be None (skip disabled)
        self.ngram = ngram
        self.cdf_converter = CdfConverter(self.runner.vocab_size)
        self.last_run_stats = CodecRunStats()
        self.verbose = bool(config.verbose)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _split_tokens(self, token_ids: list[int]) -> tuple[list[list[int]], list[int] | None]:
        """Split into (full_chunks, tail_chunk_or_None)."""
        chunk_len = self.config.normalized_chunk_len()
        n = len(token_ids)
        n_full = (n // chunk_len) * chunk_len
        full = [token_ids[i : i + chunk_len] for i in range(0, n_full, chunk_len)]
        tail: list[int] | None = token_ids[n_full:] if n_full < n else None
        return full, tail

    def _iter_batches(self, items, desc: str):
        if self.verbose and tqdm is not None:
            return tqdm(items, desc=desc, unit="batch")
        return items

    def _config_from_archive(self, meta: ArchiveMetadata) -> CodecConfig:
        return replace(
            self.config,
            chunk_len=meta.chunk_len,
            batch_size=meta.batch_size,
            use_confidence_skip=bool(meta.flags & FLAG_CONFIDENCE_SKIP),
            use_oov_bypass=bool(meta.flags & FLAG_OOV_BYPASS),
            skip_threshold=meta.skip_threshold,
            ngram_order=meta.ngram_order,
        )

    # ------------------------------------------------------------------
    # Skip logic
    # ------------------------------------------------------------------

    def _compute_skip(
        self,
        ngram_state: object,
        prev_tokens_b: np.ndarray,
        global_step: int,
    ) -> "tuple[np.ndarray | None, np.ndarray | None, object]":
        """Advance n-gram state; return (skip_mask [B]|None, ngram_probs [B,V]|None, new_state).

        Always advances n-gram state so the context stays in sync.
        Returns skip_mask=None when warmup is active or all B are not skipped.
        """
        if self.ngram is None:
            return None, None, ngram_state

        ngram_logits, ngram_state = self.ngram.step(ngram_state, prev_tokens_b)

        if global_step < self.config.warmup_tokens:
            return None, None, ngram_state

        ngram_probs = _softmax_rows(ngram_logits)  # [B, V]
        skip_mask = _entropy_bits_batch(ngram_probs) < self.config.skip_threshold  # [B]

        if not skip_mask.any():
            return None, None, ngram_state

        return skip_mask, ngram_probs, ngram_state

    # ------------------------------------------------------------------
    # Core encode/decode batch loops
    # ------------------------------------------------------------------

    def _encode_batch(
        self,
        tokens_bt: np.ndarray,      # [B, T] int
        global_offset: int,
        ngram_state: object,
    ) -> "tuple[list[ChunkRecord], int, object]":
        """Encode B chunks of length T in parallel.

        Returns (records, n_skipped, updated_ngram_state).
        ngram_state is advanced by T steps per stream.
        """
        B, T = tokens_bt.shape
        encoders = [ArithmeticEncoder() for _ in range(B)]
        state = self.runner.init_batch_state(B)
        prev_tokens = np.full(B, self.runner.bos_token_id, dtype=np.int64)
        total_skipped = 0

        for t in range(T):
            skip_mask, ngram_probs, ngram_state = self._compute_skip(
                ngram_state, prev_tokens, global_offset + t
            )

            # LLM probs for this step (already in state from previous step / BOS)
            probs_bv = state.probs  # [B, V]

            # Replace skipped rows with n-gram probs
            if skip_mask is not None:
                probs_bv = probs_bv.copy()
                probs_bv[skip_mask] = ngram_probs[skip_mask]
                total_skipped += int(skip_mask.sum())

            target_b = tokens_bt[:, t]
            for b in range(B):
                cdf = self.cdf_converter.convert(probs_bv[b], self.config.cdf_total)
                encoders[b].encode_symbol(cdf, int(target_b[b]))

            # Advance LLM state with this step's tokens
            _, state = self.runner.step_batch(target_b, state)
            prev_tokens = target_b

        records = [
            ChunkRecord(
                num_tokens=T,
                compressed_bits=enc.get_bit_count(),
                stream=enc.finish(),
            )
            for enc in encoders
        ]
        return records, total_skipped, ngram_state

    def _decode_batch(
        self,
        records: list[ChunkRecord],
        global_offset: int,
        ngram_state: object,
    ) -> "tuple[list[list[int]], int, object]":
        """Decode B chunks in parallel.

        Returns (decoded_ids_per_chunk, n_skipped, updated_ngram_state).
        """
        B = len(records)
        T = records[0].num_tokens
        decoders = [ArithmeticDecoder(r.stream) for r in records]
        state = self.runner.init_batch_state(B)
        prev_tokens = np.full(B, self.runner.bos_token_id, dtype=np.int64)
        decoded: list[list[int]] = [[] for _ in range(B)]
        total_skipped = 0

        for t in range(T):
            skip_mask, ngram_probs, ngram_state = self._compute_skip(
                ngram_state, prev_tokens, global_offset + t
            )

            probs_bv = state.probs  # [B, V]
            if skip_mask is not None:
                probs_bv = probs_bv.copy()
                probs_bv[skip_mask] = ngram_probs[skip_mask]
                total_skipped += int(skip_mask.sum())

            current_tokens = np.empty(B, dtype=np.int64)
            for b in range(B):
                cdf = self.cdf_converter.convert(probs_bv[b], self.config.cdf_total)
                current_tokens[b] = decoders[b].decode_symbol(cdf)
                decoded[b].append(int(current_tokens[b]))

            _, state = self.runner.step_batch(current_tokens, state)
            prev_tokens = current_tokens

        return decoded, total_skipped, ngram_state

    # ------------------------------------------------------------------
    # Public compress / decompress
    # ------------------------------------------------------------------

    def compress(self, text: str) -> "bytes | None":
        """Compress text.

        Returns compressed bytes when ``config.big_file=False``.
        When ``config.big_file=True``, writes directly to ``config.output_path``
        and returns ``None``.
        """
        if self.config.normalized_chunk_len() == 0:
            raise ValueError("chunk_len must be > 0.")
        start_time = time.perf_counter()

        # OOV bypass
        oov_record = OovRecord([], [])
        if self.config.use_oov_bypass:
            split = extract_oov_from_text(text, self.runner.tokenizer.sp)
            token_ids = split.filtered_ids
            if split.positions:
                oov_record = OovRecord(positions=split.positions, chunks=split.chunks)
        else:
            token_ids = self.runner.tokenizer.encode(text)

        num_tokens = len(token_ids)
        if self.verbose:
            print(f"Tokenization: {num_tokens} tokens (OOV bypass: {self.config.use_oov_bypass})")

        if not token_ids:
            self.last_run_stats = CodecRunStats(
                num_tokens=0,
                encode_seconds=time.perf_counter() - start_time,
            )
            return pack_archive(self.config, [], oov_record)

        full_chunks, tail = self._split_tokens(token_ids)
        n_chunks = len(full_chunks) + (1 if tail else 0)
        batch_size = self.config.normalized_batch_size()
        batch_starts = list(range(0, len(full_chunks), batch_size))

        if self.verbose:
            print(
                f"Chunks: {n_chunks} total "
                f"({len(full_chunks)} full, {1 if tail else 0} tail)"
            )

        # Streaming setup
        writer: ArchiveWriter | None = None
        all_records: list[ChunkRecord] = []
        if self.config.big_file:
            writer = ArchiveWriter(self.config.output_path, self.config, n_chunks, oov_record)

        # N-gram initial state (one state shared across all batches — resets per batch in loop)
        ngram_state = (
            self.ngram.init_states(batch_size)
            if self.ngram is not None
            else None
        )

        global_offset = 0
        total_skipped = 0

        for batch_start in self._iter_batches(batch_starts, "Batches"):
            batch = full_chunks[batch_start : batch_start + batch_size]
            B = len(batch)
            tokens_bt = np.array(batch, dtype=np.int64)  # [B, T]

            # Reset n-gram state for this batch (each chunk starts fresh)
            if self.ngram is not None:
                ngram_state = self.ngram.init_states(B)

            records, skipped, _ = self._encode_batch(tokens_bt, global_offset, ngram_state)
            total_skipped += skipped

            if writer is not None:
                for r in records:
                    writer.append(r)
            else:
                all_records.extend(records)

            global_offset += B * self.config.normalized_chunk_len()

        if tail:
            if self.verbose:
                print("Processing tail chunk")
            tokens_bt = np.array([tail], dtype=np.int64)  # [1, tail_len]
            tail_ngram = self.ngram.init_states(1) if self.ngram is not None else None
            records, skipped, _ = self._encode_batch(tokens_bt, global_offset, tail_ngram)
            total_skipped += skipped
            if writer is not None:
                writer.append(records[0])
            else:
                all_records.append(records[0])

        elapsed = time.perf_counter() - start_time
        self.last_run_stats = CodecRunStats(
            num_tokens=num_tokens,
            encode_seconds=elapsed,
            tokens_per_second=(num_tokens / elapsed) if elapsed > 0 else 0.0,
            skipped_tokens=total_skipped,
        )

        if writer is not None:
            writer.close()
            return None

        return pack_archive(self.config, all_records, oov_record)

    def decompress(self, data: bytes) -> str:
        start_time = time.perf_counter()
        meta, records, oov_record = unpack_archive(data)
        config = self._config_from_archive(meta)
        batch_size = max(1, config.batch_size)
        chunk_len = config.chunk_len
        total_skipped = 0
        total_tokens = 0
        all_ids: list[int] = []

        for batch_start in self._iter_batches(
            range(0, len(records), batch_size), "Batches"
        ):
            batch_records = records[batch_start : batch_start + batch_size]
            B = len(batch_records)

            ngram_state = (
                self.ngram.init_states(B) if self.ngram is not None else None
            )

            global_offset = batch_start * chunk_len
            decoded, skipped, _ = self._decode_batch(batch_records, global_offset, ngram_state)
            total_skipped += skipped

            for chunk_ids in decoded:
                all_ids.extend(chunk_ids)
                total_tokens += len(chunk_ids)

        if (meta.flags & FLAG_OOV_BYPASS) and oov_record.positions:
            text = merge_text_from_filtered(
                self.runner.tokenizer.sp,
                all_ids,
                oov_record.positions,
                oov_record.chunks,
            )
        else:
            text = self.runner.tokenizer.decode(all_ids)

        elapsed = time.perf_counter() - start_time
        self.last_run_stats = CodecRunStats(
            num_tokens=total_tokens,
            encode_seconds=elapsed,
            tokens_per_second=(total_tokens / elapsed) if elapsed > 0 else 0.0,
            skipped_tokens=total_skipped,
        )
        return text
