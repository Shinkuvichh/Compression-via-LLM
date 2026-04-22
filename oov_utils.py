"""Extract and merge OOV (unk) tokens for SPM + raw UTF-8 side channel (L3TC-style).

L3TC approach: sp.encode(text, out_type='immutable_proto') gives pieces with .id and
.surface (exact text span, no re-decoding needed). O(n) overall.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OovSplit:
    """Result of splitting token stream for LM codec."""

    positions: list[int]
    chunks: list[bytes]
    filtered_ids: list[int]


def extract_oov_from_text(text: str, sp) -> OovSplit:
    """Find unk token positions, UTF-8 bytes per unk, and ids without unk. O(n)."""
    unk_id = sp.unk_id()
    proto = sp.encode(text, out_type="immutable_proto")
    pieces = [p for p in proto.pieces if p.begin != p.end]

    if unk_id < 0:
        return OovSplit([], [], [p.id for p in pieces])

    positions: list[int] = []
    chunks: list[bytes] = []
    filtered_ids: list[int] = []

    for i, piece in enumerate(pieces):
        if piece.id == unk_id:
            positions.append(i)
            chunks.append(piece.surface.encode("utf-8"))
        else:
            filtered_ids.append(piece.id)

    return OovSplit(positions=positions, chunks=chunks, filtered_ids=filtered_ids)


def merge_text_from_filtered(
    sp, filtered_ids: list[int], positions: list[int], chunks: list[bytes]
) -> str:
    """Reconstruct UTF-8 text from LM token ids (no unk) + OOV side channel.

    Decodes only the non-OOV runs in one sp.decode call each; splices OOV bytes
    back by position. O(n) total decode work.
    """
    if len(positions) != len(chunks):
        raise ValueError("positions and chunks length mismatch.")
    if not positions:
        return sp.decode(filtered_ids)

    oov_set = set(positions)
    oov_by_pos = dict(zip(positions, chunks))
    n_total = len(filtered_ids) + len(positions)

    it = iter(filtered_ids)
    parts: list[str] = []
    run: list[int] = []

    def _flush_run() -> None:
        if run:
            parts.append(sp.decode(run))
            run.clear()

    for pos in range(n_total):
        if pos in oov_set:
            _flush_run()
            parts.append(oov_by_pos[pos].decode("utf-8"))
        else:
            run.append(next(it))
    _flush_run()

    return "".join(parts)
