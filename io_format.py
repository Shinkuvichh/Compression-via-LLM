"""Minimal text-only archive format for ZoloZip.

Format NCX2 (incompatible with NCX1):
  [HEADER  21 bytes]
  [ENTRY TABLE  n_chunks * 12 bytes]  -- zeroed at open, filled at close for streaming
  [CHUNK STREAMS  variable]
  [OOV TAIL]
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

try:
    from config import CodecConfig
except ImportError:
    from .config import CodecConfig


MAGIC = b"NCX2"

FLAG_CONFIDENCE_SKIP = 0x08
FLAG_OOV_BYPASS = 0x10

# magic(4s) flags(B) chunk_len(I) batch_size(I) ngram_order(H) skip_thr(H) n_chunks(I)
HEADER_STRUCT = struct.Struct(">4sBIIHHI")
ENTRY_STRUCT = struct.Struct(">III")   # num_tokens(I) compressed_bits(I) stream_len(I)


@dataclass(slots=True)
class ChunkRecord:
    num_tokens: int
    compressed_bits: int
    stream: bytes


@dataclass(slots=True)
class OovRecord:
    """Unk token positions (in original SPM id sequence) and raw UTF-8 bytes each."""
    positions: list[int]
    chunks: list[bytes]


@dataclass(slots=True)
class ArchiveMetadata:
    flags: int
    chunk_len: int
    batch_size: int
    ngram_order: int
    skip_threshold: float
    n_chunks: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def config_flags(config: CodecConfig) -> int:
    flags = 0
    if config.use_confidence_skip:
        flags |= FLAG_CONFIDENCE_SKIP
    if config.use_oov_bypass:
        flags |= FLAG_OOV_BYPASS
    return flags


def _pack_header(config: CodecConfig, n_chunks: int) -> bytes:
    flags = config_flags(config)
    return HEADER_STRUCT.pack(
        MAGIC,
        flags,
        int(config.normalized_chunk_len()),
        int(config.normalized_batch_size()),
        int(config.ngram_order),
        int(round(config.skip_threshold * 1000)),
        n_chunks,
    )


def _pack_oov_tail(oov: OovRecord) -> bytes:
    n = len(oov.positions)
    out = bytearray(struct.pack(">I", n))
    if n == 0:
        return bytes(out)
    out.extend(struct.pack(f">{n}I", *oov.positions))
    lens = [len(b) for b in oov.chunks]
    if len(lens) != n:
        raise ValueError("OovRecord positions/chunks length mismatch.")
    out.extend(struct.pack(f">{n}I", *lens))
    out.extend(b"".join(oov.chunks))
    return bytes(out)


def _unpack_oov_tail(data: bytes, offset: int) -> tuple[OovRecord, int]:
    if offset + 4 > len(data):
        return OovRecord([], []), offset
    n = struct.unpack_from(">I", data, offset)[0]
    offset += 4
    if n == 0:
        return OovRecord([], []), offset
    if offset + 8 * n > len(data):
        raise ValueError("OOV section truncated (positions+lengths).")
    positions = list(struct.unpack_from(f">{n}I", data, offset))
    offset += 4 * n
    lens = list(struct.unpack_from(f">{n}I", data, offset))
    offset += 4 * n
    chunks: list[bytes] = []
    for L in lens:
        if offset + L > len(data):
            raise ValueError("OOV section truncated (payload).")
        chunks.append(bytes(data[offset : offset + L]))
        offset += L
    return OovRecord(positions=positions, chunks=chunks), offset


# ---------------------------------------------------------------------------
# In-memory pack / unpack (big_file=False)
# ---------------------------------------------------------------------------

def pack_archive(
    config: CodecConfig,
    chunk_records: list[ChunkRecord],
    oov_record: OovRecord | None = None,
) -> bytes:
    oov = oov_record if oov_record is not None else OovRecord([], [])
    header = _pack_header(config, len(chunk_records))
    table = bytearray()
    streams = bytearray()
    for record in chunk_records:
        table += ENTRY_STRUCT.pack(
            int(record.num_tokens),
            int(record.compressed_bits),
            len(record.stream),
        )
        streams += record.stream
    tail = _pack_oov_tail(oov)
    return header + bytes(table) + bytes(streams) + tail


def unpack_archive(data: bytes) -> tuple[ArchiveMetadata, list[ChunkRecord], OovRecord]:
    header_size = HEADER_STRUCT.size
    if len(data) < header_size:
        raise ValueError("Archive too short.")

    magic, flags, chunk_len, batch_size, ngram_order, skip_thr_enc, n_chunks = (
        HEADER_STRUCT.unpack_from(data, 0)
    )

    if magic != MAGIC:
        raise ValueError(f"Unsupported archive magic: {magic!r} (expected {MAGIC!r})")

    meta = ArchiveMetadata(
        flags=flags,
        chunk_len=chunk_len,
        batch_size=batch_size,
        ngram_order=ngram_order,
        skip_threshold=skip_thr_enc / 1000.0,
        n_chunks=n_chunks,
    )

    entries_offset = header_size
    streams_offset = header_size + n_chunks * ENTRY_STRUCT.size
    records: list[ChunkRecord] = []
    cursor = streams_offset
    for i in range(n_chunks):
        entry_off = entries_offset + i * ENTRY_STRUCT.size
        num_tokens, compressed_bits, stream_length = ENTRY_STRUCT.unpack_from(data, entry_off)
        stream = data[cursor : cursor + stream_length]
        if len(stream) != stream_length:
            raise ValueError("Archive stream table is truncated.")
        records.append(ChunkRecord(
            num_tokens=num_tokens,
            compressed_bits=compressed_bits,
            stream=stream,
        ))
        cursor += stream_length

    if cursor + 4 > len(data):
        raise ValueError("Archive missing OOV tail (need at least 4 bytes).")
    oov_record, cursor = _unpack_oov_tail(data, cursor)
    if cursor != len(data):
        raise ValueError(f"Trailing bytes after OOV section: {len(data) - cursor}.")

    return meta, records, oov_record


# ---------------------------------------------------------------------------
# Streaming writer (big_file=True)
# ---------------------------------------------------------------------------

class ArchiveWriter:
    """Write chunk streams directly to disk; only tiny entry metadata buffered in RAM.

    Usage:
        w = ArchiveWriter(path, config, n_chunks, oov_record)
        for record in records:
            w.append(record)
        w.close()
    """

    def __init__(
        self,
        path: str,
        config: CodecConfig,
        n_chunks: int,
        oov_record: OovRecord,
    ) -> None:
        self._f = open(path, "wb")
        self._f.write(_pack_header(config, n_chunks))
        self._entry_table_offset = self._f.tell()
        # Reserve space for entry table (filled at close)
        self._f.write(b"\x00" * (n_chunks * ENTRY_STRUCT.size))
        self._oov = oov_record
        self._entries: list[tuple[int, int, int]] = []  # (num_tokens, compressed_bits, stream_len)

    def append(self, record: ChunkRecord) -> None:
        self._entries.append((record.num_tokens, record.compressed_bits, len(record.stream)))
        self._f.write(record.stream)

    def close(self) -> None:
        # Write OOV tail at end
        self._f.write(_pack_oov_tail(self._oov))
        # Seek back and fill entry table
        self._f.seek(self._entry_table_offset)
        for num_tokens, compressed_bits, stream_len in self._entries:
            self._f.write(ENTRY_STRUCT.pack(num_tokens, compressed_bits, stream_len))
        self._f.close()
