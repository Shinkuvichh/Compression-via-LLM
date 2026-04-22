"""Minimal CLI for ZoloZip."""

import argparse
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from codec import ZoloArithmeticCodec
from config import CONFIG
from models import MambaModelRunner
from utils import format_size


def _resolve_path(path: str) -> str:
    """Paths in CONFIG / argv are relative to the ZoloZip project directory, not cwd."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_PROJECT_ROOT, path))


def _normalize_mode(mode: str) -> str:
    m = mode.strip().lower()
    if m in ("compress", "encode", "кодирование"):
        return "compress"
    if m in ("decompress", "decode", "декодирование"):
        return "decompress"
    raise ValueError("mode must be compress/decompress")


def run_compress(input_path: str, output_path: str):
    input_path = _resolve_path(input_path)
    output_path = _resolve_path(output_path)
    if not output_path.lower().endswith(".zzz"):
        output_path = f"{output_path}.zzz"
    if CONFIG.big_file:
        CONFIG.output_path = output_path
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    runner = MambaModelRunner(CONFIG)
    codec = ZoloArithmeticCodec(CONFIG, runner)

    start = time.time()
    data = codec.compress(text)
    elapsed = time.time() - start

    if CONFIG.big_file:
        compressed_size = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
    else:
        with open(output_path, "wb") as f:
            f.write(data)
        compressed_size = len(data)

    original_size = len(text.encode("utf-8"))
    ratio = compressed_size / original_size if original_size else 0.0
    stats = codec.last_run_stats
    print(f"Original: {format_size(original_size)}")
    print(f"Compressed: {format_size(compressed_size)}")
    print(f"Ratio: {ratio:.4f} ({ratio * 100:.1f}%)")
    print(f"Total encode time: {stats.encode_seconds:.2f}s")
    print(f"Encode speed: {stats.tokens_per_second:.1f} tok/s ({stats.num_tokens} tokens)")
    if stats.num_tokens > 0:
        sp = 100.0 * stats.skipped_tokens / stats.num_tokens
        print(
            f"LLM-skip tokens: {stats.skipped_tokens} / {stats.num_tokens} ({sp:.1f}%)"
        )
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Output: {os.path.abspath(output_path)}")


def run_decompress(input_path: str, output_path: str):
    input_path = _resolve_path(input_path)
    output_path = _resolve_path(output_path)
    with open(input_path, "rb") as f:
        data = f.read()
    runner = MambaModelRunner(CONFIG)
    codec = ZoloArithmeticCodec(CONFIG, runner)

    start = time.time()
    text = codec.decompress(data)
    elapsed = time.time() - start

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    stats = codec.last_run_stats
    print(f"Decompressed: {format_size(len(text.encode('utf-8')))}")
    print(f"Time: {elapsed:.2f}s")
    if stats.num_tokens > 0:
        sp = 100.0 * stats.skipped_tokens / stats.num_tokens
        print(
            f"LLM-skip tokens (decode): {stats.skipped_tokens} / {stats.num_tokens} ({sp:.1f}%)"
        )
    print(f"Output: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="ZoloZip. With no arguments, uses CONFIG.mode, CONFIG.input_path, CONFIG.output_path "
        "(paths relative to the ZoloZip folder, not the shell cwd).",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default=None,
        help="compress | decompress (default: CONFIG.mode)",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="input file (default: CONFIG.input_path)",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="output file (default: CONFIG.output_path)",
    )
    args = parser.parse_args()

    mode_raw = args.mode if args.mode is not None else CONFIG.mode
    inp = args.input_path if args.input_path is not None else CONFIG.input_path
    out = args.output_path if args.output_path is not None else CONFIG.output_path

    if not inp or not out:
        parser.error("input_path and output_path must be set (CLI or CONFIG).")

    mode = _normalize_mode(mode_raw)
    if mode == "compress":
        run_compress(inp, out)
        return
    run_decompress(inp, out)


if __name__ == "__main__":
    main()
