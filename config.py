from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CodecConfig:
    # Pipeline mode: "compress" or "decompress"
    mode: str = "compress"
    input_path: str = "data/enwik82"
    output_path: str = "data/enwik82.zzz"

    # Model/runtime (local train_mamba checkpoint + SentencePiece)
    checkpoint_path: str = "models/mamba8m.pt"
    tokenizer_path: str = "models/tokenizer.model"
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float32
    use_int8: bool = False
    seed: int = 67
    verbose: bool = True

    # Skip unk tokens in LM path; store raw UTF-8 + positions (L3TC-style)
    use_oov_bypass: bool = True

    chunk_len: int = 2048
    batch_size: int = 512

    # use_confidence_skip=True → n-gram model is used; low-entropy positions skip LLM
    use_confidence_skip: bool = False
    skip_threshold: float = 2.6  # bits; entropy below this → skip LLM
    warmup_tokens: int = 200     # first N global tokens always use LLM (no skip)
    ngram_order: int = 4         # passed to n-gram model constructor

    cdf_total: int = 1 << 24
    bos_token_id: int = 2        # train_mamba SentencePiece bos_id

    # If True, chunk streams are written directly to output_path during compress
    # (reduces peak RAM for large files). compress() returns None in this mode.
    big_file: bool = False

    def normalized_chunk_len(self) -> int:
        return max(1, int(self.chunk_len))

    def normalized_batch_size(self) -> int:
        return max(1, int(self.batch_size))


# Edit this object manually before launch.
CONFIG = CodecConfig()
