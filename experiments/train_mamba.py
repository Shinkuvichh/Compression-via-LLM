import os

import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# paths
DATA_PATH      = "enwik8"
CHECKPOINT_DIR = "mamba_checkpoints"
TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer.model")

# tokenizer
VOCAB_SIZE = 16384
COVERAGE   = 0.999
DROP_UNK_TOKENS = True

# model
D_MODEL        = 128
N_LAYER        = 3
D_INTERMEDIATE = 0
SSM_CFG = {
    "layer":   "Mamba2",
    "d_state":  64,
    "d_conv":   4,
    "expand":   2,
    "headdim":  32,
    "ngroups":  1,
}
RMS_NORM         = True
RESIDUAL_IN_FP32 = True
FUSED_ADD_NORM   = True
TIE_EMBEDDINGS   = False

# training
BLOCK_SIZE     = 2048
BATCH_SIZE     = 256
EPOCHS         = 20
LR             = 5e-4
# One cosine cycle over full training (scheduler.step once per epoch).
COSINE_T_MAX   = EPOCHS
COSINE_ETA_MIN = 1e-6
WEIGHT_DECAY   = 0.01
GRAD_CLIP      = 1.0
GRAD_ACCUM     = 1

# eval
VAL_FRACTION  = 0.05
SAVE_INTERVAL = 2000

# misc
SEED        = 67
AMP_DTYPE   = torch.bfloat16
NUM_WORKERS = 0


# tokenizer

def _train_tokenizer():
    if os.path.exists(TOKENIZER_PATH):
        print(f"Tokenizer found: {TOKENIZER_PATH}")
        return
    print("Training tokenizer...")
    spm.SentencePieceTrainer.train(
        input=DATA_PATH,
        model_prefix=os.path.splitext(TOKENIZER_PATH)[0],
        model_type="bpe",
        vocab_size=VOCAB_SIZE,
        character_coverage=COVERAGE,
        byte_fallback=False,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        train_extremely_large_corpus=True,
    )
    print(f"Tokenizer saved: {TOKENIZER_PATH}")


def _load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    return sp


# dataset

class TokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens     = tokens
        self.block_size = block_size
        self.n_samples  = (len(tokens) - 1) // block_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.tokens[start : start + self.block_size + 1]
        return chunk[:-1].long(), chunk[1:].long()


def _build_datasets(sp):
    print("Reading data...")
    with open(DATA_PATH, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    print("Tokenizing...")
    tokens = torch.tensor(sp.encode(text, out_type=int), dtype=torch.int32)
    unk_id = sp.unk_id()
    if unk_id >= 0:
        unk_count = int((tokens == unk_id).sum().item())
        if DROP_UNK_TOKENS and unk_count > 0:
            tokens = tokens[tokens != unk_id]
        print(f"UNK tokens: {unk_count:,}")
    max_id = int(tokens.max().item()) if len(tokens) > 0 else -1
    sp_vocab = len(sp)
    if max_id >= sp_vocab:
        raise ValueError(f"Token id {max_id} is out of tokenizer vocab size {sp_vocab}")
    print(f"Total tokens: {len(tokens):,}")
    split = int(len(tokens) * (1.0 - VAL_FRACTION))
    return TokenDataset(tokens[:split], BLOCK_SIZE), TokenDataset(tokens[split:], BLOCK_SIZE)


# eval

@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    criterion    = nn.CrossEntropyLoss(reduction="sum")
    total_loss   = 0.0
    total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            logits = model(x).logits
            B, T, V = logits.shape
            total_loss   += criterion(logits.reshape(B * T, V), y.reshape(B * T)).item()
            total_tokens += B * T
    model.train()
    return total_loss / total_tokens


# main

if __name__ == "__main__":
    torch.manual_seed(SEED)
    device = torch.device("cuda")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    _train_tokenizer()
    sp = _load_tokenizer()
    real_vocab_size = len(sp)
    print(f"Tokenizer vocab size: {real_vocab_size}")
    if real_vocab_size != VOCAB_SIZE:
        print(f"Requested vocab: {VOCAB_SIZE}, using actual vocab: {real_vocab_size}")

    train_ds, val_ds = _build_datasets(sp)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )

    # model
    cfg_dict = {
        "d_model": D_MODEL,
        "n_layer": N_LAYER,
        "d_intermediate": D_INTERMEDIATE,
        "vocab_size": real_vocab_size,
        "ssm_cfg": SSM_CFG,
        "rms_norm": RMS_NORM,
        "residual_in_fp32": RESIDUAL_IN_FP32,
        "fused_add_norm": FUSED_ADD_NORM,
        "tie_embeddings": TIE_EMBEDDINGS,
        "pad_vocab_size_multiple": 8,
    }
    cfg = MambaConfig(**cfg_dict)
    model = MambaLMHeadModel(cfg, device="cuda", dtype=AMP_DTYPE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,} ({n_params/1e6:.2f}M)")

    # optimizer
    decay_params    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LR, betas=(0.9, 0.999), eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=COSINE_T_MAX, eta_min=COSINE_ETA_MIN
    )

    criterion     = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    global_step   = 0

    model.train()

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

        for micro_step, (x, y) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                logits = model(x).logits
                B, T, V = logits.shape
                loss = criterion(logits.reshape(B * T, V), y.reshape(B * T)) / GRAD_ACCUM

            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            is_accum_step = (micro_step + 1) % GRAD_ACCUM == 0
            is_last_batch = (micro_step + 1) == len(train_loader)

            if is_accum_step or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % SAVE_INTERVAL == 0:
                    ckpt = os.path.join(CHECKPOINT_DIR, f"step_{global_step:07d}.pt")
                    torch.save({
                        "epoch": epoch, "global_step": global_step,
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "val_loss": best_val_loss, "cfg": cfg_dict,
                    }, ckpt)

        avg_loss = epoch_loss / len(train_loader)

        torch.cuda.empty_cache()
        val_loss = _evaluate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch, "global_step": global_step,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "val_loss": val_loss, "cfg": cfg_dict,
            }, os.path.join(CHECKPOINT_DIR, "best.pt"))

        print(f"Epoch: {epoch}/{EPOCHS}  Loss: {avg_loss:.4f}  Val loss: {val_loss:.4f}")

        scheduler.step()

    final_val_loss = _evaluate(model, val_loader, device)
    print(f"Final Val loss: {final_val_loss:.4f}")
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        torch.save({
            "epoch": EPOCHS, "global_step": global_step,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "val_loss": final_val_loss, "cfg": cfg_dict,
        }, os.path.join(CHECKPOINT_DIR, "best.pt"))

    torch.save({
        "epoch": EPOCHS, "global_step": global_step,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "val_loss": final_val_loss, "cfg": cfg_dict,
    }, os.path.join(CHECKPOINT_DIR, "last.pt"))

    print(f"Done. Best val loss: {best_val_loss:.4f}")
