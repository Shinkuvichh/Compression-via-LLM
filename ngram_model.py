"""Token-level N-gram model reused from Nacrith."""

import numpy as np


def _context_hash(context_tokens, order):
    h = 0
    end = len(context_tokens)
    for i in range(end - order, end):
        h = (h * 49157 + context_tokens[i]) & 0xFFFFFFFFFFFFFFFF
    return h


class NgramModel:
    ESCAPE = 5
    MAX_TABLE_ENTRIES = 500_000
    MAX_INNER_ENTRIES = 64

    def __init__(self, max_order: int = 4, vocab_size: int = 49152):
        self.max_order = max_order
        self.vocab_size = vocab_size
        self.unigram_counts = np.zeros(vocab_size, dtype=np.float64)
        self.total_unigram = 0
        self._slot_map = [None] + [dict() for _ in range(max_order)]
        self._inner_ids = [None] + [
            np.empty((self.MAX_TABLE_ENTRIES, self.MAX_INNER_ENTRIES), dtype=np.int32)
            for _ in range(max_order)
        ]
        self._inner_counts = [None] + [
            np.empty((self.MAX_TABLE_ENTRIES, self.MAX_INNER_ENTRIES), dtype=np.int32)
            for _ in range(max_order)
        ]
        self._inner_sizes = [None] + [
            np.zeros(self.MAX_TABLE_ENTRIES, dtype=np.int16) for _ in range(max_order)
        ]
        self._ctx_totals = [None] + [
            np.zeros(self.MAX_TABLE_ENTRIES, dtype=np.int32) for _ in range(max_order)
        ]
        self._next_slot = [0] * (max_order + 1)
        self._free_slots = [None] + [[] for _ in range(max_order)]
        self._buf = np.zeros(vocab_size, dtype=np.float64)
        self._probs = np.zeros(vocab_size, dtype=np.float64)

    def reset(self):
        self.unigram_counts[:] = 0
        self.total_unigram = 0
        self._slot_map = [None] + [dict() for _ in range(self.max_order)]
        self._next_slot = [0] * (self.max_order + 1)
        self._free_slots = [None] + [[] for _ in range(self.max_order)]
        for order in range(1, self.max_order + 1):
            self._inner_sizes[order][:] = 0
            self._ctx_totals[order][:] = 0

    def predict(self, context_tokens: list[int]) -> np.ndarray:
        probs = self._probs
        np.add(self.unigram_counts, 1.0, out=probs)
        probs /= (self.total_unigram + self.vocab_size)

        for order in range(1, self.max_order + 1):
            if len(context_tokens) < order:
                break

            ctx = _context_hash(context_tokens, order)
            slot = self._slot_map[order].get(ctx)
            if slot is None:
                continue

            total = int(self._ctx_totals[order][slot])
            if total == 0:
                continue

            lam = total / (total + self.ESCAPE)
            buf = self._buf
            buf[:] = 0
            size = int(self._inner_sizes[order][slot])
            ids = self._inner_ids[order][slot, :size]
            cts = self._inner_counts[order][slot, :size]
            buf[ids] = cts
            buf /= buf.sum()

            probs *= (1.0 - lam)
            buf *= lam
            probs += buf

        return probs

    def _alloc_slot(self, order: int) -> int:
        if self._free_slots[order]:
            return self._free_slots[order].pop()
        slot = self._next_slot[order]
        self._next_slot[order] += 1
        return slot

    def update(self, context_tokens: list[int], actual_token: int):
        self.unigram_counts[actual_token] += 1
        self.total_unigram += 1

        for order in range(1, self.max_order + 1):
            if len(context_tokens) < order:
                break

            ctx = _context_hash(context_tokens, order)
            slot_map = self._slot_map[order]

            if ctx not in slot_map and len(slot_map) >= self.MAX_TABLE_ENTRIES:
                evict_ctx = next(iter(slot_map))
                evict_slot = slot_map.pop(evict_ctx)
                self._free_slots[order].append(evict_slot)

            if ctx in slot_map:
                slot = slot_map[ctx]
                size = int(self._inner_sizes[order][slot])
                ids = self._inner_ids[order][slot]
                counts = self._inner_counts[order][slot]

                mask = ids[:size] == actual_token
                if mask.any():
                    idx = int(np.argmax(mask))
                    counts[idx] += 1
                    self._ctx_totals[order][slot] += 1
                elif size < self.MAX_INNER_ENTRIES:
                    ids[size] = actual_token
                    counts[size] = 1
                    self._inner_sizes[order][slot] = size + 1
                    self._ctx_totals[order][slot] += 1
                else:
                    min_count = int(counts[:size].min())
                    if min_count == 1:
                        min_idx = int(np.argmin(counts[:size]))
                        if min_idx < size - 1:
                            ids[min_idx:size - 1] = ids[min_idx + 1:size]
                            counts[min_idx:size - 1] = counts[min_idx + 1:size]
                        ids[size - 1] = actual_token
                        counts[size - 1] = 1
            else:
                slot = self._alloc_slot(order)
                slot_map[ctx] = slot
                self._inner_ids[order][slot, 0] = actual_token
                self._inner_counts[order][slot, 0] = 1
                self._inner_sizes[order][slot] = 1
                self._ctx_totals[order][slot] = 1
