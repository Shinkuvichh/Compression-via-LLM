"""Arithmetic coding: probability → integer CDF → bitstream."""

import numpy as np

MIN_CDF_COUNT = 1


class CdfConverter:
    """Float probability vector → integer cumulative distribution for the coder."""

    __slots__ = ("_n", "_float_buf", "_counts", "_cdf")

    def __init__(self, vocab_size: int):
        self._n = vocab_size
        self._float_buf = np.zeros(vocab_size, dtype=np.float64)
        self._counts = np.zeros(vocab_size, dtype=np.int64)
        self._cdf = np.zeros(vocab_size + 1, dtype=np.int64)

    def convert(self, probs: np.ndarray, total: int) -> np.ndarray:
        n = self._n
        scale = total - n * MIN_CDF_COUNT

        np.multiply(probs, scale, out=self._float_buf)
        self._counts[:] = self._float_buf
        np.clip(self._counts, 0, None, out=self._counts)
        self._counts += MIN_CDF_COUNT

        diff = total - int(self._counts.sum())
        if diff != 0:
            self._counts[int(self._counts.argmax())] += diff

        self._cdf[0] = 0
        np.cumsum(self._counts, out=self._cdf[1:])
        return self._cdf


class ArithmeticEncoder:
    PRECISION = 32
    FULL = 1 << PRECISION
    HALF = 1 << (PRECISION - 1)
    QUARTER = 1 << (PRECISION - 2)
    MAX_RANGE = FULL - 1

    def __init__(self):
        self.low = 0
        self.high = self.MAX_RANGE
        self.pending_bits = 0
        self._buf = bytearray()
        self._cur_byte = 0
        self._bits_in_cur = 0
        self._total_bits = 0

    def _write_bit(self, bit: int):
        self._cur_byte = (self._cur_byte << 1) | bit
        self._bits_in_cur += 1
        self._total_bits += 1
        if self._bits_in_cur == 8:
            self._buf.append(self._cur_byte)
            self._cur_byte = 0
            self._bits_in_cur = 0

    def _output_bit(self, bit: int):
        self._write_bit(bit)
        for _ in range(self.pending_bits):
            self._write_bit(1 - bit)
        self.pending_bits = 0

    def encode_symbol(self, cdf, symbol_index: int):
        total = int(cdf[-1])
        rng = self.high - self.low + 1
        sym_lo = int(cdf[symbol_index])
        sym_hi = int(cdf[symbol_index + 1])

        self.high = self.low + (rng * sym_hi) // total - 1
        self.low = self.low + (rng * sym_lo) // total

        while True:
            if self.high < self.HALF:
                self._output_bit(0)
                self.low = self.low << 1
                self.high = (self.high << 1) | 1
            elif self.low >= self.HALF:
                self._output_bit(1)
                self.low = (self.low - self.HALF) << 1
                self.high = ((self.high - self.HALF) << 1) | 1
            elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                self.pending_bits += 1
                self.low = (self.low - self.QUARTER) << 1
                self.high = ((self.high - self.QUARTER) << 1) | 1
            else:
                break

        self.low &= self.MAX_RANGE
        self.high &= self.MAX_RANGE

    def finish(self) -> bytes:
        self.pending_bits += 1
        if self.low < self.QUARTER:
            self._output_bit(0)
        else:
            self._output_bit(1)

        while self._bits_in_cur != 0:
            self._write_bit(0)

        return bytes(self._buf)

    def get_bit_count(self) -> int:
        return self._total_bits + self.pending_bits


class ArithmeticDecoder:
    PRECISION = 32
    FULL = 1 << PRECISION
    HALF = 1 << (PRECISION - 1)
    QUARTER = 1 << (PRECISION - 2)
    MAX_RANGE = FULL - 1

    def __init__(self, data: bytes):
        self._data = data
        self._byte_pos = 0
        self._bit_buf = 0
        self._bits_left = 0
        self.low = 0
        self.high = self.MAX_RANGE
        self.value = 0
        for _ in range(self.PRECISION):
            self.value = (self.value << 1) | self._read_bit()

    def _read_bit(self) -> int:
        if self._bits_left == 0:
            if self._byte_pos < len(self._data):
                self._bit_buf = self._data[self._byte_pos]
                self._byte_pos += 1
                self._bits_left = 8
            else:
                return 0
        self._bits_left -= 1
        return (self._bit_buf >> self._bits_left) & 1

    def decode_symbol(self, cdf) -> int:
        total = int(cdf[-1])
        rng = self.high - self.low + 1
        scaled_value = ((self.value - self.low + 1) * total - 1) // rng

        num_symbols = len(cdf) - 1
        lo, hi = 0, num_symbols - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if int(cdf[mid + 1]) <= scaled_value:
                lo = mid + 1
            else:
                hi = mid - 1
        symbol = lo

        sym_lo = int(cdf[symbol])
        sym_hi = int(cdf[symbol + 1])
        self.high = self.low + (rng * sym_hi) // total - 1
        self.low = self.low + (rng * sym_lo) // total

        while True:
            if self.high < self.HALF:
                self.low = self.low << 1
                self.high = (self.high << 1) | 1
                self.value = (self.value << 1) | self._read_bit()
            elif self.low >= self.HALF:
                self.low = (self.low - self.HALF) << 1
                self.high = ((self.high - self.HALF) << 1) | 1
                self.value = ((self.value - self.HALF) << 1) | self._read_bit()
            elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                self.low = (self.low - self.QUARTER) << 1
                self.high = ((self.high - self.QUARTER) << 1) | 1
                self.value = ((self.value - self.QUARTER) << 1) | self._read_bit()
            else:
                break

        self.low &= self.MAX_RANGE
        self.high &= self.MAX_RANGE
        self.value &= self.MAX_RANGE
        return symbol
