"""Bit-for-bit reimplementation of the dashboard's `mulberry32` PRNG.

This guarantees that the same seed produces the same sample stream in
both the TypeScript dashboard and the Python package — which the thesis
cites as cross-language reproducibility.
"""
from __future__ import annotations

import math
from typing import Iterator


_U32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """Equivalent to JavaScript's Math.imul (32-bit signed multiply)."""
    a &= _U32
    b &= _U32
    # multiply low 32 bits
    product_low = ((a & 0xFFFF) * b + ((((a >> 16) & 0xFFFF) * b) << 16)) & _U32
    # interpret as signed
    if product_low & 0x80000000:
        return product_low - 0x100000000
    return product_low


class Mulberry32:
    """Stateful PRNG matching `src/components/dashboard/rng.ts`."""

    __slots__ = ("_t",)

    def __init__(self, seed: int = 1337) -> None:
        self._t = seed & _U32

    def next(self) -> float:
        self._t = (self._t + 0x6D2B79F5) & _U32
        t = self._t
        r = _imul(t ^ (t >> 15), 1 | t) & _U32
        r = (r + (_imul(r ^ (r >> 7), 61 | r) & _U32)) & _U32
        out = (r ^ (r >> 14)) & _U32
        return out / 4294967296.0

    # Convenience iterator
    def __iter__(self) -> Iterator[float]:  # pragma: no cover - trivial
        while True:
            yield self.next()


def randn(rng: Mulberry32) -> float:
    """Box–Muller standard normal using the Mulberry32 stream."""
    u = 0.0
    v = 0.0
    while u == 0.0:
        u = rng.next()
    while v == 0.0:
        v = rng.next()
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


def normal(rng: Mulberry32, mean: float, sd: float) -> float:
    return mean + sd * randn(rng)
