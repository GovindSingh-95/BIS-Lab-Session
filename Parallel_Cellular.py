"""
Parallel Cellular Automata in Python
-----------------------------------
Two fast update kernels using Numba:
  1) Conway's Game of Life (binary states {0,1})
  2) Generic outer-totalistic binary CA via (S/B) rule masks, e.g. Life = S23/B3

Features
- 2D grid, Moore neighborhood (8 neighbors)
- Optional toroidal wraparound or fixed (zero) boundary
- prange-parallelized update
- Vectorized seed helpers and a tiny demo (no GUI)

Requires: numpy, numba
Install:  pip install numpy numba
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numba import njit, prange


# ------------------------------
# Utilities
# ------------------------------

def rule_masks_from_string(rule: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse outer-totalistic rule like "S23/B3" into boolean masks of length 10.
    mask_survive[c] == 1 if a live cell with c neighbors survives.
    mask_born[c]    == 1 if a dead cell with c neighbors becomes alive.
    """
    rule = rule.upper().replace(" ", "")
    s_part, b_part = "", ""
    for part in rule.split("/"):
        if part.startswith("S"):
            s_part = part[1:]
        elif part.startswith("B"):
            b_part = part[1:]
    survive = np.zeros(10, dtype=np.uint8)
    born = np.zeros(10, dtype=np.uint8)
    for ch in s_part:
        if ch.isdigit():
            survive[int(ch)] = 1
    for ch in b_part:
        if ch.isdigit():
            born[int(ch)] = 1
    return survive, born


# ------------------------------
# Numba-parallel kernels
# ------------------------------

@njit(parallel=True, fastmath=True)
def _neighbor_count_wrap(grid: np.ndarray) -> np.ndarray:
    """Count Moore neighbors with toroidal wrapping."""
    h, w = grid.shape
    out = np.zeros_like(grid, dtype=np.uint8)
    for y in prange(h):
        ym = (y - 1 + h) % h
        yp = (y + 1) % h
        for x in range(w):
            xm = (x - 1 + w) % w
            xp = (x + 1) % w
            n = (
                grid[ym, xm] + grid[ym, x] + grid[ym, xp]
                + grid[y, xm]              + grid[y, xp]
                + grid[yp, xm] + grid[yp, x] + grid[yp, xp]
            )
            out[y, x] = n
    return out


@njit(parallel=True, fastmath=True)
def _neighbor_count_clamp(grid: np.ndarray) -> np.ndarray:
    """Count Moore neighbors with zero-padded (non-wrapping) boundary."""
    h, w = grid.shape
    out = np.zeros_like(grid, dtype=np.uint8)
    for y in prange(h):
        y0 = max(0, y - 1)
        y1 = min(h - 1, y + 1)
        for x in range(w):
            x0 = max(0, x - 1)
            x1 = min(w - 1, x + 1)
            total = 0
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    total += grid[yy, xx]
            total -= grid[y, x]
            out[y, x] = total
    return out


@njit(parallel=True, fastmath=True)
def life_step(grid: np.ndarray, wrap: bool = True) -> np.ndarray:
    """One parallel step of Conway's Game of Life on a binary grid (uint8 0/1).

    Returns a *new* array; does not modify input.
    """
    neighbors = _neighbor_count_wrap(grid) if wrap else _neighbor_count_clamp(grid)
    h, w = grid.shape
    next_grid = np.empty_like(grid)
    for y in prange(h):
        for x in range(w):
            n = neighbors[y, x]
            if grid[y, x] == 1:
                next_grid[y, x] = 1 if (n == 2 or n == 3) else 0
            else:
                next_grid[y, x] = 1 if (n == 3) else 0
    return next_grid


@njit(parallel=True, fastmath=True)
def outer_totalistic_step(
    grid: np.ndarray,
    survive_mask: np.ndarray,
    born_mask: np.ndarray,
    wrap: bool = True,
) -> np.ndarray:
    """One parallel step of an outer-totalistic binary CA defined by masks.

    `survive_mask[c] = 1` if live cell with c neighbors survives.
    `born_mask[c]    = 1` if dead cell with c neighbors becomes alive.
    """
    neighbors = _neighbor_count_wrap(grid) if wrap else _neighbor_count_clamp(grid)
    h, w = grid.shape
    next_grid = np.empty_like(grid)
    for y in prange(h):
        for x in range(w):
            n = neighbors[y, x]
            alive = grid[y, x]
            # n is 0..8; masks length is 10 for safety
            if alive == 1:
                next_grid[y, x] = survive_mask[n]
            else:
                next_grid[y, x] = born_mask[n]
    return next_grid


# ------------------------------
# High-level runner
# ------------------------------

@dataclass
class CARunner:
    grid: np.ndarray  # dtype uint8, values 0/1
    wrap: bool = True

    def step_life(self) -> None:
        self.grid = life_step(self.grid, self.wrap)

    def step_rule(self, rule: str) -> None:
        s, b = rule_masks_from_string(rule)
        self.grid = outer_totalistic_step(self.grid, s, b, self.wrap)

    def run(self, steps: int, rule: str = "S23/B3") -> None:
        s, b = rule_masks_from_string(rule)
        for _ in range(steps):
            self.grid = outer_totalistic_step(self.grid, s, b, self.wrap)


# ------------------------------
# Seeding helpers
# ------------------------------

def random_seed(h: int, w: int, p: float = 0.5, *, dtype=np.uint8) -> np.ndarray:
    return (np.random.rand(h, w) < p).astype(dtype)


def insert_glider(grid: np.ndarray, top: int = 1, left: int = 1) -> None:
    pattern = np.array([[0,1,0],[0,0,1],[1,1,1]], dtype=np.uint8)
    h, w = pattern.shape
    grid[top:top+h, left:left+w] = pattern


# ------------------------------
# Demo & simple benchmark
# ------------------------------

if __name__ == "__main__":
    H, W = 2048, 2048  # adjust to taste
    grid = random_seed(H, W, p=0.25)
    runner = CARunner(grid, wrap=True)

    # Warm-up to compile Numba
    runner.step_life()

    t0 = time.perf_counter()
    for _ in range(20):
        runner.step_life()
    t1 = time.perf_counter()

    gen_per_sec = 20.0 / (t1 - t0)
    cells = H * W
    mcells_per_sec = (cells * gen_per_sec) / 1e6

    print(f"Size: {H}x{W}")
    print(f"Generations/sec: {gen_per_sec:.2f}")
    print(f"Throughput: {mcells_per_sec:.1f} Mcells/s")

    # Example: switch to HighLife (S23/B36)
    s_mask, b_mask = rule_masks_from_string("S23/B36")
    grid = outer_totalistic_step(runner.grid, s_mask, b_mask, wrap=True)
    print("Switched one step of HighLife.")
