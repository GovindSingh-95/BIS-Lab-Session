import threading
import numpy as np

# Cellular Automaton rule: Rule 30 (example)
def rule_30(left, center, right):
    pattern = (left << 2) | (center << 1) | right
    return (0b00011110 >> pattern) & 1  # Rule 30 binary pattern

# Worker function for each thread
def update_segment(start, end, old_grid, new_grid, left_halo, right_halo):
    for i in range(start, end):
        left = old_grid[i - 1] if i > 0 else left_halo
        right = old_grid[i + 1] if i < len(old_grid) - 1 else right_halo
        new_grid[i] = rule_30(left, old_grid[i], right)

# Main
n_cells = 20
n_steps = 10
n_threads = 4

grid = np.zeros(n_cells, dtype=int)
grid[n_cells // 2] = 1  # Start with a single 1 in the center

print("Initial:", ''.join(str(x) for x in grid))

for step in range(n_steps):
    new_grid = np.zeros_like(grid)
    threads = []
    chunk = n_cells // n_threads

    for t in range(n_threads):
        start = t * chunk
        end = (t + 1) * chunk if t != n_threads - 1 else n_cells
        left_halo = grid[start - 1] if start > 0 else 0
        right_halo = grid[end] if end < n_cells else 0

        thread = threading.Thread(target=update_segment,
                                  args=(start, end, grid, new_grid, left_halo, right_halo))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    grid = new_grid.copy()
    print('Step', step + 1, ':', ''.join(str(x) for x in grid))
