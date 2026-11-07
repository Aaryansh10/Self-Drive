import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from parking.slot_core import apply_grid_update
g = np.zeros((10, 20), np.int8)
assert apply_grid_update(g, 3, 2, 4, 3, [100] * 12) and (g[2:5, 3:7] == 100).all() and g.sum() == 1200
assert not apply_grid_update(g, 18, 8, 4, 3, [1] * 12)          # out of bounds -> rejected, grid untouched
assert not apply_grid_update(g, 0, 0, 2, 2, [1, 2, 3])          # wrong length -> rejected
print("OK")
