#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lane_mapper {

// Zhang-Suen thinning to convergence. A zero halo lets lanes touching the rolling
// map boundary thin normally; unknown cells never become foreground.
inline std::vector<int8_t> skeletonize(const std::vector<int8_t>& grid, int width,
                                     int height, int threshold = 50, int min_cells = 3) {
  if (width <= 0 || height <= 0 || grid.size() != size_t(width) * height)
    throw std::invalid_argument("Invalid occupancy grid dimensions");
  const int stride = width + 2;
  std::vector<uint8_t> mask(size_t(stride) * (height + 2), 0);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      mask[size_t(y + 1) * stride + x + 1] = grid[size_t(y) * width + x] >= threshold;
  std::vector<size_t> removed;
  bool changed;
  do {
    changed = false;
    for (int pass = 0; pass < 2; ++pass) {
      removed.clear();
      for (int y = 1; y <= height; ++y) {
        for (int x = 1; x <= width; ++x) {
          const size_t i = size_t(y) * stride + x;
          if (!mask[i]) continue;
          const int n[8] = {mask[i-stride], mask[i-stride+1], mask[i+1], mask[i+stride+1],
                            mask[i+stride], mask[i+stride-1], mask[i-1], mask[i-stride-1]};
          int count = 0, transitions = 0;
          for (int k = 0; k < 8; ++k) {
            count += n[k];
            transitions += !n[k] && n[(k + 1) % 8];
          }
          if (count < 2 || count > 6 || transitions != 1) continue;
          const bool keep = pass == 0 ? ((n[0] && n[2] && n[4]) || (n[2] && n[4] && n[6]))
                                      : ((n[0] && n[2] && n[6]) || (n[0] && n[4] && n[6]));
          if (!keep) removed.push_back(i);
        }
      }
      for (const auto i : removed) mask[i] = 0;
      changed = changed || !removed.empty();
    }
  } while (changed);

  std::vector<int8_t> result(grid.size(), 0);
  for (size_t i = 0; i < grid.size(); ++i) if (grid[i] < 0) result[i] = -1;
  const int offsets[8] = {-stride, -stride+1, 1, stride+1, stride, stride-1, -1, -stride-1};
  std::vector<size_t> component;
  for (int y = 1; y <= height; ++y) {
    for (int x = 1; x <= width; ++x) {
      const size_t start = size_t(y) * stride + x;
      if (!mask[start]) continue;
      component.assign(1, start);
      mask[start] = 0;
      for (size_t j = 0; j < component.size(); ++j) {
        for (int offset : offsets) {
          const size_t next = component[j] + offset;
          if (mask[next]) { mask[next] = 0; component.push_back(next); }
        }
      }
      if (component.size() < static_cast<size_t>(std::max(1, min_cells))) continue;
      for (size_t i : component) result[(i / stride - 1) * width + i % stride - 1] = 100;
    }
  }
  return result;
}
}  // namespace lane_mapper
