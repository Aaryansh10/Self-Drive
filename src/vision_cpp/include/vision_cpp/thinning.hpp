#pragma once
#include <cstdint>
#include <vector>

#include "vision_cpp/types.hpp"

namespace vision_cpp {

struct SkelScratch {
  struct Slot {
    uint32_t stamp;
    int label;
    int idx;
  };
  std::vector<Slot> table;  // tiny open-addressing hash: component label -> dense index
  uint32_t stamp = 0;
  std::vector<int> cid, labels, cnt, order, rank, bucket, cursor, su;
  std::vector<double> sumu;
};

// CPU half of thinning.py::order_skeleton_pixels.
//
// The GPU has already thinned, labelled (8-connected) and dropped components shorter than min_len,
// and delivered the survivors as an UNORDERED list of (component, v, u). This reproduces the python
// ordering exactly:  components left -> right by mean column (ties: label), and inside a component
// bottom -> top (v descending), then u ascending. It is an O(n) counting sort over
// (component rank, row) - no std::sort on thousands of elements.
//
// Pixels have v in [v_min, v_min + rows). Output v is in the same (full-mask row) coordinates.
void order_skeleton_pixels(const SkelPix* pix, int n, int v_min, int rows, SkelScratch& sc,
                           std::vector<int>& out_v, std::vector<int>& out_u);

}  // namespace vision_cpp
