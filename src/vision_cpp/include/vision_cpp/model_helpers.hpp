#pragma once
#include <vector>

#include "vision_cpp/types.hpp"

namespace vision_cpp {

struct Detection {
  float x1, y1, x2, y2, score;
  int label;
};

struct DetScratch {
  std::vector<int> idx, keep;
};

// CPU half of model_helpers.py::postprocess_detections. The GPU already evaluated
// score = sqrt(sigmoid(cls) * sigmoid(ctr)) > thr and compacted the survivors into r.cands, so
// what is left (sort, pre-NMS top-1000, class-wise NMS, clamp, tiny-box filter) runs on a few
// dozen boxes. Boxes are scaled by (sx, sy) and clamped to [0, fw-1] x [0, fh-1] after NMS.
// Boxes with width or height <= min_box are dropped (pass a negative value to keep everything).
// Scratch vectors are reused between calls: no allocation in the steady state.
void select_detections(const FrameResults& r, float nms_iou, int max_dets, float sx, float sy,
                       int fw, int fh, float min_box, DetScratch& scratch,
                       std::vector<Detection>& out);

}  // namespace vision_cpp
