#include "vision_cpp/model_helpers.hpp"

#include <algorithm>
#include <numeric>

namespace vision_cpp {
namespace {

inline float iou(const Candidate& a, const Candidate& b) {
  const float iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  const float ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (iw <= 0.f || ih <= 0.f) return 0.f;
  const float inter = iw * ih;
  const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return uni > 0.f ? inter / uni : 0.f;
}

inline float clampf(float v, float lo, float hi) { return std::min(std::max(v, lo), hi); }

}  // namespace

void select_detections(const FrameResults& r, float nms_iou, int max_dets, float sx, float sy,
                       int fw, int fh, float min_box, DetScratch& sc, std::vector<Detection>& out) {
  out.clear();
  const int n = std::min(r.cand_count, kMaxCand);
  if (n <= 0) return;

  auto& idx = sc.idx;
  idx.resize(n);
  std::iota(idx.begin(), idx.end(), 0);
  auto better = [&](int a, int b) {
    const Candidate &A = r.cands[a], &B = r.cands[b];
    return A.score > B.score || (A.score == B.score && A.flat < B.flat);
  };
  if (n > 1000) {  // pre-NMS top-k, as in the python version
    std::nth_element(idx.begin(), idx.begin() + 1000, idx.end(), better);
    idx.resize(1000);
  }
  std::sort(idx.begin(), idx.end(), better);

  auto& keep = sc.keep;
  keep.clear();
  for (int i : idx) {  // class-wise greedy NMS == torchvision batched_nms
    const Candidate& c = r.cands[i];
    bool suppressed = false;
    for (int j : keep) {
      const Candidate& k = r.cands[j];
      if (k.label == c.label && iou(c, k) > nms_iou) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) keep.push_back(i);
  }
  if (static_cast<int>(keep.size()) > max_dets) keep.resize(max_dets);

  for (int i : keep) {
    const Candidate& c = r.cands[i];
    Detection d;
    d.x1 = clampf(c.x1 * sx, 0.f, static_cast<float>(fw - 1));
    d.x2 = clampf(c.x2 * sx, 0.f, static_cast<float>(fw - 1));
    d.y1 = clampf(c.y1 * sy, 0.f, static_cast<float>(fh - 1));
    d.y2 = clampf(c.y2 * sy, 0.f, static_cast<float>(fh - 1));
    d.score = c.score;
    d.label = c.label;
    if ((d.x2 - d.x1) > min_box && (d.y2 - d.y1) > min_box) out.push_back(d);
  }
}

}  // namespace vision_cpp
