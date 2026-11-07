#include "vision_cpp/thinning.hpp"

#include <algorithm>
#include <numeric>

namespace vision_cpp {

void order_skeleton_pixels(const SkelPix* pix, int n, int v_min, int rows, SkelScratch& sc,
                           std::vector<int>& out_v, std::vector<int>& out_u) {
  out_v.clear();
  out_u.clear();
  if (n <= 0 || rows <= 0) return;

  constexpr uint32_t kSlots = 1u << 15;  // >= 4x the maximum possible number of components
  if (sc.table.empty()) sc.table.assign(kSlots, SkelScratch::Slot{0, 0, 0});
  if (++sc.stamp == 0) {  // wrap-around: invalidate everything
    for (auto& s : sc.table) s.stamp = 0;
    sc.stamp = 1;
  }

  // 1. dense component ids + per-component pixel count and column sum
  sc.cid.resize(n);
  sc.labels.clear();
  sc.cnt.clear();
  sc.sumu.clear();
  for (int i = 0; i < n; ++i) {
    const int label = pix[i].label;
    uint32_t h = (static_cast<uint32_t>(label) * 2654435761u) >> 17;
    for (;;) {
      auto& s = sc.table[h];
      if (s.stamp != sc.stamp) {
        s.stamp = sc.stamp;
        s.label = label;
        s.idx = static_cast<int>(sc.labels.size());
        sc.labels.push_back(label);
        sc.cnt.push_back(0);
        sc.sumu.push_back(0.0);
      }
      if (s.label == label) {
        sc.cid[i] = s.idx;
        ++sc.cnt[s.idx];
        sc.sumu[s.idx] += pix[i].u;
        break;
      }
      h = (h + 1) & (kSlots - 1);
    }
  }

  // 2. component rank: (mean u, label)
  const int nc = static_cast<int>(sc.labels.size());
  sc.order.resize(nc);
  std::iota(sc.order.begin(), sc.order.end(), 0);
  std::sort(sc.order.begin(), sc.order.end(), [&](int a, int b) {
    const double ma = sc.sumu[a] / sc.cnt[a], mb = sc.sumu[b] / sc.cnt[b];
    return ma < mb || (ma == mb && sc.labels[a] < sc.labels[b]);
  });
  sc.rank.resize(nc);
  for (int k = 0; k < nc; ++k) sc.rank[sc.order[k]] = k;

  // 3. counting sort by bucket = rank * rows + (rows - 1 - (v - v_min))   [row descending]
  const size_t nb = static_cast<size_t>(nc) * rows;
  sc.bucket.assign(nb + 1, 0);
  auto bucket_of = [&](int i) -> long {
    const int v = pix[i].v - v_min;
    if (v < 0 || v >= rows) return -1;
    return static_cast<long>(sc.rank[sc.cid[i]]) * rows + (rows - 1 - v);
  };
  for (int i = 0; i < n; ++i) {
    const long b = bucket_of(i);
    if (b >= 0) ++sc.bucket[b + 1];
  }
  for (size_t b = 0; b < nb; ++b) sc.bucket[b + 1] += sc.bucket[b];
  const int total = sc.bucket[nb];
  sc.cursor.assign(sc.bucket.begin(), sc.bucket.end() - 1);
  sc.su.resize(total);
  for (int i = 0; i < n; ++i) {
    const long b = bucket_of(i);
    if (b >= 0) sc.su[sc.cursor[b]++] = pix[i].u;
  }

  // 4. emit; u ascending inside a (component, row) bucket
  out_v.reserve(total);
  out_u.reserve(total);
  for (size_t b = 0; b < nb; ++b) {
    const int s = sc.bucket[b], e = sc.bucket[b + 1];
    if (s == e) continue;
    if (e - s > 1) std::sort(sc.su.begin() + s, sc.su.begin() + e);
    const int v = v_min + rows - 1 - static_cast<int>(b % rows);
    for (int j = s; j < e; ++j) {
      out_v.push_back(v);
      out_u.push_back(sc.su[j]);
    }
  }
}

}  // namespace vision_cpp
