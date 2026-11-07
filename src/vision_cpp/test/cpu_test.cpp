// Reads test_in.txt (written by cpu_test.py), runs the C++ ipm + skeleton ordering, writes test_out.txt
#include <cstdio>
#include <vector>
#include "vision_cpp/ipm.hpp"
#include "vision_cpp/thinning.hpp"
using namespace vision_cpp;
int main() {
  FILE* f = fopen("test_in.txt", "r");
  IpmParams p; int n, r0, rows;
  double pitch;
  if (fscanf(f, "%lf %lf %lf %lf %lf %lf %lf %d %d %d %d %lf", &p.cx,&p.cy,&p.fx,&p.fy,&p.cam_pitch,&p.cam_height,&p.max_range,&p.subsample_step,&p.full_h,&r0,&rows,&pitch)!=12) return 1;
  if (fscanf(f, "%d", &n)!=1) return 1;
  std::vector<SkelPix> pix(n);
  for (auto& s : pix) { int v,u,l; if (fscanf(f,"%d %d %d",&l,&v,&u)!=3) return 1; s.label=l; s.v=v; s.u=u; }
  Ipm ipm; ipm.init(p);
  SkelScratch sc; std::vector<int> ov, ou;
  for (int rep = 0; rep < 3; ++rep) order_skeleton_pixels(pix.data(), n, r0, rows, sc, ov, ou);  // scratch reuse must be safe
  std::vector<float> pts(3 * ov.size() + 3);
  size_t k = ipm.project(ov.data(), ou.data(), ov.size(), pitch, pts.data());
  FILE* o = fopen("test_out.txt", "w");
  fprintf(o, "%zu %zu\n", ov.size(), k);
  for (size_t i = 0; i < ov.size(); ++i) fprintf(o, "%d %d\n", ov[i], ou[i]);
  for (size_t i = 0; i < k; ++i) fprintf(o, "%.9g %.9g %.9g\n", pts[3*i], pts[3*i+1], pts[3*i+2]);
  return 0;
}
