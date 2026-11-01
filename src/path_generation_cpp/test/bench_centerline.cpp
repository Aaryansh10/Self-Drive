// Standalone harness (no ROS): replays recorded frames through the generator.
//   centerline_bench frames.txt [out.txt] [repeat]
// frames.txt: "F" then per frame "nW nY" followed by nW then nY lines of "x y".
// out.txt:    per frame "ok nL nR" followed by nL then nR lines of "x y".
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "path_generation_cpp/centerline_generator.hpp"

using centerline::Point2;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: centerline_bench frames.txt [out.txt] [repeat]\n";
    return 1;
  }
  centerline::Params params;
  for (int i = 4; i < argc; ++i) {  // optional key=value overrides
    const std::string kv = argv[i];
    const auto eq = kv.find('=');
    const std::string k = kv.substr(0, eq);
    const double v = std::atof(kv.c_str() + eq + 1);
    if (k == "sg_polyorder") params.sg_polyorder = static_cast<int>(v);
    else if (k == "lane_change_bezier") params.lane_change_bezier = v != 0.0;
    else if (k == "nb_min") params.normal_blend_tx_min = v;
    else if (k == "nb_max") params.normal_blend_tx_max = v;
    else if (k == "min_bin_points") params.min_bin_points = static_cast<int>(v);
    else if (k == "max_bad_frame_streak") params.max_bad_frame_streak = static_cast<int>(v);
    else { std::cerr << "unknown override " << k << "\n"; return 1; }
  }
  std::ifstream in(argv[1]);
  const std::string out_path = argc > 2 ? argv[2] : "";
  const int repeat = argc > 3 ? std::atoi(argv[3]) : 1;

  struct Frame { std::vector<Point2> w, y; };
  std::vector<Frame> frames;
  int F = 0;
  in >> F;
  for (int f = 0; f < F; ++f) {
    int nw = 0, ny = 0;
    in >> nw >> ny;
    Frame fr;
    fr.w.resize(nw);
    fr.y.resize(ny);
    for (auto& p : fr.w) in >> p.x >> p.y;
    for (auto& p : fr.y) in >> p.x >> p.y;
    frames.push_back(std::move(fr));
  }

  std::ofstream out;
  if (!out_path.empty()) {
    out.open(out_path);
    out.precision(10);
  }

  std::vector<double> times_us;
  for (int rep = 0; rep < repeat; ++rep) {
    centerline::CenterlineGenerator gen(params);  // fresh state per pass
    for (const auto& fr : frames) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto res = gen.generate(fr.w, fr.y);
      const auto t1 = std::chrono::steady_clock::now();
      times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      if (rep == 0 && out.is_open()) {
        out << (res.ok ? 1 : 0) << ' ' << res.left.size() << ' ' << res.right.size() << '\n';
        for (const auto& p : res.left) out << p.x << ' ' << p.y << '\n';
        for (const auto& p : res.right) out << p.x << ' ' << p.y << '\n';
      }
    }
  }

  std::sort(times_us.begin(), times_us.end());
  double sum = 0;
  for (double t : times_us) sum += t;
  std::printf("frames=%zu  mean=%.1f us  median=%.1f us  p99=%.1f us  max=%.1f us\n", times_us.size(),
              sum / times_us.size(), times_us[times_us.size() / 2],
              times_us[static_cast<size_t>(times_us.size() * 0.99)], times_us.back());
  return 0;
}