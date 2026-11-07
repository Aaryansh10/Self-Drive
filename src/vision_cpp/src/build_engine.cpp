// Offline ONNX -> TensorRT engine builder (the node also does this automatically on first start).
//   build_engine model.onnx [--fp32] [--out model.engine] [--h 544] [--w 960] [--workspace-mb 2048]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "vision_cpp/trt_engine.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s model.onnx [--fp32] [--out file] [--h N] [--w N] [--workspace-mb N]\n", argv[0]);
    return 2;
  }
  vision_cpp::TrtEngine::Options o;
  o.onnx_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--fp32")) o.fp16 = false;
    else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) o.engine_path = argv[++i];
    else if (!std::strcmp(argv[i], "--h") && i + 1 < argc) o.in_h = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--w") && i + 1 < argc) o.in_w = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--workspace-mb") && i + 1 < argc) o.workspace_mb = std::atoi(argv[++i]);
  }
  auto log = [](const std::string& s) { std::cerr << s << "\n"; };
  if (!vision_cpp::TrtEngine::build(o, log)) return 1;
  std::cerr << "wrote " << vision_cpp::TrtEngine::default_engine_path(o) << "\n";
  return 0;
}
