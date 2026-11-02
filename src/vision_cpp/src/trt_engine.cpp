#include "vision_cpp/trt_engine.hpp"

#include <NvOnnxParser.h>

#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace vision_cpp {
namespace {

class Logger : public nvinfer1::ILogger {
 public:
  explicit Logger(TrtEngine::LogFn f) : f_(std::move(f)) {}
  void log(Severity sev, const char* msg) noexcept override {
    if (sev > Severity::kWARNING || !f_) return;
    try {
      f_(std::string("[TensorRT] ") + msg);
    } catch (...) {
    }
  }

 private:
  TrtEngine::LogFn f_;
};

bool read_file(const std::string& path, std::vector<char>& out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const std::streamsize n = f.tellg();
  if (n <= 0) return false;
  out.resize(static_cast<size_t>(n));
  f.seekg(0);
  return static_cast<bool>(f.read(out.data(), n));
}

size_t volume(const nvinfer1::Dims& d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
  return v;
}

std::string dims_str(const nvinfer1::Dims& d) {
  std::ostringstream o;
  o << "(";
  for (int i = 0; i < d.nbDims; ++i) o << (i ? "," : "") << d.d[i];
  o << ")";
  return o.str();
}

}  // namespace

std::string TrtEngine::default_engine_path(const Options& o) {
  if (!o.engine_path.empty()) return o.engine_path;
  std::string base = o.onnx_path;
  const auto dot = base.find_last_of('.');
  if (dot != std::string::npos) base.resize(dot);
  return base + (o.fp16 ? ".fp16" : ".fp32") + ".engine";
}

// -------------------------------------------------------------------------------------------------
bool TrtEngine::build(const Options& o, const LogFn& log) {
  using namespace nvinfer1;
  Logger lg(log);
  std::unique_ptr<IBuilder> builder(createInferBuilder(lg));
  if (!builder) return false;
// TensorRT >= 10.12 removed the weakly-typed precision API (BuilderFlag::kFP16, ITensor::setType,
// setAllowedFormats, platformHasFastFp16). There the network is strongly typed: precision comes from
// the dtypes in the ONNX file (export it in fp16, see scripts/export_onnx.py).
#if (NV_TENSORRT_MAJOR > 10) || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 12)
#define VISION_TRT_STRONGLY_TYPED 1
#else
#define VISION_TRT_STRONGLY_TYPED 0
#endif
#if VISION_TRT_STRONGLY_TYPED
  std::unique_ptr<INetworkDefinition> net(builder->createNetworkV2(
      1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
#elif NV_TENSORRT_MAJOR >= 10
  std::unique_ptr<INetworkDefinition> net(builder->createNetworkV2(0));
#else
  std::unique_ptr<INetworkDefinition> net(builder->createNetworkV2(
      1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
#endif
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, lg));
  if (!parser->parseFromFile(o.onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
    for (int i = 0; i < parser->getNbErrors(); ++i) log(std::string("[onnx] ") + parser->getError(i)->desc());
    return false;
  }

  std::unique_ptr<IBuilderConfig> cfg(builder->createBuilderConfig());
  cfg->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, static_cast<size_t>(o.workspace_mb) << 20);
#if !VISION_TRT_STRONGLY_TYPED
  if (o.fp16) {
    if (!builder->platformHasFastFp16()) log("[build] warning: platform has no fast FP16");
    cfg->setFlag(BuilderFlag::kFP16);
  }
#else
  log("[build] strongly-typed network: precision is taken from the ONNX dtypes");
#endif
#if (NV_TENSORRT_MAJOR > 8) || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 6)
  cfg->setBuilderOptimizationLevel(o.opt_level);
#endif

#if !VISION_TRT_STRONGLY_TYPED
  // Linear-layout IO. fp16 everywhere except the boxes (fp16 only resolves 0.5 px at x~512-1024).
  const uint32_t linear = 1U << static_cast<int>(TensorFormat::kLINEAR);
  const DataType lo = o.fp16 ? DataType::kHALF : DataType::kFLOAT;
  for (int i = 0; i < net->getNbInputs(); ++i) {
    ITensor* t = net->getInput(i);
    t->setType(lo);
    t->setAllowedFormats(linear);
  }
  for (int i = 0; i < net->getNbOutputs(); ++i) {
    ITensor* t = net->getOutput(i);
    t->setType(std::string(t->getName()) == o.box_name ? DataType::kFLOAT : lo);
    t->setAllowedFormats(linear);
  }
#endif

  log("[build] building TensorRT engine (this takes a few minutes, once)...");
  std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*net, *cfg));
  if (!plan || plan->size() == 0) return false;
  std::ofstream f(default_engine_path(o), std::ios::binary);
  f.write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
  return static_cast<bool>(f);
}

// -------------------------------------------------------------------------------------------------
TrtEngine::TrtEngine(const Options& o, LogFn log) : opt_(o), log_(std::move(log)) {
  logger_ = std::make_unique<Logger>(log_);
  const std::string path = default_engine_path(opt_);

  bool ok = !opt_.rebuild && try_load(path);
  if (!ok) {
    if (opt_.onnx_path.empty()) throw std::runtime_error("No usable engine at " + path + " and no onnx_path given");
    log_("[trt] building engine from " + opt_.onnx_path);
    if (!build(opt_, log_)) throw std::runtime_error("TensorRT engine build failed");
    ok = try_load(path);
  }
  if (!ok) throw std::runtime_error("Could not load TensorRT engine " + path);
  allocate_io();
  log_("[trt] engine ready: " + path);
}

TrtEngine::~TrtEngine() = default;

bool TrtEngine::try_load(const std::string& path) {
  std::vector<char> blob;
  if (!read_file(path, blob)) return false;
  ctx_.reset();
  engine_.reset();
  runtime_.reset(nvinfer1::createInferRuntime(*logger_));
  if (!runtime_) return false;
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) {
    log_("[trt] could not deserialise " + path + " (different TensorRT/GPU?) -> will rebuild");
    return false;
  }
  ctx_.reset(engine_->createExecutionContext());
  return ctx_ && validate();
}

bool TrtEngine::validate() {
  std::set<std::string> names;
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) names.insert(engine_->getIOTensorName(i));
  for (const std::string* n : {&opt_.input_name, &opt_.lane_name, &opt_.cls_name, &opt_.ctr_name, &opt_.box_name})
    if (!names.count(*n)) {
      std::string all;
      for (const auto& s : names) all += s + " ";
      log_("[trt] tensor '" + *n + "' not in engine (has: " + all + ")");
      return false;
    }
  const auto in = engine_->getTensorShape(opt_.input_name.c_str());
  const auto lane = engine_->getTensorShape(opt_.lane_name.c_str());
  const auto cls = engine_->getTensorShape(opt_.cls_name.c_str());
  const auto ctr = engine_->getTensorShape(opt_.ctr_name.c_str());
  const auto box = engine_->getTensorShape(opt_.box_name.c_str());

  if (in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3 || in.d[2] != opt_.in_h || in.d[3] != opt_.in_w) {
    log_("[trt] engine input is " + dims_str(in) + ", expected (1,3," + std::to_string(opt_.in_h) + "," + std::to_string(opt_.in_w) + ")");
    return false;
  }
  if (lane.nbDims != 4 || lane.d[0] != 1 || lane.d[1] < 4 || lane.d[2] != opt_.in_h || lane.d[3] != opt_.in_w) {
    log_("[trt] lane_logits is " + dims_str(lane) + ", expected (1,>=4,H,W) at input resolution");
    return false;
  }
  if (cls.nbDims != 3 || ctr.nbDims != 3 || box.nbDims != 3 || cls.d[1] != ctr.d[1] ||
      cls.d[1] != box.d[1] || ctr.d[2] != 1 || box.d[2] != 4) {
    log_("[trt] detection head shapes unexpected: " + dims_str(cls) + " " + dims_str(ctr) + " " + dims_str(box));
    return false;
  }
  lane_c_ = lane.d[1];
  lane_h_ = lane.d[2];
  lane_w_ = lane.d[3];
  n_anchors_ = cls.d[1];
  n_classes_ = cls.d[2];
  return true;
}

void TrtEngine::allocate_io() {
  const std::string* names[5] = {&opt_.input_name, &opt_.lane_name, &opt_.cls_name, &opt_.ctr_name, &opt_.box_name};
  for (int i = 0; i < 5; ++i) {
    const char* n = names[i]->c_str();
    const auto dt = engine_->getTensorDataType(n);
    if (dt != nvinfer1::DataType::kFLOAT && dt != nvinfer1::DataType::kHALF)
      throw std::runtime_error("IO tensor '" + *names[i] + "' must be fp32 or fp16");
    half_[i] = dt == nvinfer1::DataType::kHALF;
    io_[i].alloc(volume(engine_->getTensorShape(n)) * (half_[i] ? 2 : 4));
    if (!ctx_->setTensorAddress(n, io_[i].get())) throw std::runtime_error("setTensorAddress failed for " + *names[i]);
  }
}

}  // namespace vision_cpp