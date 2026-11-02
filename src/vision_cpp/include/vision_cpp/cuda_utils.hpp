#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#define VCUDA(call)                                                                        \
  do {                                                                                     \
    cudaError_t vcuda_err_ = (call);                                                       \
    if (vcuda_err_ != cudaSuccess)                                                         \
      throw std::runtime_error(std::string("CUDA error '") + cudaGetErrorString(vcuda_err_) + \
                               "' at " + __FILE__ + ":" + std::to_string(__LINE__));       \
  } while (0)

namespace vision_cpp {

// Device memory, allocated once and zero-initialised. Never reallocated in the steady state.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) { alloc(n); }
  ~DeviceBuffer() { release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept { swap(o); }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    swap(o);
    return *this;
  }

  void alloc(size_t n) {
    release();
    if (n == 0) return;
    VCUDA(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(T)));
    VCUDA(cudaMemset(p_, 0, n * sizeof(T)));
    n_ = n;
  }
  void release() noexcept {
    if (p_) cudaFree(p_);
    p_ = nullptr;
    n_ = 0;
  }
  T* get() const { return p_; }
  size_t size() const { return n_; }

 private:
  void swap(DeviceBuffer& o) noexcept {
    std::swap(p_, o.p_);
    std::swap(n_, o.n_);
  }
  T* p_ = nullptr;
  size_t n_ = 0;
};

// Page-locked host memory (fast, truly asynchronous DMA; usable inside CUDA graphs).
template <typename T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(size_t n) { alloc(n); }
  ~PinnedBuffer() { release(); }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  void alloc(size_t n) {
    release();
    if (n == 0) return;
    VCUDA(cudaHostAlloc(reinterpret_cast<void**>(&p_), n * sizeof(T), cudaHostAllocPortable));
    std::memset(static_cast<void*>(p_), 0, n * sizeof(T));
    n_ = n;
  }
  void release() noexcept {
    if (p_) cudaFreeHost(p_);
    p_ = nullptr;
    n_ = 0;
  }
  T* get() const { return p_; }
  size_t size() const { return n_; }

 private:
  T* p_ = nullptr;
  size_t n_ = 0;
};

}  // namespace vision_cpp
