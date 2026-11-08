# vision_cpp — C++ / CUDA / TensorRT port of the vision stack

| Python | C++ |
|---|---|
| `vision.py` | `vision_node.{hpp,cpp}`, `vision_main.cpp` |
| `model_output.py` | `hydranet_recorder_node.cpp` |
| `model_helpers.py` | `model_helpers.{hpp,cpp}` (CPU NMS) + kernels in `gpu_kernels.cu` |
| `ipm.py` | `ipm.{hpp,cpp}` |
| HydraNet / torch | `trt_engine.{hpp,cpp}`, `scripts/export_onnx.py`, `build_engine` |
| (glue) | `gpu_pipeline.{hpp,cpp}` — the whole GPU workload as one CUDA graph |

## Build / run
```bash
# 1. export (in your python env)
python3 scripts/export_onnx.py --weights sim_finetuned.pt --out hydranet.onnx --verify
# 2. build
colcon build --packages-select vision_cpp --cmake-args -DTENSORRT_ROOT=/path/to/TensorRT   # (optional) -DCMAKE_CUDA_ARCHITECTURES=87
# 3. run (first start builds + caches hydranet.fp16.engine next to the ONNX, a few minutes)
ros2 run vision_cpp vision_node --ros-args --params-file src/vision_cpp/config/vision.yaml
```
Needs TensorRT >= 8.6 (written against the `enqueueV3` API, also fine on 10.x), CUDA >= 11.8, OpenCV (recorder only).

## Per-frame data flow
```
ROS msg --memcpy--> pinned staging
 [ ONE cudaGraphLaunch ]
   H2D -> preprocess(BGR->RGB, norm, fp16) -> TensorRT -> det threshold+compaction
   -> argmax -> open 3x3 -> close 5x5 -> union-find CC + area/perimeter filter (3 masks)
   -> stop-line edge -> D2H(cleaned white/yellow masks + detection/stop-line results)
 CPU: NMS -> cleaned pixel extraction -> IPM -> publish cleaned lane point clouds
 /pc_whitelane + /pc_yellowlane -> mapping_cpp -> white/yellow occupancy maps
 -> map_skeleton_node -> skeleton maps -> centerline_node -> left/right paths
```
Differences from the python pipeline, by design:
* **Mask cleaning moved to the GPU.** In python it ran on the CPU only because torch needed ~100 tiny kernels. Here it is 4 CC kernels, and only the cleaned lane masks are downloaded for IPM. `cpu_threads` no longer exists.
* **Skeletonization runs on accumulated occupancy maps**, in `mapping_cpp/map_skeleton_node`. Vision no longer thins, orders, or truncates white/yellow lane pixels. `thin_row_start` and `thin_max_iter` are removed. Stop-line leading edges and detections retain their existing interfaces.
* **IPM** tabulates `atan/sin/cos` per image row once (when intrinsics arrive).
* No cv_bridge / OpenCV in the vision node; encodings bgr8, rgb8, bgra8, rgba8 are handled by the preprocess kernel. If the camera size differs from the model input, the frame is bilinearly resized on the GPU and the intrinsics are scaled.
* Image callback has its own callback group/thread; IMU pitch is an atomic, so it is always the newest.
* All memory (pinned staging, device buffers, TensorRT IO, ROS messages, scratch vectors) is allocated at startup.
* Engine IO: fp16 everywhere except `boxes` (fp32, fp16 only resolves 0.5 px at x~512+). Engine is rebuilt automatically if missing/stale; force with `rebuild_engine:=true`.

## Knobs for latency
`spin_wait` (busy-wait on the GPU), `rt_priority` (SCHED_FIFO), `lock_memory`, `cuda_graph`, `trt_opt_level` (try 4/5 when building), `subsample_step`.
The `[profile]` log line reports copy / GPU (CUDA events) / wait / CPU tail / publish and camera-stamp -> done latency, without adding GPU syncs.
Not done: loaned messages / zero-copy publishing (needs a shared-memory RMW such as Cyclone+iceoryx or Fast DDS data-sharing; the persistent messages are already ready for `borrow_loaned_message`).

## Validation
The vision, mapping, and path-generation packages compile together. Synthetic regression tests cover map evidence, thinning, coordinate transforms, and centerline extraction; see [mapping_cpp](../mapping_cpp/README.md). Camera/TensorRT inference still needs validation with a working NVIDIA driver and the model on the target machine.

Detection candidates remain capped at 2048. Cleaned lane masks use buffers sized from the model image dimensions, so dense lanes are no longer subject to the old 8192-pixel skeleton cap. `thinning.{hpp,cpp}` remains only as a legacy CPU test utility and is not linked into the vision node.
