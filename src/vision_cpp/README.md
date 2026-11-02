# vision_cpp — C++ / CUDA / TensorRT port of the vision stack

| Python | C++ |
|---|---|
| `vision.py` | `vision_node.{hpp,cpp}`, `vision_main.cpp` |
| `model_output.py` | `hydranet_recorder_node.cpp` |
| `model_helpers.py` | `model_helpers.{hpp,cpp}` (CPU NMS) + kernels in `gpu_kernels.cu` |
| `thinning.py` | `thinning.{hpp,cpp}` (CPU ordering) + kernels in `gpu_kernels.cu` |
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
   -> stop-line edge | thin (2 masks, fused tile kernels, device-side early exit) -> corners
   -> skeleton CC + compaction -> D2H(FrameResults, one 200 KB copy)
 CPU: NMS on ~dozens of boxes -> O(n) skeleton ordering -> IPM (no transcendentals) -> publish
```
Differences from the python pipeline, by design:
* **Mask cleaning moved to the GPU.** In python it ran on the CPU only because torch needed ~100 tiny kernels. Here it is 4 CC kernels, so the masks never leave the GPU and the old GPU->CPU->GPU round trip for thinning is gone. `cpu_threads` no longer exists.
* **Thinning**: one kernel per full Zhang-Suen iteration (smem tile, 2px halo, both sub-steps), 64 launches inside the graph; iterations after convergence exit immediately on a device counter, so there is no per-chunk host sync. Result is bit-identical to the original loop (checked with a numpy emulation of the scheme).
* **IPM** tabulates `atan/sin/cos` per image row once (when intrinsics arrive).
* No cv_bridge / OpenCV in the vision node; encodings bgr8, rgb8, bgra8, rgba8 are handled by the preprocess kernel. If the camera size differs from the model input, the frame is bilinearly resized on the GPU and the intrinsics are scaled.
* Image callback has its own callback group/thread; IMU pitch is an atomic, so it is always the newest.
* All memory (pinned staging, device buffers, TensorRT IO, ROS messages, scratch vectors) is allocated at startup.
* Engine IO: fp16 everywhere except `boxes` (fp32, fp16 only resolves 0.5 px at x~512+). Engine is rebuilt automatically if missing/stale; force with `rebuild_engine:=true`.

## Knobs for latency
`spin_wait` (busy-wait on the GPU), `rt_priority` (SCHED_FIFO), `lock_memory`, `cuda_graph`, `trt_opt_level` (try 4/5 when building), `subsample_step`, `thin_row_start`.
The `[profile]` log line reports copy / GPU (CUDA events) / wait / CPU tail / publish and camera-stamp -> done latency, without adding GPU syncs.
Not done: loaned messages / zero-copy publishing (needs a shared-memory RMW such as Cyclone+iceoryx or Fast DDS data-sharing; the persistent messages are already ready for `borrow_loaned_message`).

## Status — please read
Written without access to a GPU, TensorRT or ROS. Verified here: the CPU parts (skeleton ordering identical to `order_skeleton_pixels`, IPM within 1e-7 of `project_pixels_np`; see `test/`) and the thinning early-exit scheme. **Not compiled or run:** the `.cu` kernels, TensorRT wrapper, pipeline and nodes. Expect to fix small compile errors on first build. Suggested first checks: run the python node and this one side by side and diff `/pc_whitelane`, `/pc_yellowlane`, `/pc_stopline`, `/detections`; small differences are expected from fp16 TensorRT vs torch fp16.
Known approximations: detection candidates are capped at 2048 on the GPU (python keeps top-1000 of everything; identical unless >2048 anchors pass the threshold), skeleton lists cap at 8192 px per lane (warns on overflow).
