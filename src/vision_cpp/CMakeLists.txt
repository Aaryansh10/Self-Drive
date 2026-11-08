cmake_minimum_required(VERSION 3.18)

# Declare standards before project initialization
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(vision_cpp LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -lineinfo")

# Set default GPU architecture to 86 (RTX 30-series Ampere) if unspecified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES "86")
  endif()
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_interfaces REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)
find_package(CUDAToolkit REQUIRED)

# TensorRT lookup
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
  PATHS /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/local/TensorRT /usr/src/tensorrt
  PATH_SUFFIXES include)
find_library(TENSORRT_LIB nvinfer
  HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
  PATHS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/TensorRT
  PATH_SUFFIXES lib lib64)
find_library(TENSORRT_ONNX_LIB nvonnxparser
  HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
  PATHS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/TensorRT
  PATH_SUFFIXES lib lib64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB OR NOT TENSORRT_ONNX_LIB)
  message(FATAL_ERROR "TensorRT not found; pass -DTENSORRT_ROOT=<dir>")
endif()

# Core library
add_library(vision_core STATIC
  src/gpu_kernels.cu
  src/gpu_pipeline.cpp
  src/trt_engine.cpp
  src/model_helpers.cpp
  src/ipm.cpp)


target_include_directories(vision_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${TENSORRT_INCLUDE_DIR})
target_link_libraries(vision_core PUBLIC CUDA::cudart ${TENSORRT_LIB} ${TENSORRT_ONNX_LIB})
set_target_properties(vision_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Executables
add_executable(vision_node src/vision_node.cpp src/vision_main.cpp)
target_link_libraries(vision_node vision_core)
ament_target_dependencies(vision_node rclcpp sensor_msgs std_msgs vision_interfaces)

add_executable(hydranet_recorder_node src/hydranet_recorder_node.cpp)
target_link_libraries(hydranet_recorder_node vision_core ${OpenCV_LIBS})
target_include_directories(hydranet_recorder_node PRIVATE ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(hydranet_recorder_node rclcpp sensor_msgs)

add_executable(build_engine src/build_engine.cpp)
target_link_libraries(build_engine vision_core)

install(TARGETS vision_node hydranet_recorder_node build_engine DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/export_onnx.py DESTINATION lib/${PROJECT_NAME})

ament_package()