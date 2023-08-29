cmake_minimum_required(VERSION 3.10)
project(trt_infer)

# Require C++11 (or later)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
  message(STATUS ${CUDA_INCLUDE_DIRS})
  cuda_add_executable(${PROJECT_NAME} trt_infer.cpp)
  target_link_libraries(${PROJECT_NAME} nvinfer)
endif(CUDA_FOUND)
