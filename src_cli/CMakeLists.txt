cmake_minimum_required(VERSION 3.0)

project(demucs.onnx)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -flto -fno-signed-zeros -fassociative-math -freciprocal-math -fno-math-errno -fno-rounding-math -funsafe-math-optimizations -fno-trapping-math -fno-rtti -DNDEBUG")

# Define the path to the compiled ONNX Runtime static library
set(ONNX_RUNTIME_LIB ${CMAKE_SOURCE_DIR}/../build/build-ort-linux/MinSizeRel/libonnxruntime.so)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/onnxruntime/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../onnx-models/model)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/eigen)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../src_cli/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../onnx-models/model/*.ort.c")
add_executable(demucs ${SOURCES})

target_link_libraries(demucs ${ONNX_RUNTIME_LIB} libnyquist)
