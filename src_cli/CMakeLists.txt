cmake_minimum_required(VERSION 3.0)

# Define the path to the compiled ONNX Runtime static library
# Preferred override:
#   -DDEMUCS_ONNXRUNTIME_ROOT=/path/to/onnxruntime-<platform>-<version>
# Explicit library override:
#   -DDEMUCS_ONNX_RUNTIME_LIB=/absolute/path/to/libonnxruntime.a|.so|.dylib
set(ONNX_RUNTIME_INCLUDE_DIR "")
set(ONNX_RUNTIME_LIB "")

if (DEFINED DEMUCS_ONNXRUNTIME_ROOT AND EXISTS "${DEMUCS_ONNXRUNTIME_ROOT}")
  if (EXISTS "${DEMUCS_ONNXRUNTIME_ROOT}/include")
    set(ONNX_RUNTIME_INCLUDE_DIR "${DEMUCS_ONNXRUNTIME_ROOT}/include")
  endif()

  set(_onnxruntime_root_lib_candidates
    "${DEMUCS_ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib"
    "${DEMUCS_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
    "${DEMUCS_ONNXRUNTIME_ROOT}/lib/libonnxruntime.a"
    "${DEMUCS_ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
  )
  foreach(_candidate IN LISTS _onnxruntime_root_lib_candidates)
    if (EXISTS "${_candidate}")
      set(ONNX_RUNTIME_LIB "${_candidate}")
      break()
    endif()
  endforeach()
endif()

if (ONNX_RUNTIME_LIB STREQUAL "" AND DEFINED DEMUCS_ONNX_RUNTIME_LIB AND EXISTS "${DEMUCS_ONNX_RUNTIME_LIB}")
  set(ONNX_RUNTIME_LIB "${DEMUCS_ONNX_RUNTIME_LIB}")
endif()

if (ONNX_RUNTIME_LIB STREQUAL "")
  set(_onnxruntime_candidates
    "${CMAKE_SOURCE_DIR}/../third_party/onnxruntime-release/current/lib/libonnxruntime.dylib"
    "${CMAKE_SOURCE_DIR}/../third_party/onnxruntime-release/current/lib/libonnxruntime.so"
    "${CMAKE_SOURCE_DIR}/../third_party/onnxruntime-release/current/lib/libonnxruntime.a"
set(ONNX_RUNTIME_LIB ${CMAKE_SOURCE_DIR}/../build/build-ort-linux/MinSizeRel/libonnxruntime.so)
    "${CMAKE_SOURCE_DIR}/../vendor/ort-builder/libs/macos-arm64_x86_64/libonnxruntime.a"
    "${CMAKE_SOURCE_DIR}/../vendor/ort-builder/libs/macos-arm64/libonnxruntime.a"
    "${CMAKE_SOURCE_DIR}/../vendor/ort-builder/libs/linux-x86_64/libonnxruntime.a"
  )
  foreach(_candidate IN LISTS _onnxruntime_candidates)
    if (EXISTS "${_candidate}")
      set(ONNX_RUNTIME_LIB "${_candidate}")
      break()
    endif()
  endforeach()
endif()

if (ONNX_RUNTIME_INCLUDE_DIR STREQUAL "")
  if (DEFINED DEMUCS_ONNXRUNTIME_ROOT AND EXISTS "${DEMUCS_ONNXRUNTIME_ROOT}/include")
    set(ONNX_RUNTIME_INCLUDE_DIR "${DEMUCS_ONNXRUNTIME_ROOT}/include")
  elseif (EXISTS "${CMAKE_SOURCE_DIR}/../third_party/onnxruntime-release/current/include")
    set(ONNX_RUNTIME_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../third_party/onnxruntime-release/current/include")
  elseif (EXISTS "${CMAKE_SOURCE_DIR}/../vendor/onnxruntime/include")
    set(ONNX_RUNTIME_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../vendor/onnxruntime/include")
  endif()
endif()

if (ONNX_RUNTIME_LIB STREQUAL "")
  message(FATAL_ERROR
    "Could not find ONNX Runtime library for Demucs CLI. "
    "Run scripts/fetch-onnxruntime-release.py or pass "
    "-DDEMUCS_ONNXRUNTIME_ROOT=/path/to/onnxruntime-package "
    "or -DDEMUCS_ONNX_RUNTIME_LIB=/absolute/path/to/libonnxruntime.")
endif()

if (ONNX_RUNTIME_INCLUDE_DIR STREQUAL "")
  message(FATAL_ERROR
    "Could not find ONNX Runtime headers. "
    "Pass -DDEMUCS_ONNXRUNTIME_ROOT=/path/to/onnxruntime-package (contains include/).")
endif()

include_directories(${ONNX_RUNTIME_INCLUDE_DIR})

get_filename_component(ONNX_RUNTIME_LIB_DIR "${ONNX_RUNTIME_LIB}" DIRECTORY)
if (UNIX AND ONNX_RUNTIME_LIB MATCHES "\\.(so|dylib)$")
  set_target_properties(demucs PROPERTIES
    BUILD_RPATH "${ONNX_RUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNX_RUNTIME_LIB_DIR}")
endif()
