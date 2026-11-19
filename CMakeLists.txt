cmake_minimum_required(VERSION 3.24)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" NISSHOKU_VERSION)
string(STRIP "${NISSHOKU_VERSION}" NISSHOKU_VERSION)
project(NisshokuSR VERSION ${NISSHOKU_VERSION} LANGUAGES CXX CUDA)

# Максимум перфа под Windows: Release x64, без отладки в hot-path.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)

set(CMAKE_CUDA_ARCHITECTURES "75;86;89;120" CACHE STRING "" FORCE) # Turing sm_75 / Ampere sm_86 / Ada sm_89 / Blackwell sm_120
find_package(CUDAToolkit REQUIRED)

# --- Зависимости через env-пути (не хардкодим версии) ---
# NVCODEC_SDK_DIR = корень Video Codec SDK (там Samples/Utils + include)
# TRT_ROOT       = корень TensorRT (include + lib)
# FFMPEG_ROOT    = корень FFmpeg shared (include + lib)
if(NOT DEFINED ENV{NVCODEC_SDK_DIR})
  message(FATAL_ERROR "Задай env NVCODEC_SDK_DIR (Video Codec SDK 13.1+)")
endif()
if(NOT DEFINED ENV{TRT_ROOT})
  message(FATAL_ERROR "Задай env TRT_ROOT (TensorRT 10.8+)")
endif()
if(NOT DEFINED ENV{FFMPEG_ROOT})
  message(FATAL_ERROR "Задай env FFMPEG_ROOT (FFmpeg shared)")
endif()
set(NVCODEC_SDK_DIR $ENV{NVCODEC_SDK_DIR})
set(TRT_ROOT $ENV{TRT_ROOT})
set(FFMPEG_ROOT $ENV{FFMPEG_ROOT})

add_library(prepost_kernels STATIC
  kernels/pre_post.cu
)
set_target_properties(prepost_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(prepost_kernels PUBLIC kernels ${CUDAToolkit_INCLUDE_DIRS})

# Утилиты SDK собираем как есть (без правок): NvDecoder / NvEncoder.
# Раскладка SDK 13.x: Samples/NvCodec/NvDecoder + Samples/NvCodec/NvEncoder.
add_library(nvcodec_utils STATIC
  ${NVCODEC_SDK_DIR}/Samples/NvCodec/NvDecoder/NvDecoder.cpp
  ${NVCODEC_SDK_DIR}/Samples/NvCodec/NvEncoder/NvEncoder.cpp
  ${NVCODEC_SDK_DIR}/Samples/NvCodec/NvEncoder/NvEncoderCuda.cpp
)
target_include_directories(nvcodec_utils PUBLIC
  ${NVCODEC_SDK_DIR}/Samples/NvCodec
  ${NVCODEC_SDK_DIR}/Samples/NvCodec/NvDecoder
  ${NVCODEC_SDK_DIR}/Samples/NvCodec/NvEncoder
  ${NVCODEC_SDK_DIR}/Samples/Utils
  ${NVCODEC_SDK_DIR}/Interface
  ${CUDAToolkit_INCLUDE_DIRS}
)
# nvcuvid (парсер/декодер) + nvencodeapi (энкодер) из комплекта SDK.
find_library(NVCUVID_LIB NAMES nvcuvid PATHS ${NVCODEC_SDK_DIR}/Lib/win/x64 NO_DEFAULT_PATH REQUIRED)
find_library(NVENCODEAPI_LIB NAMES nvencodeapi PATHS ${NVCODEC_SDK_DIR}/Lib/win/x64 NO_DEFAULT_PATH REQUIRED)
target_link_libraries(nvcodec_utils PUBLIC ${NVCUVID_LIB} ${NVENCODEAPI_LIB} CUDA::cuda_driver CUDA::cudart)

add_executable(nisshoku
  src/main.cpp
  src/Batch.cpp
  src/Cli.cpp
  src/CudaContext.cpp
  src/Demuxer.cpp
  src/NvDecWrapper.cpp
  src/TrtUpscaler.cpp
  src/NvEncWrapper.cpp
  src/Muxer.cpp
  src/Pipeline.cpp
  src/WinThread.cpp
)
target_include_directories(nisshoku PRIVATE
  src kernels
  ${NVCODEC_SDK_DIR}/Samples/NvCodec
  ${NVCODEC_SDK_DIR}/Samples/NvCodec/NvDecoder
  ${NVCODEC_SDK_DIR}/Samples/NvCodec/NvEncoder
  ${NVCODEC_SDK_DIR}/Samples/Utils
  ${NVCODEC_SDK_DIR}/Interface
  ${TRT_ROOT}/include
  ${FFMPEG_ROOT}/include
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_directories(nisshoku PRIVATE
  ${FFMPEG_ROOT}/lib
)
# TRT 10.x шипает версионированные import-lib: nvinfer_10.lib и т.д.
find_library(TRT_NVINFER_LIB NAMES nvinfer_10 nvinfer PATHS ${TRT_ROOT}/lib NO_DEFAULT_PATH REQUIRED)
find_library(TRT_NVINFER_PLUGIN_LIB NAMES nvinfer_plugin_10 nvinfer_plugin PATHS ${TRT_ROOT}/lib NO_DEFAULT_PATH REQUIRED)
find_library(TRT_ONNXPARSER_LIB NAMES nvonnxparser_10 nvonnxparser PATHS ${TRT_ROOT}/lib NO_DEFAULT_PATH REQUIRED)
target_link_libraries(nisshoku PRIVATE
  prepost_kernels
  nvcodec_utils
  CUDA::cuda_driver CUDA::cudart CUDA::nppicc CUDA::nppidei
  ${TRT_NVINFER_LIB} ${TRT_NVINFER_PLUGIN_LIB} ${TRT_ONNXPARSER_LIB}
  avformat avcodec avutil swresample
  avrt winmm dbghelp
)

# Флаги под скорость: /O2 /GL, CUDA --use_fast_math.
target_compile_options(nisshoku PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2;/GL;/EHsc>)
target_compile_definitions(nisshoku PRIVATE NISSHOKU_VERSION="${PROJECT_VERSION}")
target_compile_options(prepost_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3>)
if(MSVC)
  # CUDA 12.8 не знает компилятор VS18/2026: разрешаем явно (проверено на Blackwell-билдах).
  target_compile_options(prepost_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)
endif()
set_target_properties(nisshoku PROPERTIES LINK_FLAGS "/LTCG")
