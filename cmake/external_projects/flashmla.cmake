include(FetchContent)

# If FLASH_MLA_SRC_DIR is set, flash-mla is installed from that directory 
# instead of downloading.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{FLASH_MLA_SRC_DIR})
  set(FLASH_MLA_SRC_DIR $ENV{FLASH_MLA_SRC_DIR})
endif()

if(FLASH_MLA_SRC_DIR)
  get_filename_component(flashmla_SOURCE_DIR "${FLASH_MLA_SRC_DIR}" ABSOLUTE)
else()
  FetchContent_Declare(
        flashmla
        GIT_REPOSITORY https://github.com/vllm-project/FlashMLA
        GIT_TAG 46d64a8ebef03fa50b4ae74937276a5c940e3f95
        GIT_PROGRESS TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
  )
  FetchContent_MakeAvailable(flashmla)
endif()

message(STATUS "FlashMLA is available at ${flashmla_SOURCE_DIR}")

# Detect available FlashMLA kernel families and repo layout.
set(FLASH_MLA_HAS_SM90 FALSE)
if(EXISTS "${flashmla_SOURCE_DIR}/csrc/sm90/decode/dense/splitkv_mla.cu")
    set(FLASH_MLA_HAS_SM90 TRUE)
endif()

set(FLASH_MLA_HAS_SM100 FALSE)
if(EXISTS "${flashmla_SOURCE_DIR}/csrc/sm100/decode/sparse_fp8/splitkv_mla.cu")
    set(FLASH_MLA_HAS_SM100 TRUE)
endif()

set(FLASH_MLA_HAS_SM120 FALSE)
if(EXISTS "${flashmla_SOURCE_DIR}/csrc/sm120")
    set(FLASH_MLA_HAS_SM120 TRUE)
endif()

set(FLASH_MLA_NEEDS_BUILD_MACROS FALSE)
if(EXISTS "${flashmla_SOURCE_DIR}/csrc/msvc_compat.h")
    set(FLASH_MLA_NEEDS_BUILD_MACROS TRUE)
endif()

set(FLASH_MLA_HAS_EXTENSION FALSE)
if(EXISTS "${flashmla_SOURCE_DIR}/csrc/extension/torch_api.cpp")
    set(FLASH_MLA_HAS_EXTENSION TRUE)
endif()

# The FlashMLA kernels only work on hopper and require CUDA 12.3 or later.
# Only build FlashMLA kernels if we are building for something compatible with 
# sm90a

set(SUPPORT_ARCHS)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3)
    # Enable PTX fallback so newer architectures can JIT from sm90a.
    # NOTE: Keep 9.0+PTX so Blackwell (12.0) can match via loose intersection.
    if(FLASH_MLA_HAS_SM90)
        list(APPEND SUPPORT_ARCHS "9.0+PTX" "9.0a+PTX")
    endif()
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.9)
    # CUDA 12.9 has introduced "Family-Specific Architecture Features"
    # this supports all compute_10x family
    if(FLASH_MLA_HAS_SM100)
        list(APPEND SUPPORT_ARCHS "10.0f")
    endif()
elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
    if(FLASH_MLA_HAS_SM100)
        list(APPEND SUPPORT_ARCHS "10.0a")
    endif()
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
    # CUDA 12.8 adds SM120 support (Blackwell).
    if(FLASH_MLA_HAS_SM120)
        # Allow opting in via env until this path is fully validated.
        if(NOT DEFINED ENV{FLASH_MLA_ENABLE_SM120} OR "$ENV{FLASH_MLA_ENABLE_SM120}" STREQUAL "0")
            # Keep disabled unless explicitly enabled.
        else()
            list(APPEND SUPPORT_ARCHS "12.0" "12.0a")
        endif()
    endif()
endif()


cuda_archs_loose_intersection(FLASH_MLA_ARCHS "${SUPPORT_ARCHS}" "${CUDA_ARCHS}")
if(FLASH_MLA_ARCHS)
    message(STATUS "FlashMLA CUDA architectures: ${FLASH_MLA_ARCHS}")
    set(VLLM_FLASHMLA_GPU_FLAGS ${VLLM_GPU_FLAGS})
    list(APPEND VLLM_FLASHMLA_GPU_FLAGS "--expt-relaxed-constexpr" "--expt-extended-lambda" "--use_fast_math")

    set(FLASH_MLA_BUILD_SM90 FALSE)
    if(FLASH_MLA_ARCHS MATCHES "9\\.0")
        set(FLASH_MLA_BUILD_SM90 TRUE)
    endif()
    set(FLASH_MLA_BUILD_SM100 FALSE)
    if(FLASH_MLA_ARCHS MATCHES "10\\.0")
        set(FLASH_MLA_BUILD_SM100 TRUE)
    endif()
    set(FLASH_MLA_BUILD_SM120 FALSE)
    if(FLASH_MLA_ARCHS MATCHES "12\\.0")
        set(FLASH_MLA_BUILD_SM120 TRUE)
    endif()
    if(FLASH_MLA_BUILD_SM120 AND NOT FLASH_MLA_NEEDS_BUILD_MACROS)
        if(FLASH_MLA_HAS_SM90)
            set(FLASH_MLA_BUILD_SM90 TRUE)
        endif()
        if(FLASH_MLA_HAS_SM100)
            set(FLASH_MLA_BUILD_SM100 TRUE)
        endif()
    endif()

    set(FlashMLA_SOURCES
        ${flashmla_SOURCE_DIR}/csrc/pybind.cpp
        ${flashmla_SOURCE_DIR}/csrc/smxx/get_mla_metadata.cu
        ${flashmla_SOURCE_DIR}/csrc/smxx/mla_combine.cu
    )
    if(EXISTS "${flashmla_SOURCE_DIR}/csrc/torch_api.cpp")
        list(APPEND FlashMLA_SOURCES ${flashmla_SOURCE_DIR}/csrc/torch_api.cpp)
    endif()
    if(FLASH_MLA_BUILD_SM90 AND EXISTS "${flashmla_SOURCE_DIR}/csrc/sm90/decode/dense/splitkv_mla.cu")
        list(APPEND FlashMLA_SOURCES
            ${flashmla_SOURCE_DIR}/csrc/sm90/decode/dense/splitkv_mla.cu
            ${flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/splitkv_mla.cu
            ${flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/fwd.cu
        )
    endif()
    if(FLASH_MLA_BUILD_SM100 AND EXISTS "${flashmla_SOURCE_DIR}/csrc/sm100/decode/sparse_fp8/splitkv_mla.cu")
        list(APPEND FlashMLA_SOURCES
            ${flashmla_SOURCE_DIR}/csrc/sm100/decode/sparse_fp8/splitkv_mla.cu
        )
    endif()
    if(FLASH_MLA_BUILD_SM100 AND EXISTS "${flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu")
        list(APPEND FlashMLA_SOURCES
            ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu
            ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu
        )
    endif()
    if(FLASH_MLA_BUILD_SM100 AND EXISTS "${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd.cu")
        list(APPEND FlashMLA_SOURCES
            ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd.cu
        )
    endif()
    if(FLASH_MLA_BUILD_SM120 AND EXISTS "${flashmla_SOURCE_DIR}/csrc/sm120/decode/sparse_fp8/splitkv_mla.cu")
        list(APPEND FlashMLA_SOURCES
            ${flashmla_SOURCE_DIR}/csrc/sm120/decode/sparse_fp8/splitkv_mla.cu
        )
    endif()
    if(FLASH_MLA_BUILD_SM120 AND EXISTS "${flashmla_SOURCE_DIR}/csrc/sm120/mla_combine.cu")
        list(APPEND FlashMLA_SOURCES
            ${flashmla_SOURCE_DIR}/csrc/sm120/mla_combine.cu
        )
    endif()
    if(FLASH_MLA_BUILD_SM120 AND EXISTS "${flashmla_SOURCE_DIR}/csrc/sm120/decode/dense/splitkv_mla.cu")
        list(APPEND FlashMLA_SOURCES
            ${flashmla_SOURCE_DIR}/csrc/sm120/decode/dense/splitkv_mla.cu
        )
    endif()
    if(FLASH_MLA_BUILD_SM120 AND EXISTS "${flashmla_SOURCE_DIR}/csrc/sm120/prefill/dense/fmha_cutlass_fwd_sm120.cu")
        list(APPEND FlashMLA_SOURCES
            ${flashmla_SOURCE_DIR}/csrc/sm120/prefill/dense/fmha_cutlass_fwd_sm120.cu
            ${flashmla_SOURCE_DIR}/csrc/sm120/prefill/dense/fmha_cutlass_bwd_sm120.cu
        )
    endif()
    if(FLASH_MLA_BUILD_SM120 AND EXISTS "${flashmla_SOURCE_DIR}/csrc/sm120/prefill/sparse/fwd.cu")
        list(APPEND FlashMLA_SOURCES
            ${flashmla_SOURCE_DIR}/csrc/sm120/prefill/sparse/fwd.cu
        )
    endif()

    set(FlashMLA_Extension_SOURCES)
    if(FLASH_MLA_HAS_EXTENSION)
        set(FlashMLA_Extension_SOURCES
            ${flashmla_SOURCE_DIR}/csrc/extension/torch_api.cpp
            ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/pybind.cpp
            ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_fp8_sm90.cu
            ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_metadata.cu
        )
    endif()

    set(FlashMLA_INCLUDES
        ${flashmla_SOURCE_DIR}/csrc
        ${flashmla_SOURCE_DIR}/csrc/sm90
        ${flashmla_SOURCE_DIR}/csrc/cutlass/include
        ${flashmla_SOURCE_DIR}/csrc/cutlass/tools/util/include
    )

    set(FlashMLA_Extension_INCLUDES)
    if(FLASH_MLA_HAS_EXTENSION)
        set(FlashMLA_Extension_INCLUDES
            ${flashmla_SOURCE_DIR}/csrc
            ${flashmla_SOURCE_DIR}/csrc/sm90
            ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/
            ${flashmla_SOURCE_DIR}/csrc/cutlass/include
            ${flashmla_SOURCE_DIR}/csrc/cutlass/tools/util/include
        )
    endif()

    set_gencode_flags_for_srcs(
        SRCS "${FlashMLA_SOURCES}"
        CUDA_ARCHS "${FLASH_MLA_ARCHS}")

    set_gencode_flags_for_srcs(
        SRCS "${FlashMLA_Extension_SOURCES}"
        CUDA_ARCHS "${FLASH_MLA_ARCHS}")

    define_extension_target(
        _flashmla_C
        DESTINATION vllm
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES ${FlashMLA_SOURCES}
        COMPILE_FLAGS ${VLLM_GPU_FLAGS}
        ARCHITECTURES ${VLLM_GPU_ARCHES}
        INCLUDE_DIRECTORIES ${FlashMLA_INCLUDES}
        USE_SABI 3
        WITH_SOABI)

    # Keep Stable ABI for the module, but *not* for CUDA/C++ files.
    # This prevents Py_LIMITED_API from affecting nvcc and C++ compiles.
    target_compile_options(_flashmla_C PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-UPy_LIMITED_API>
        $<$<COMPILE_LANGUAGE:CXX>:-UPy_LIMITED_API>)

    if(FlashMLA_Extension_SOURCES)
        define_extension_target(
            _flashmla_extension_C
            DESTINATION vllm
            LANGUAGE ${VLLM_GPU_LANG}
            SOURCES ${FlashMLA_Extension_SOURCES}
            COMPILE_FLAGS ${VLLM_FLASHMLA_GPU_FLAGS}
            ARCHITECTURES ${VLLM_GPU_ARCHES}
            INCLUDE_DIRECTORIES ${FlashMLA_Extension_INCLUDES}
            USE_SABI 3
            WITH_SOABI)

        # Keep Stable ABI for the module, but *not* for CUDA/C++ files.
        # This prevents Py_LIMITED_API from affecting nvcc and C++ compiles.
        target_compile_options(_flashmla_extension_C PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-UPy_LIMITED_API>
            $<$<COMPILE_LANGUAGE:CXX>:-UPy_LIMITED_API>)
    else()
        add_custom_target(_flashmla_extension_C)
    endif()

    if(FLASH_MLA_NEEDS_BUILD_MACROS)
        if(FLASH_MLA_ARCHS MATCHES "12\\.0")
            target_compile_definitions(_flashmla_C PRIVATE
                FLASH_MLA_BUILD_SM120 FLASH_MLA_DISABLE_SM100
                FLASH_MLA_DISABLE_SM90)
        elseif(FLASH_MLA_ARCHS MATCHES "10\\.0")
            target_compile_definitions(_flashmla_C PRIVATE
                FLASH_MLA_BUILD_SM100 FLASH_MLA_DISABLE_SM90)
        endif()
        if(EXISTS "${flashmla_SOURCE_DIR}/csrc/msvc_compat.h")
            target_compile_options(_flashmla_C PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-include ${flashmla_SOURCE_DIR}/csrc/msvc_compat.h>)
        endif()
    endif()
else()
    message(STATUS "FlashMLA will not compile: unsupported CUDA architecture ${CUDA_ARCHS}")
    # Create empty targets for setup.py on unsupported systems
    add_custom_target(_flashmla_C)
    add_custom_target(_flashmla_extension_C)
endif()
