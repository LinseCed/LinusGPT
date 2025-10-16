cmake_minimum_required(VERSION 3.18)
project(llm LANGUAGES CXX CUDA)

# === Options ===
option(USE_CUDA "Enable CUDA support" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# === Build type (default: Release) ===
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug or Release)" FORCE)
endif()

# === Include directory ===
include_directories(${PROJECT_SOURCE_DIR}/include)

# === Source discovery ===
file(GLOB_RECURSE SRC_CPP CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE SRC_CU  CONFIGURE_DEPENDS src/*.cu)

set(SOURCES ${SRC_CPP})

# === Try to find CUDA if enabled ===
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
        add_definitions(-DUSE_CUDA)
        list(APPEND SOURCES ${SRC_CU})
    else()
        message(WARNING "CUDA not found! Building CPU-only version.")
        set(USE_CUDA OFF)
    endif()
endif()

# === Target ===
add_executable(llm ${SOURCES})

# === Compiler flags ===
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(llm PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -g -O0>
        $<$<COMPILE_LANGUAGE:CUDA>:-G -g -O0>
    )
else()
    target_compile_options(llm PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -s>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-s>
    )
endif()

# === CUDA-specific settings ===
if(USE_CUDA)
    set_target_properties(llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# === Output directory ===
set_target_properties(llm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

