cmake_minimum_required(VERSION 3.20)
project(MLIRGymCompilerTests LANGUAGES C CXX)

find_package(MLIR CONFIG REQUIRED)
find_package(LLVM CONFIG REQUIRED)
find_package(GTest REQUIRED)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

set(MLIR_ROOT "/root/compiler_gym/mlir_playground")
set(PROJECT_ROOT "/root/compiler_gym/mlir_playground/install/lib/cmake/MLIRGymCompiler")
set(PASSES_LIB "${MLIR_ROOT}/install/lib")
set(PASSES_DIR "${MLIR_ROOT}/install/include")

message(STATUS "project root: ${PROJECT_ROOT}")
message(STATUS "passes library: ${PASSES_LIB}")
message(STATUS "passes include dir: ${PASSES_DIR}")

if(EXISTS "${PROJECT_ROOT}/MLIRGymCompilerConfig.cmake")
    message(STATUS "Found backend/passes: ${PROJECT_ROOT}")
    include("${PROJECT_ROOT}/MLIRGymCompilerConfig.cmake")
    include_directories(${PASSES_DIR})
    link_directories(${PASSES_LIB})
else()
    message(FATAL_ERROR "MLIRGymCompiler not found in path: ${PROJECT_ROOT}/MLIRGymCompilerConfig.cmake\nAbort")
endif()

add_executable(MLIRGymCompilerTests test_mlir_gym_termination.cpp)

target_link_libraries(MLIRGymCompilerTests
    PRIVATE
    MLIRGymCompiler
    GTest::GTest
    GTest::Main
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRTransforms
)

enable_testing()
add_test(NAME MLIRGymCompilerTests COMMAND MLIRGymCompilerTests)
