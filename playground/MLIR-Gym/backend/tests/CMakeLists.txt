cmake_minimum_required(VERSION 3.20)
project(MLIRGymBackendTests LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS "out dir: ${CMAKE_RUNTIME_OUTPUT_DIR}")

enable_testing()

# subdir for each dependecy tests
add_subdirectory(protobuf_test)
add_subdirectory(llvm_test)
add_subdirectory(mlir_test)
add_subdirectory(grpc_test)
