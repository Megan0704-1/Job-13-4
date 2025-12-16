cmake_minimum_required(VERSION 3.20)
project(MLIRService LANGUAGES C CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set MLIR and LLVM default fall back path
set(LLVM_PROJECT_ROOT "/root/llvm-project/build/lib/cmake/llvm")
set(MLIR_PROJECT_ROOT "/root/llvm-project/build/lib/cmake/mlir")

# prompt the user to set config.cmake path
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
link_directories(${MLIR_LIBRARY_DIR} ${LLVM_LIBRARY_DIR})

set(
    MLIR_LIBS
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRSupport
    )

# append module list to find grpc packages
include(FetchContent)

FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc.git
  GIT_TAG        v1.55.x  # or whatever version
)
FetchContent_MakeAvailable(gRPC)
find_package(gRPC REQUIRED CONFIG)

find_package(Protobuf REQUIRED)

set(PROTO_SRC service/compiler_service.pb.cc service/compiler_service.grpc.pb.cc)
set(PROTO_HDR service/compiler_service.pb.h service/compiler_service.grpc.pb.h)

add_executable(mlir_server server/mlir_opt_service_impl.cpp ${PROTO_SRC})

target_include_directories(mlir_server PRIVATE ${Protobuf_INCLUDE_DIRS} ${gRPC_INCLUDE_DIRS})
target_link_libraries(mlir_server PRIVATE ${MLIR_LIBS} ${Protobuf_LIBRARIES} grpc::grpc++)

set_target_properties(mlir_server PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${MLIR_LIBRARY_DIR}"
)

message(STATUS "MLIR Include Directories: ${MLIR_INCLUDE_DIRS}")
message(STATUS "MLIR Libraries: ${MLIR_LIBS}")
