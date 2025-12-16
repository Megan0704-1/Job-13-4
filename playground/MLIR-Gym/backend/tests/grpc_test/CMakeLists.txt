cmake_minimum_required(VERSION 3.20)
project(MinimalGrpcTest LANGUAGES C CXX)

# Make sure to use at least C++11/17 or whichever you need
set(CMAKE_CXX_STANDARD 17)

# Find your dependencies. Adjust these if your installs differ:
find_package(absl REQUIRED)

# static library issue, force to use shared files
set(Protobuf_USE_STATIC_LIBS OFF)
find_package(Protobuf REQUIRED)
find_package(GTest REQUIRED)
find_package(gRPC REQUIRED CONFIG)

# The .proto file is in the same folder as this CMakeLists
set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/test.proto")
set(SERVER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/test_server.cpp")
set(CPP_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/cpp_out")
set(GRPC_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc_out")

file(MAKE_DIRECTORY ${CPP_OUT_DIR})
file(MAKE_DIRECTORY ${GRPC_OUT_DIR})

# !!) First ensure grpc_cpp_plugin is in your PATH env var
find_program(gRPC_CPP_PLUGIN NAMES grpc_cpp_plugin)

if(NOT gRPC_CPP_PLUGIN)
    message(FATAL_ERROR "grpc_cpp_plugin not found! Ensure coorect installation of protobuf compiler.")
endif()

message(STATUS "Using grpc plugin: ${gRPC_CPP_PLUGIN}")

# 1) Generate code with protoc:
add_custom_command(
    OUTPUT
        "${CPP_OUT_DIR}/test.pb.cc"
        "${CPP_OUT_DIR}/test.pb.h"
        "${GRPC_OUT_DIR}/test.grpc.pb.cc"
        "${GRPC_OUT_DIR}/test.grpc.pb.h"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CPP_OUT_DIR} ${GRPC_OUT_DIR}
    COMMAND
        "${Protobuf_PROTOC_EXECUTABLE}"
        ARGS -I ${CMAKE_CURRENT_SOURCE_DIR}
        --cpp_out=${CPP_OUT_DIR}
        --grpc_out=${GRPC_OUT_DIR}
        --plugin=protoc-gen-grpc=${gRPC_CPP_PLUGIN}
        "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
    COMMENT "Generating Protobuf & gRPC C++ sources from ${PROTO_FILE}"
    VERBATIM
)

message(STATUS "grpc: ${gRPC_CPP_PLUGIN}")

add_library(server_lib STATIC
    ${CPP_OUT_DIR}/test.pb.cc
    ${GRPC_OUT_DIR}/test.grpc.pb.cc
)

target_include_directories(server_lib PUBLIC
    "${CPP_OUT_DIR}"
    "${GRPC_OUT_DIR}"
    ${Protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS}
)

target_link_libraries(server_lib PUBLIC
    protobuf::libprotobuf
    gRPC::grpc++
    absl::base
    absl::core_headers
    absl::strings
    absl::log_internal_check_op
    GTest::gtest_main
)

# server test executable
add_executable(server_test ${SERVER_FILE})
target_include_directories(server_test PUBLIC ${CPP_OUT_DIR} ${GRPC_OUT_DIR})
target_link_libraries(server_test PUBLIC server_lib)

enable_testing()
add_test(NAME test_grpc COMMAND server_test)
