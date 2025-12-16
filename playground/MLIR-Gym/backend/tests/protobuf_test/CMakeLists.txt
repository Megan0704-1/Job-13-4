cmake_minimum_required(VERSION 3.20)
project(ProtobufUnitTest LANGUAGES CXX)

# Find Protobuf
find_package(Protobuf REQUIRED)

# Find Google Test
find_package(GTest REQUIRED)

set(CPP_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp_out")

file(MAKE_DIRECTORY ${CPP_OUT_DIR})

# Generate Protobuf files
set(PROTO_FILE test.proto)
add_custom_command(
    OUTPUT
        ${CPP_OUT_DIR}/test.pb.cc
        ${CPP_OUT_DIR}/test.pb.h
    COMMAND
        ${Protobuf_PROTOC_EXECUTABLE}
        ARGS -I ${CMAKE_CURRENT_SOURCE_DIR}
        --cpp_out=${CPP_OUT_DIR}
        ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating Protobug C++ sources from ${PROTO_FILE}"
    VERBATIM
)

add_library(test_lib STATIC ${CPP_OUT_DIR}/test.pb.cc)

target_include_directories(test_lib PUBLIC ${CPP_OUT_DIR} ${Protobuf_INCLUDE_DIRS})

target_link_libraries(test_lib PUBLIC
  protobuf::libprotobuf
  GTest::GTest
  GTest::Main
)

# Add the test executable
add_executable(test_protobuf test_protobuf.cpp)
target_link_libraries(test_protobuf PUBLIC test_lib)

# Add a test
enable_testing()
add_test(NAME test_protobuf COMMAND test_protobuf)
