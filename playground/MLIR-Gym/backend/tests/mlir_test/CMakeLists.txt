# tests/mlir_test/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(MLIRTest LANGUAGES C CXX)

# Find MLIR
find_package(MLIR REQUIRED CONFIG)

# Find LLVM
find_package(LLVM REQUIRED CONFIG)

# Find GoogleTest
find_package(GTest REQUIRED)

# Find Threads
find_package(Threads REQUIRED)

# Add MLIR and LLVM Include Directories
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

# Link MLIR and LLVM Libraries
link_directories(${MLIR_LIBRARY_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

add_executable(test_mlir test_mlir.cpp)

# Ensure MLIR Libraries Are Linked
# /root/llvm-project/build/lib/*.a
target_link_libraries(test_mlir PRIVATE
    GTest::gtest
    GTest::gtest_main
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
    MLIRAffineDialect
    MLIRArithDialect
    MLIRLLVMDialect
    Threads::Threads
    ${LLVM_LIBRARIES}
)

# Enable Testing
enable_testing()
add_test(NAME test_mlir COMMAND test_mlir)
