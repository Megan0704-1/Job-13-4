# tests/llvm_test/CMakeLists.txt
project(LLVMTest LANGUAGES C CXX)

# find llvm
find_package(LLVM REQUIRED CONFIG)

# find google test
find_package(GTest REQUIRED)

# fine threads
find_package(Threads REQUIRED)

# include hdrs and libs
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

add_executable(test_llvm test_llvm.cpp)

# link llvm core libs
target_link_libraries(test_llvm PRIVATE
    GTest::GTest
    GTest::Main
    LLVMCore
    LLVMSupport
    Threads::Threads
)

enable_testing()

# add to ctest
add_test(NAME test_llvm COMMAND test_llvm)
