## backend/tests
- This folder is used for testing dependency installation using google test

## Run Google test
To test if your env is config correctly, follow the instructions below
```bash
cd /path/to/backend/tests
mkdir -p build
cd build/
cmake .. -DMLIR_DIR=/path/to/build/lib/cmake/mlir
make -j32

ctest
```
If unfortunately, something went wrong when you try to build the tests
check your PATH variable, LD_LIBRARY_PATH, ...
All CMakeLists.txt in this folder depends on how you configure your system path.
[Refer to ~/.bsahrc to see how to expose to mlir-gym backend]
Be careful and good luck!

## Tricky and annoying things to be refactored
- build protobuf from source w/ version 3.21
- build protobuf with shared library support (w/ flag -DBUILD_SHARED_LIBS=ON)
- place protobuf PATH (.../bin), CMAKE_PREFIX_PATH, LD_LIBRARY_PATH before grpc
```bash
# for instance something like this
$ echo $PATH
>> /path/to/protobuf:/path/to/grpc
```
**The reason is that if we use static lib for building the server, we are required to link against a lot of dynamic libs, which can be cumbersome**

### Folder structure
./protobuf_test
./bazel_test
./llvm_test
./mlir_test
./grpc_test

### Knowledge of CMake (personal note)
- How does CMake find the libraries?
> by `find_package(PACKAGE_NAME REQUIRED CONFIG)`, we are essentially telling cmake to locate PACKAGE_NAME
> If a package is installed correctly, and support CMake build, CMake will look for a file like
> - PACKAGE_NAME/.../lib/cmake/.../xxxConfig.cmake
> This config.cmake file will contains predefined varaibles like xxx_INCLUDE_DIRS, xxx_LIBRARY_DIRS, xxx_DEFINITIONS, xxx_AVAILABLE_LIBS (list of available package libraries)

- So `find_pacakge(PACKAGE_NAME)` automatically sets xxx_INCLUDE_DIRS, xxx_LIBRARY_DIRS, then we'll tell cmake where to find the headers by the lines
```cmake
include_directories(${xxx_INCLUDE_DIRS})
link_directories(${xxx_LIBRARY_DIRS})
```

- Then when we write `target_link_libraries(xxx_exe PRIVATE r1 r2)`, cmake search for ${xxx_LIBRARY_DIRS}/libr1.so (or .a) and links them against xxx_exe
