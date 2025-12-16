cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MLIRGymCompiler VERSION 1.0 LANGUAGES C CXX)

include(CMakePackageConfigHelpers)

set(LLVM_DIR "" CACHE PATH "Path to LLVM's CMake package dir (…/lib/cmake/llvm)")
set(MLIR_DIR "" CACHE PATH "Path to MLIR's CMake package dir (…/lib/cmake/mlir)")

message(STATUS "mlir dir: ${MLIR_DIR}")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(Threads REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB LIB_SOURCES "${CMAKE_SOURCE_DIR}/lib/*.cpp")
file(GLOB PY_SOURCES "${CMAKE_SOURCE_DIR}/pybind/*.cpp")

add_library(MLIRGymCompiler SHARED ${LIB_SOURCES})

target_link_libraries(MLIRGymCompiler
    PRIVATE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRLLVMDialect
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    MLIRTransforms
    MLIRReconcileUnrealizedCasts
    LLVMCore
    LLVMSupport
    LLVMX86CodeGen
    LLVMDemangle
    LLVMX86AsmParser
)

llvm_map_components_to_libnames(llvm_libs Core Support)
target_link_libraries(MLIRGymCompiler PRIVATE ${llvm_libs})

# for pybind11
add_library(mlir_gym_compiler MODULE "${PY_SOURCES}")
target_link_libraries(mlir_gym_compiler
    PRIVATE
    pybind11::module
    MLIRGymCompiler
)
set_target_properties(mlir_gym_compiler
    PROPERTIES
    SUFFIX ".so"
    PREFIX ""
    CXX_VISIBILITY_PRESET "hidden"
)

# generate config.cmake
configure_package_config_file(
    cmake/MLIRGymCompilerConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MLIRGymCompilerConfig.cmake
    INSTALL_DESTINATION lib/cmake/MLIRGymCompiler
)

# version file
write_basic_package_version_file(
    MLIRGymCompilerConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# build tree target
export(
    TARGETS MLIRGymCompiler
    FILE "${CMAKE_CURRENT_BINARY_DIR}/MLIRGymCompilerTargets.cmake"
    NAMESPACE MLIRGymCompiler::
)

# install targets and headers
install(TARGETS MLIRGymCompiler
    EXPORT MLIRGymCompilerTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

# install the exported set
install(EXPORT MLIRGymCompilerTargets
    FILE MLIRGymCompilerTargets.cmake
    NAMESPACE MLIRGymCompiler::
    DESTINATION lib/cmake/MLIRGymCompiler
)

# install generated config and version files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MLIRGymCompilerConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MLIRGymCompilerConfigVersion.cmake
    DESTINATION lib/cmake/MLIRGymCompiler
)

install(TARGETS
    mlir_gym_compiler
    LIBRARY DESTINATION lib/python
)

message(STATUS "install MLIRGymCompiler shared library to lib/ and include/")
