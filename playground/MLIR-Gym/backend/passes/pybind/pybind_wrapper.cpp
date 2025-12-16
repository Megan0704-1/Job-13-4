#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir_termination_compile.h"

namespace py = pybind11;

PYBIND11_MODULE(mlir_gym_compiler, m) {
  py::class_<CompileResult>(m, "CompileResult")
      .def_readonly("success", &CompileResult::success)
      .def_readonly("error_type", &CompileResult::errorType)
      .def_readonly("error_message", &CompileResult::errorMessage)
      .def_readonly("binary_path", &CompileResult::binary)
      .def_readonly("binary_size", &CompileResult::binarySize)
      .def_readonly("execution_time", &CompileResult::executionTime)
      .def_readwrite("same_output", &CompileResult::sameOutput);

  py::enum_<CompileErrorTypes>(m, "CompileErrorTypes")
      .value("NoError", CompileErrorTypes::NoError)
      .value("ParsingError", CompileErrorTypes::ParsingError)
      .value("TranslationError", CompileErrorTypes::TranslationError)
      .value("FileIOError", CompileErrorTypes::FileIOError)
      .value("CompilationError", CompileErrorTypes::CompilationError)
      .value("BinaryTypeError", CompileErrorTypes::BinaryTypeError)
      .value("ExecutionTimeError", CompileErrorTypes::ExecutionTimeError)
      .value("AccuracyError", CompileErrorTypes::AccuracyError);

  py::class_<MLIRGymCompiler>(m, "MLIRGymCompiler")
      .def(
          py::init<const std::string&, const std::string&, const std::string>(),
          py::arg("input_ir"), py::arg("output_path"),
          py::arg("llvm_build_dir") =
              "/home/megankuo/workspace/install/llvm-20")
      .def("compile", &MLIRGymCompiler::compile,
           "Compile the given ir to llvm ir and binary.")
      .def("parse_module", &MLIRGymCompiler::parseModule,
           "parse the input mlir.")
      .def("lower_module", &MLIRGymCompiler::lowerModule,
           "lower the mlir module to llvm-mlir, should run after parse_module "
           "method.");
}
