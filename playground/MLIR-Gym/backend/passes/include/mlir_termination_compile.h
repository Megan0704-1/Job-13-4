#ifndef MLIR_TERMINATION_COMPILE_H
#define MLIR_TERMINATION_COMPILE_H

#pragma once
#include <chrono>
#include <fstream>
#include <string>
#include <system_error>

#include "clang_mlir_toolchain.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

enum class CompileErrorTypes {
  NoError,
  ParsingError,
  LoweringError,
  TranslationError,
  FileIOError,
  DependencyNotFoundError, /*raise when clang not found in sys path*/
  CompilationError,        /*clang compilation error*/
  ExecutionTimeError,      /*generate crashed binary*/
  BinaryTypeError,         /*binary is un-executable*/
  AccuracyError,           /*output mismatch*/
};

struct CompileResult {
  bool success;
  CompileErrorTypes errorType;
  std::string errorMessage;
  std::string binary;
  uint64_t binarySize;
  double executionTime;
  bool sameOutput = false;
};

class MLIRGymCompiler {
 public:
  // Constructor
  MLIRGymCompiler(const std::string &inputIR, const std::string &output,
                  const std::string llvmBuildDir = "")
      : inputIR(inputIR),
        output(output),
        llvmBuildDir(llvmBuildDir),
        config(getConfig(llvmBuildDir, inputIR, output)),
        clangBin(config) {
    llvmIRFilename = output + ".ll";
  }

  // Runs the full compilation pipeline to binary
  CompileResult compile();

  // stage 1
  bool parseModule();
  // stage 2
  bool lowerModule();
  // stage 3
  bool translateModule();
  // stage 4
  bool writeLLVMIR();
  // stage 5
  bool compileToBin();
  // stage 6
  bool getBinSize();
  // stage 7
  bool execute();

 private:
  void returnError(CompileResult &result, CompileErrorTypes errType);

  // input params
  std::string inputIR;
  std::string output;
  std::string llvmBuildDir;
  std::string llvmIRFilename;

  // internal state
  CompilerConfig config;
  ClangCompiler clangBin;
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> mlirModule;
  llvm::LLVMContext llvmCtx;
  std::unique_ptr<llvm::Module> llvmModule;
  std::string errMsg;
  double executionTime = 0.0;
  uint64_t binarySize = 0;

  // clang config method
  static CompilerConfig getConfig(std::string, std::string, std::string);
};

#endif  // MLIR_TERMINATION_COMPILE_H
