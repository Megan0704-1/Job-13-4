#include "mlir_termination_compile.h"

CompilerConfig MLIRGymCompiler::getConfig(std::string llvmBuildDir,
                                          std::string inputIR,
                                          std::string output) {
  CompilerConfig config{.llvmBuildDir = llvmBuildDir,
                        .inputFile = output + ".ll",
                        .outputFile = output,
                        .verbose = true,
                        .extraArgs = {"-Wall", "-Wextra"},
                        .libs = {"m"}};
  return config;
}

bool MLIRGymCompiler::parseModule() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  ctx.appendDialectRegistry(registry);

  mlirModule = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
  if (!mlirModule) {
    errMsg = "Failed to parse MLIR.";
    return false;
  }
  return true;
}

bool MLIRGymCompiler::lowerModule() {
  mlir::PassManager pm(&ctx);
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (mlir::failed(pm.run(*mlirModule))) {
    errMsg = "Failed to lower to llvm-mlir";
    return false;
  }

  return true;
}

bool MLIRGymCompiler::translateModule() {
  mlir::registerBuiltinDialectTranslation(*mlirModule->getContext());
  mlir::registerLLVMDialectTranslation(*mlirModule->getContext());

  llvmModule = mlir::translateModuleToLLVMIR(*mlirModule, llvmCtx);
  if (!llvmModule) {
    errMsg = "Failed to translate llvm-mlir to llvm-ir.";
    return false;
  }
  return true;
}

bool MLIRGymCompiler::writeLLVMIR() {
  std::error_code ec;
  llvm::raw_fd_ostream llvmStream(llvmIRFilename, ec);
  if (ec) {
    errMsg = "Failed IO for writing LLVMIR: " + ec.message();
    return false;
  }

  llvmModule->print(llvmStream, nullptr);
  llvmStream.close();
  return true;
}

bool MLIRGymCompiler::compileToBin() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  int clangErrCode = clangBin.compile();

  if (clangErrCode != 0) {
    errMsg = "Clang failed: " + errMsg;
    return false;
  }

  return true;
}

bool MLIRGymCompiler::getBinSize() {
  std::error_code ec;
  llvm::sys::fs::file_size(output, binarySize);

  if (ec) {
    errMsg = "Failed to get size of the binary: " + ec.message();
    return false;
  }

  return true;
}

bool MLIRGymCompiler::execute() {
  auto start_time = std::chrono::steady_clock::now();
  int retErrCode =
      llvm::sys::ExecuteAndWait(output, {}, std::nullopt, {}, 0, 0, &errMsg);
  auto end_time = std::chrono::steady_clock::now();

  if (!errMsg.empty()) {
    errMsg = "Execution failed: " + errMsg;
    return false;
  }

  std::chrono::microseconds elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time);
  executionTime = static_cast<double>(elapsed.count());

  return true;
}

void MLIRGymCompiler::returnError(CompileResult &result,
                                  CompileErrorTypes errType) {
  result.success = false;
  result.errorType = errType;
  result.errorMessage = errMsg;
}

CompileResult MLIRGymCompiler::compile() {
  CompileResult result;

  if (!parseModule()) {
    returnError(result, CompileErrorTypes::ParsingError);
    return result;
  }
  if (!lowerModule()) {
    returnError(result, CompileErrorTypes::LoweringError);
    return result;
  }
  if (!translateModule()) {
    returnError(result, CompileErrorTypes::TranslationError);
    return result;
  }
  if (!writeLLVMIR()) {
    returnError(result, CompileErrorTypes::FileIOError);
    return result;
  }
  if (!compileToBin()) {
    returnError(result, CompileErrorTypes::CompilationError);
    return result;
  }
  if (!getBinSize()) {
    returnError(result, CompileErrorTypes::BinaryTypeError);
    return result;
  }
  if (!execute()) {
    returnError(result, CompileErrorTypes::ExecutionTimeError);
    return result;
  }

  result.success = true;
  result.errorType = CompileErrorTypes::NoError;
  result.errorMessage = "";
  result.binary = output;
  result.binarySize = binarySize;
  result.executionTime = executionTime;
  return result;
}
