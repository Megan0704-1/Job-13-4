#include "clang_mlir_toolchain.h"

int ClangCompiler::compile() {
  if (!validateEnv()) return -1;

  std::vector<std::string> args;
  buildBaseCommand(args);
  addIncludes(args);
  addLibraries(args);
  addUserArguments(args);
  addOutput(args);

  return executeCompilation(args);
}

void ClangCompiler::reset() {
  defaultIncludes = {this->config.llvmBuildDir + "/include"};
  defaultLibDirs = {this->config.llvmBuildDir + "/lib"};
  defaultLibs = {"mlir_c_runner_utils", "mlir_runner_utils"};
  clangPath = this->config.llvmBuildDir + "/bin/clang";
}

bool ClangCompiler::validateEnv() {
  if (config.llvmBuildDir.empty()) {
    llvm::outs() << "Warning: LLVM build dir not specified.\nTry reading from "
                    "sys path.\n";
    std::string sysLlvmDir = std::getenv("LLVM_BUILD_DIR");
    if (sysLlvmDir.empty()) {
      llvm::errs() << "Error: LLVM build dir not specified.\n";
      return false;
    }

    config.llvmBuildDir = sysLlvmDir;
    reset();
  }

  if (!llvm::sys::fs::exists(config.inputFile)) {
    llvm::errs() << "Error: Input file not found.\n";
    return false;
  }

  return true;
}

void ClangCompiler::buildBaseCommand(std::vector<std::string>& args) {
  args.push_back(clangPath);
  args.push_back("-O3");
  args.push_back("-x");
  args.push_back("ir");

  if (config.verbose) {
    args.push_back("-v");
  }
}

void ClangCompiler::addIncludes(std::vector<std::string>& args) {
  for (const auto& include : defaultIncludes) {
    args.push_back("-I");
    args.push_back(include);
  }
}

void ClangCompiler::addLibraries(std::vector<std::string>& args) {
  // lib search path
  for (const auto& lib : defaultLibDirs) {
    args.push_back("-L");
    args.push_back(lib);
  }

  // -lmlir_runner_utils
  for (const auto& lib : defaultLibs) {
    args.push_back((config.linkStatic ? "-l:lib" + lib + ".a" : "-l" + lib));
  }

  for (const auto& lib : config.libs) {
    args.push_back("-l" + lib);
  }
}

void ClangCompiler::addUserArguments(std::vector<std::string>& args) {
  for (const auto& arg : config.extraArgs) {
    args.push_back(arg);
  }
}

void ClangCompiler::addOutput(std::vector<std::string>& args) {
  args.push_back("-o");
  args.push_back(config.outputFile);
  args.push_back(config.inputFile);
}

int ClangCompiler::executeCompilation(const std::vector<std::string>& args) {
  std::string errMsg;

  llvm::SmallVector<llvm::StringRef, 4> env;

  if (!config.linkStatic) {
    std::string ldPath = "LD_LIBRARY_PATH=" + defaultLibDirs[0];
    if (const char* curLibs = std::getenv("LD_LIBRARY_PATH")) {
      ldPath += curLibs;
    }
    env.push_back(ldPath);
  }

  std::vector<llvm::StringRef> llvmArgs;
  for (const auto& arg : args)
    llvmArgs.push_back(arg);  // implicitly cast to stringref

  int result = llvm::sys::ExecuteAndWait(
      clangPath, llvmArgs,
      std::nullopt,  // llvm::ArrayRef(env.data(), env.size()),
      {}, 0, 0, &errMsg);

  if (result != 0) {
    llvm::errs() << "Compilation failed: " << errMsg << "\n";
  }

  return result;
}
