#ifndef CLANG_MLIR_TOOLCHAIN_H
#define CLANG_MLIR_TOOLCHAIN_H

#include <llvm/Support/raw_ostream.h>

#include <string>
#include <vector>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"

struct CompilerConfig {
  std::string llvmBuildDir;
  std::string inputFile;
  std::string outputFile;
  bool verbose = false;
  bool linkStatic = false;
  std::vector<std::string> extraArgs;
  std::vector<std::string> libs;
};

class ClangCompiler {
 public:
  ClangCompiler(CompilerConfig config) : config(std::move(config)) { reset(); }

  int compile();

 private:
  CompilerConfig config;
  std::vector<std::string> defaultIncludes;
  std::vector<std::string> defaultLibDirs;
  std::vector<std::string> defaultLibs;
  std::string clangPath;

  void reset();
  bool validateEnv();
  void buildBaseCommand(std::vector<std::string>& args);
  void addIncludes(std::vector<std::string>& args);
  void addLibraries(std::vector<std::string>& args);
  void addUserArguments(std::vector<std::string>& args);
  void addOutput(std::vector<std::string>& args);
  int executeCompilation(const std::vector<std::string>& args);
};

#endif
