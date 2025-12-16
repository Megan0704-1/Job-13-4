#include <gtest/gtest.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

#include "mlir_termination_compile.h"

class MLIRGymCompilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    llvm::SmallString<128> tmpDir;
    llvm::sys::fs::createUniqueDirectory("mlir-gym-compiler-test", tmpDir);
    testDir = tmpDir.str().str();
  }

  void TearDown() override { llvm::sys::fs::remove_directories(testDir); }

 public:
  std::string testDir;
};

/**
 * @brief Test valid compilation flow
 * @description 1. parse the mlir module; 2. lower to llvm ir; 3. write to
 * file; 4. compile with clang
 */
TEST_F(MLIRGymCompilerTest, ValidMLIRCompilation) {
  const std::string mlirModuleString = R"mlir(
        module attributes {"test.name" = "dummy"} {
            llvm.func @main() -> i32 {
                %0 = llvm.mlir.constant(0: i32): i32
                llvm.return %0: i32
            }
        }
        )mlir";

  CompileResult result =
      MLIRGymCompiler(mlirModuleString, testDir + "/valid_output").compile();

  EXPECT_EQ(result.errorType, CompileErrorTypes::NoError);
  EXPECT_TRUE(result.success);
  EXPECT_TRUE(llvm::sys::fs::exists(result.binary));
}

/**
 * @brief This tests the parsing logic of the termination pass
 * @description expect to have parsing err in their compile result messagge
 */
TEST_F(MLIRGymCompilerTest, InvalidMLIRSyntax) {
  const std::string invalidMLIRStirng = "module {invalid_operation}";
  CompileResult result =
      MLIRGymCompiler(invalidMLIRStirng, testDir + "/invalid_syntax").compile();
  EXPECT_FALSE(result.success);
  EXPECT_NE(result.errorMessage.find("parse"), std::string::npos);
}

/**
 * @brief This tests if the give mlir is valid in terms of dialect, but the
 * dialect is not LLVM.
 * @description expect to have parseing err in their compile result message.
 */
TEST_F(MLIRGymCompilerTest, NonSupportedMLIR) {
  const std::string nonLLVMMLIR = R"mlir(
        module {
            func.func @main() -> i32 {
                %0 = arith.constant 17: i32
                return %0
            }
        }
    )mlir";

  CompileResult result =
      MLIRGymCompiler(nonLLVMMLIR, testDir + "/invalid_dialect").compile();
  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.errorType, CompileErrorTypes::ParsingError);
}

/**
 * @brief This tests the invalid llvm ir semantic
 * @description Expects to ahve parsing err
 */
TEST_F(MLIRGymCompilerTest, InvalidLLVMSyntax) {
  const std::string invalidLLVMMLIR = R"mlir(
        module {
            llvm.func @main() -> i32 {
                %0 = llvm.mlir.constant(17: i32): i32
                llvm.return %1: i32
            }
        }
    )mlir";

  CompileResult result =
      MLIRGymCompiler(invalidLLVMMLIR, testDir + "/invalid_llvm_mlir")
          .compile();
  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.errorType, CompileErrorTypes::ParsingError);
}

/**
 * @brief This tests the clang compile for no main module
 * @description Expects to have clang err
 */
TEST_F(MLIRGymCompilerTest, MLIRClangError) {
  const std::string noMainMLIR = R"mlir(
        module {
        }
    )mlir";

  CompileResult result =
      MLIRGymCompiler(noMainMLIR, testDir + "/clang_error").compile();
  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.errorType, CompileErrorTypes::CompilationError);
}

/**
 * @brief This tests the write to file condition in mlir_termination_compile.cpp
 * @description Expects to have FileIO err
 */
TEST_F(MLIRGymCompilerTest, FileIOError) {
  const std::string MLIRFileIO = R"mlir(
        module {
        llvm.func @main() {}
        }
    )mlir";

  CompileResult result =
      MLIRGymCompiler(MLIRFileIO, testDir + "/make_up_name" + "/file_io_error")
          .compile();
  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.errorType, CompileErrorTypes::FileIOError);
}
