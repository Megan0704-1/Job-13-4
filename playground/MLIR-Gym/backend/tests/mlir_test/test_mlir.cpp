#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Support/LogicalResult.h>

/**
 * gtest.h: Google Test framework
 * MLIRContext.h: manage MLIR type systems
 * BuiltinOps.h: Built-in operations
 * Verifier.h: mlir module verifier
 * LogicalResult.h: logical result utility
 * InitAllDialects.h: register all mlir default dialects
 * InitAllPasses.h: register all mlir default passes
 * raw_ostream.h: print utility
 */

TEST(MLIRTest, CreateModule) {
  // context: mlir global env to work with
  mlir::MLIRContext ctx;
  ctx.loadDialect<mlir::LLVM::LLVMDialect>();
  ctx.loadDialect<mlir::arith::ArithDialect>();
  ctx.loadDialect<mlir::affine::AffineDialect>();

  // mlir module -> container for mlir operations, uses unknown loc to assign a
  // default location for ir
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  EXPECT_TRUE(succeeded(mlir::verify(*module)));

  module->print(llvm::outs());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
