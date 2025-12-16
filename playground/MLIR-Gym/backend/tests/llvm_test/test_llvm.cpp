#include <gtest/gtest.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

/**
 * gtest/gtest.h: GoogleTest framework
 * IR/LLVMContext.h: LLVm Context, manages IR objects
 * IR/Module.h: LLVm Module, the top level IR container
 * IR/IRBuilder.h: IRBuilder, constructs IR
 * Support/raw_ostream: for printing llvm ir
 */

TEST(LLVMTest, CreateModule) {
  // llvm object:
  // context -> global ir context for ir object
  // module -> compilation unit, module repr entire llvm ir program
  // builder -> utility for constructing llvm ir instruction

  llvm::LLVMContext ctx;
  llvm::Module module("test_module", ctx);
  llvm::IRBuilder<> builder(ctx);

  // function prototype obj: name it 'main'
  llvm::FunctionType* funcType =
      llvm::FunctionType::get(builder.getInt32Ty(), false);
  llvm::Function* func = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, "main", module);

  // Add bb and insertion points
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", func);
  builder.SetInsertPoint(entry);

  // create ret i32 0
  builder.CreateRet(builder.getInt32(0));

  // gtest verifier with llvm module
  EXPECT_FALSE(llvm::verifyModule(module, &llvm::errs()));

  module.print(llvm::outs(), nullptr);
  /*
   * expected output
   * ; ModuleID = 'test_module'
   * source_filename = "test_module"
   * define i32 @main() {
   * entry:
   *     ret i32 0
   * }
   */
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
