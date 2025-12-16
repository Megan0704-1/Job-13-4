#include <memory>
#include <sstream>
#include <string>
#include <vector>

// MLIR header files
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"

// gRPC header files
#include <grpcpp/grpcpp.h>

#include "compiler_service.grpc.pb.h"
