#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

// Include the generated headers (names match what we'll generate below)
#include "test.grpc.pb.h"
#include "test.pb.h"

using myproto::msgRequest;
using myproto::msgResponse;
using myproto::TestService;

// Implementation of the service
class TestServiceImpl final : public TestService::Service {
 public:
  grpc::Status test(grpc::ServerContext* ctx, const msgRequest* req,
                    msgResponse* reply) override {
    // Just greet the name passed in
    reply->set_greeting("Hello, " + req->name() + "!");
    return grpc::Status::OK;
  }
};

TEST(GrpcExampleTest, BasicClientServer) {
  // 1) Start a gRPC server on localhost:50051
  TestServiceImpl service;
  std::string server_address("127.0.0.1:50051");

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  // Build and start
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  ASSERT_TRUE(server != nullptr) << "Server failed to start";

  // 2) Create a client stub that calls the server
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  std::unique_ptr<TestService::Stub> stub = TestService::NewStub(channel);

  // 3) Prepare request/response
  msgRequest request;
  request.set_name("Alice");
  msgResponse response;
  grpc::ClientContext context;

  // 4) Call the RPC
  grpc::Status status = stub->test(&context, request, &response);

  // 5) Check results
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(response.greeting(), "Hello, Alice!");

  // 6) Shut down server
  server->Shutdown();
}

// Typical gtest main()
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
