#include <gtest/gtest.h>

#include "test.pb.h"

TEST(ProtobufTest, BasicTest) {
  TestMessage msg;
  msg.set_text("Hello, Protobuf!");
  msg.set_number(123);

  EXPECT_EQ(msg.text(), "Hello, Protobuf!");
  EXPECT_EQ(msg.number(), 123);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
