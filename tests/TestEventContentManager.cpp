// Mimic "assert always".
#ifdef NDEBUG
#undef NDEBUG
#endif


#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

//#include <gtest/gtest.h>

#include "EventStore.hpp"
#include "StatusCode.hpp"
#include <gtest/gtest.h>

TEST(EventContentMAnager, BasicFunctionality) {
    EventStore store;
    ASSERT_TRUE(store.record(std::make_unique<int>(42), "test_int"));
    
    const int* retrieved = nullptr;
    ASSERT_TRUE(store.retrieve(retrieved, "test_int"));
    ASSERT_EQ(*retrieved, 42);
}


int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
