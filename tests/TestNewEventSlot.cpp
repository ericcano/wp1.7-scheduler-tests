#include <gtest/gtest.h>

#include "NewScheduler.hpp"

TEST(NewEventSlotTest, Basics) {
    NewScheduler::NewEventSlot evSlot;
    ASSERT_EQ(evSlot.eventNumber, -1); // Default event number
    ASSERT_TRUE(evSlot.algorithms.empty()); // No algorithms by default
    ASSERT_EQ(evSlot.stream, (cudaStream_t)cudaStreamDefault); // Stream should be initialized
    // Mutex cannot be directly tested, but we can check that it exists
    ASSERT_NO_THROW(evSlot.schedulingMutex.lock());
    evSlot.schedulingMutex.unlock();
    // EventContentManager should be empty initially
    // evSlot.eventContentManager.dumpContents(); // Just to ensure it can be called 
}

TEST(NewEventSlotTest, Vector) {
    std::vector<NewScheduler::NewEventSlot> evSlotVec;
    new (&evSlotVec) std::vector<NewScheduler::NewEventSlot>(3);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
