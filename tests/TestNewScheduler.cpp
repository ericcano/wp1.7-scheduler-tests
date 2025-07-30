#include "Scheduler.hpp"
#include "gtest/gtest.h"
#include "MockAlgorithm.hpp"

TEST(NewSchedulerTest, RegisterFiveAlgorithms) {
    NewScheduler sched;
    MockAlgorithm algA{{}, {"prodA"}};
    MockAlgorithm algB{{"prodA"}, {"prodB"}};
    MockAlgorithm algC{{"prodB"}, {"prodC"}};
    MockAlgorithm algD{{"prodC"}, {"prodD"}};
    MockAlgorithm algE{{"prodD"}, {"prodE"}};
    sched.addAlgorithm(algA);
    sched.addAlgorithm(algB);
    sched.addAlgorithm(algC);
    sched.addAlgorithm(algD);
    sched.addAlgorithm(algE);
    // Check that 5 algorithms are registered
    
    ASSERT_EQ(sched.algorithms.size(), 5);
}

