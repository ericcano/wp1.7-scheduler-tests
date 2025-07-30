#include "NewScheduler.hpp"
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
    
    ASSERT_EQ(sched.m_algorithms.size(), 5);
}

TEST(NewSchedulerTest, initSchedulerState) {
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
    // sched.initSchedulerState(3); // 3 slots
    // ASSERT_EQ(sched.maxConcurrentEvents, 3);
    // ASSERT_EQ(sched.eventSlots.size(), 3);
    // for (int i = 0; i < 3; ++i) {
    //     ASSERT_EQ(sched.eventSlots[i].slots.size(), 1);
    //     auto& slot = sched.eventSlots[i].slots[0];
    //     ASSERT_EQ(slot.algorithms.size(), 5);
    //     ASSERT_EQ(slot.eventNumber, i);
    // }
}


