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
    sched.initSchedulerState(); // 
    ASSERT_EQ(sched.m_algorithms.size(), 5);
    ASSERT_EQ(sched.m_eventSlotsNumber, 4); // Should be the default (4)
    for (int i = 0; i < sched.m_eventSlotsNumber; ++i) {
        auto & evSlot = sched.m_eventSlots[i];
        ASSERT_EQ(evSlot.algorithms.size(), 5);
        // At initialization, eventNumber should be equal to slot index
        ASSERT_EQ(evSlot.eventNumber, i);
    }
}


