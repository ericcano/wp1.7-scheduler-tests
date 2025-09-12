#include "NewScheduler.hpp"
#include "gtest/gtest.h"
#include "MockAlgorithm.hpp"
#include <ranges>

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

TEST(NewSchedulerTest, NewEventSlot) {
    MockAlgorithm algA{{}, {"prodA"}};
    std::vector<std::reference_wrapper<AlgorithmBase>> algorithms{{algA}};
    NewAlgoDependencyMap depMap{algorithms};
    NewScheduler::NewEventSlot evSlot;
    evSlot.initialize(depMap, -1);
    ASSERT_EQ(evSlot.eventNumber, -1); // Default event number
    // evSlot.eventContentManager.dumpContents(depMap);
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
    sched.initSchedulerState(); // Initialize the scheduler state
    ASSERT_EQ(sched.m_algorithms.size(), 5);
    ASSERT_EQ(sched.m_eventSlotsNumber, 4); // Should be the default (4)
    ASSERT_EQ(sched.m_eventSlots.size(), 4); // Should be the default (4)
    for (auto i: std::ranges::iota_view(0, sched.m_eventSlotsNumber)) {
        auto & evSlot = sched.m_eventSlots[i];
        ASSERT_EQ(evSlot.algorithms.size(), 5);
        // At initialization, eventNumber should be equal to slot index
        ASSERT_EQ(evSlot.eventNumber, i);
    }
}


