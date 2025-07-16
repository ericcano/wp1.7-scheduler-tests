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

#include <gtest/gtest.h>

#include "AlgorithmBase.hpp"
#include "EventContentManager.hpp"
#include "EventContext.hpp"
#include "EventStore.hpp"
#include "StatusCode.hpp"

class MockAlgorithm : public AlgorithmBase {
public:
    MockAlgorithm(const std::vector<std::string>& dependencies,
                 const std::vector<std::string>& products) {
        for (const auto& dep : dependencies) {
            auto s = addDependency<int>(dep);
            assert(s);
        }
        for (const auto& prod : products) {
            auto s = addProduct<int>(prod);
            assert(s);
        }
    }
    StatusCode initialize() override { return StatusCode::SUCCESS; }
    AlgCoInterface execute(EventContext) const override { return {}; }
    StatusCode finalize() override { return StatusCode::SUCCESS; }    
};

class EventContentManagerTest : public ::testing::Test {
protected:
    // Algorithms forming a chain: algA -> algB -> algC
    MockAlgorithm algA{{}, {"prodA"}};
    MockAlgorithm algB{{"prodA"}, {"prodB"}};
    MockAlgorithm algC{{"prodB"}, {"prodC"}};
    std::vector<std::reference_wrapper<AlgorithmBase>> chainAlgs{algA, algB, algC};
    EventContentManager manager{chainAlgs};
    void SetUp() override {
        // Any setup code can go here if needed.
    }
    void TearDown() override {
        // Any cleanup code can go here if needed.
    }
    
};

TEST_F(EventContentManagerTest, DependencyMap) {
  const auto& depMap = manager.getDependantAndReadyAlgs(0);
  ASSERT_EQ(depMap.size(), 0);
  auto s = manager.setAlgExecuted(0); // Mark algA as executed
  ASSERT_TRUE(s);
  const auto& depMapAfterA = manager.getDependantAndReadyAlgs(0);
  manager.dumpContents();
  ASSERT_EQ(depMapAfterA.size(), 1);
  ASSERT_EQ(depMapAfterA[0], 1); // algB should be ready
  s = manager.setAlgExecuted(1); // Mark algB as executed
  ASSERT_TRUE(s);
  const auto& depMapAfterB = manager.getDependantAndReadyAlgs(1);
  ASSERT_EQ(depMapAfterB.size(), 1);
  ASSERT_EQ(depMapAfterB[0], 2); // algC should be ready
  s = manager.setAlgExecuted(2); // Mark algC as executed
  ASSERT_TRUE(s);
  const auto& depMapAfterC = manager.getDependantAndReadyAlgs(2);
  ASSERT_EQ(depMapAfterC.size(), 0); // No more algorithms should be ready
}


int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
