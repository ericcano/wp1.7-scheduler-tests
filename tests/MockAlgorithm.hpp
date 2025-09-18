#pragma once
#include "NewAlgorithmBase.hpp"
#include "EventContext.hpp"
#include <vector>
#include <string>
#include <cassert>

/**
 * @brief A mock algorithm for testing purposes. Records algorithm execution. Can inject errors.
 */
class MockAlgorithm : public NewAlgorithmBase {
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
    AlgCoInterface execute(NewAlgoContext& ctx) const override {
        getExecutionTracker().tracker.insert({ctx.eventNumber, ctx.algorithmNumber});
        if (ctx.eventNumber == m_injectErrorAtEvent) {
            co_return StatusCode::FAILURE;
        }
        co_return {}; 
    }
    StatusCode finalize() override { return StatusCode::SUCCESS; }

    // A record of the executed algorithms
    using ExecutionTracker = std::set<std::tuple<int, std::size_t>>;

    struct LockedTracker {
        std::scoped_lock<std::mutex> lock;
        ExecutionTracker& tracker;
    };

    static LockedTracker getExecutionTracker() {
        static std::mutex mutex;
        static ExecutionTracker executionTracker;
        return {std::scoped_lock<std::mutex>(mutex), executionTracker};
    }

    static void clear() {
        auto lockedTracker = getExecutionTracker();
        lockedTracker.tracker.clear();
    }
    
    // Inject errors,
    private:
    std::size_t m_injectErrorAtEvent = std::numeric_limits<std::size_t>::max();

    public:
    void setInjectErrorAtEvent(std::size_t event) { m_injectErrorAtEvent = event; }

    // TODO: add event store interface.
};
