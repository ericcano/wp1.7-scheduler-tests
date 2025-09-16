#pragma once
#include "NewAlgorithmBase.hpp"
#include "EventContext.hpp"
#include <vector>
#include <string>
#include <cassert>

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
    AlgCoInterface execute(NewAlgoContext& ctx) const override { return {}; }
    StatusCode finalize() override { return StatusCode::SUCCESS; }
};

class MockTrackingAlgorithm : public NewAlgorithmBase {
public:
    MockTrackingAlgorithm(const std::vector<std::string>& dependencies,
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
    AlgCoInterface execute(NewAlgoContext& ctx) const override { return {}; }
    StatusCode finalize() override { return StatusCode::SUCCESS; }
    // TODO: add event store interface.
};
