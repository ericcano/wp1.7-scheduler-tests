#pragma once
#include "AlgorithmBase.hpp"
#include "EventContext.hpp"
#include <vector>
#include <string>
#include <cassert>

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
