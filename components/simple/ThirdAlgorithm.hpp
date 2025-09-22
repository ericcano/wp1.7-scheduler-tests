#pragma once

#include "NewAlgorithmBase.hpp"
#include "CachingGraphContainer.hpp"

#include <cuda_runtime_api.h>

// Forward declarations
struct Notification;

class ThirdAlgorithmGraph {
public:
    ThirdAlgorithmGraph();
    ~ThirdAlgorithmGraph();
    void launchGraph(cudaStream_t stream, NewAlgoContext* context);
    void launchGraphDelegated(cudaStream_t stream, NewAlgoContext* context);
private:
    cudaGraph_t m_graph{};
    cudaGraphExec_t m_graphExec{};
    cudaGraphNode_t m_kernel5Node{};
    cudaGraphNode_t m_HostFunctionNode{};
    cudaKernelNodeParams m_kernel5Params{};
    cudaHostNodeParams m_hostFunctionParams{};
    std::mutex m_graphMutex;
};

class ThirdAlgorithm : public NewAlgorithmBase {
public:
    // Constructor with verbose parameter
    explicit ThirdAlgorithm(bool verbose = false);

    StatusCode initialize() override;
    AlgCoInterface execute(NewAlgoContext ctx) const override;
    // Exceute straight is identical to execute (it will fall back to execute())
    AlgCoInterface executeStraightDelegated(NewAlgoContext ctx) const override;
    AlgCoInterface executeStraightMutexed(NewAlgoContext ctx) const override;
    AlgCoInterface executeStraightThreadLocalStreams(NewAlgoContext ctx) const override;
    AlgCoInterface executeStraightThreadLocalContext(NewAlgoContext ctx) const override;
    AlgCoInterface executeGraph(NewAlgoContext ctx) const override;
    AlgCoInterface executeGraphFullyDelegated(NewAlgoContext ctx) const override;
    AlgCoInterface executeCachedGraph(NewAlgoContext ctx) const override;
    AlgCoInterface executeCachedGraphDelegated(NewAlgoContext ctx) const override;
    StatusCode finalize() override;

private:
    bool m_verbose; // Whether verbose output is enabled
    mutable ThirdAlgorithmGraph m_graphImpl; // Graph helper instance
    mutable CachingGraphContainer<ThirdAlgorithmGraph> m_graphContainer; // Container for caching graph instances
};
