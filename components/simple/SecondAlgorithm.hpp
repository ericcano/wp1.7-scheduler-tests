#pragma once

#include "NewAlgorithmBase.hpp"
#include "CachingGraphContainer.hpp"

#include <cuda_runtime_api.h>

// Forward declarations
struct Notification;

// Helper class for CUDA Graph logic
class SecondAlgorithmGraph {
public:
    SecondAlgorithmGraph();
    ~SecondAlgorithmGraph();

    void launchGraph(cudaStream_t stream, NewAlgoContext* notification);
    void launchGraphDelegated(cudaStream_t stream, NewAlgoContext* notification);

private:
    cudaGraph_t m_graph{};
    cudaGraphExec_t m_graphExec{};
    cudaGraphNode_t m_kernel3Node{};
    cudaGraphNode_t m_kernel4Node{};
    cudaGraphNode_t m_HostFunctionNode{};
    cudaKernelNodeParams m_kernel3Params{}, m_kernel4Params{};
    cudaHostNodeParams m_hostFunctionParams{};
    std::mutex m_graphMutex;
};

class SecondAlgorithm : public NewAlgorithmBase {
public:
    // Constructor with verbose parameter
    explicit SecondAlgorithm(bool verbose = false);

    StatusCode initialize() override;
    AlgCoInterface execute(NewAlgoContext ctx) const override;
    AlgCoInterface executeStraight(NewAlgoContext ctx) const override;
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
    mutable SecondAlgorithmGraph m_graphImpl; // Graph helper instance
    mutable CachingGraphContainer<SecondAlgorithmGraph> m_graphContainer; // Container for caching graph instances
};
