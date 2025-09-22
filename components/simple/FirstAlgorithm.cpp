#include "FirstAlgorithm.hpp"

#include <iostream>
#include <memory>
#include <future>

#include "CudaKernels.cuh"
#include "EventContext.hpp"
#include "EventStore.hpp"
#include "MemberFunctionName.hpp"
#include "Scheduler.hpp"
#include "CUDAThread.hpp"
#include "CUDAMutex.hpp"
#include "CUDAThreadLocalStream.hpp"
#include "CUDAThreadLocalContext.hpp"
#include "../../tests/NVTXUtils.hpp"
using WP17Scheduler::NVTXUtils::nvtxcolor;

#include <cuda_runtime.h>

// --- FirstAlgorithmGraph Implementation ---

FirstAlgorithmGraph::FirstAlgorithmGraph() {
    std::lock_guard<std::mutex> lock(m_graphMutex);

    CUDA_ASSERT(cudaGraphCreate(&m_graph, 0));

    m_kernel1Params.func = kernel1Address();
    m_kernel1Params.gridDim = dim3(2);
    m_kernel1Params.blockDim = dim3(2);
    m_kernel1Params.sharedMemBytes = 0;
    m_kernel1Params.kernelParams = nullptr;
    m_kernel1Params.extra = nullptr;
    
    CUDA_ASSERT(cudaGraphAddKernelNode(&m_kernel1Node, m_graph, nullptr, 0, &m_kernel1Params));
    m_kernel2Params.func = kernel2Address();
    m_kernel2Params.gridDim = dim3(2);
    m_kernel2Params.blockDim = dim3(2); 
    m_kernel2Params.sharedMemBytes = 0;
    m_kernel2Params.kernelParams = nullptr;
    m_kernel2Params.extra = nullptr;
    CUDA_ASSERT(cudaGraphAddKernelNode(&m_kernel2Node, m_graph, &m_kernel1Node, 1, &m_kernel2Params));

    m_hostFunctionParams.fn = NewAlgoContext::newScheduleResumeCallback;
    m_hostFunctionParams.userData = nullptr;
    CUDA_ASSERT(cudaGraphAddHostNode(&m_HostFunctionNode, m_graph, &m_kernel2Node, 1, &m_hostFunctionParams));
    
    CUDA_ASSERT(cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0));
}

FirstAlgorithmGraph::~FirstAlgorithmGraph() {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    if (m_graphExec) cudaGraphExecDestroy(m_graphExec);
    if (m_graph) cudaGraphDestroy(m_graph);
}

void FirstAlgorithmGraph::launchGraph(cudaStream_t stream, NewAlgoContext* notification) {
    std::lock_guard<std::mutex> lock(m_graphMutex);

    // Only update host function userData for this launch
    m_hostFunctionParams.userData = notification;
    CUDA_ASSERT(cudaGraphExecHostNodeSetParams(m_graphExec, m_HostFunctionNode, &m_hostFunctionParams));

    CUDA_ASSERT(cudaGraphLaunch(m_graphExec, stream));
}

void FirstAlgorithmGraph::launchGraphDelegated(cudaStream_t stream, NewAlgoContext* notification) {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    std::promise<void> promise;
    std::future<void> future = promise.get_future();

    // Only update host function userData for this launch
    m_hostFunctionParams.userData = notification;
    CUDA_ASSERT(cudaGraphExecHostNodeSetParams(m_graphExec, m_HostFunctionNode, &m_hostFunctionParams));

    CUDAThread::post([&]() {
        CUDA_ASSERT(cudaGraphLaunch(m_graphExec, stream));
        promise.set_value();
    });

    future.get();
}

// --- FirstAlgorithm Implementation ---

FirstAlgorithm::FirstAlgorithm(bool errorEnabled, int errorEventId, bool verbose)
    : m_errorEnabled(errorEnabled), m_errorEventId(errorEventId), m_verbose(verbose) {}

StatusCode FirstAlgorithm::initialize() {
    nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(FirstAlgorithm)};
    SC_CHECK(addProduct<int>("Object1"));
    SC_CHECK(addProduct<int>("Object2"));
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) << std::endl;
    }
    return StatusCode::SUCCESS;
}

NewAlgorithmBase::AlgCoInterface FirstAlgorithm::execute(NewAlgoContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 start, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    auto output1 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output1), products()[0]));
    auto output2 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output2), products()[1]));

    // Inject error if enabled
    if (m_errorEnabled && ctx.eventNumber == m_errorEventId) {
        StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
        status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
        status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
        range1.reset();
        co_return status;
    }

    launchTestKernel1(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, NewAlgoContext::newScheduleResumeCallback, new NewAlgoContext{ctx});
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 end, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    range1.reset();
    co_yield StatusCode::SUCCESS;

    auto range2 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part2, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part2, " << ctx.info() << std::endl;
    }
    launchTestKernel2(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, NewAlgoContext::newScheduleResumeCallback, new NewAlgoContext{ctx});
    range2.reset();
    co_yield StatusCode::SUCCESS;

    auto range3 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " conclusion, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    co_return StatusCode::SUCCESS;
}

NewAlgorithmBase::AlgCoInterface FirstAlgorithm::executeStraight(NewAlgoContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 start, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    auto output1 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output1), products()[0]));
    auto output2 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output2), products()[1]));

    // Inject error if enabled
    if (m_errorEnabled && ctx.eventNumber == m_errorEventId) {
        StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
        status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
        status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
        range1.reset();
        co_return status;
    }

    launchTestKernel1(ctx.stream);
    launchTestKernel2(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, NewAlgoContext::newScheduleResumeCallback, new NewAlgoContext{ctx});
    range1.reset();
    co_yield StatusCode::SUCCESS;

    auto range3 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " conclusion, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    co_return StatusCode::SUCCESS;
}

NewAlgorithmBase::AlgCoInterface FirstAlgorithm::executeStraightDelegated(NewAlgoContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 start, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    auto output1 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output1), products()[0]));
    auto output2 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output2), products()[1]));

    // Inject error if enabled
    if (m_errorEnabled && ctx.eventNumber == m_errorEventId) {
        StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
        status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
        status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
        range1.reset();
        co_return status;
    }

    // Launch kernels in a single thread to avoid performance drop
    auto * notif = new NewAlgoContext{ctx};
    CUDAThread::post([ctx, notif]() {
        launchTestKernel1(ctx.stream);
        launchTestKernel2(ctx.stream);
        cudaLaunchHostFunc(ctx.stream, NewAlgoContext::newScheduleResumeCallback, notif);
    });
    range1.reset();
    co_yield StatusCode::SUCCESS;

    auto range3 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " conclusion, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    co_return StatusCode::SUCCESS;
}

NewAlgorithmBase::AlgCoInterface FirstAlgorithm::executeStraightMutexed(NewAlgoContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 start, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    auto output1 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output1), products()[0]));
    auto output2 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output2), products()[1]));

    // Inject error if enabled
    if (m_errorEnabled && ctx.eventNumber == m_errorEventId) {
        StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
        status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
        status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
        range1.reset();
        co_return status;
    }

    auto cudaLock = CUDAMutex::lock();
    launchTestKernel1(ctx.stream);
    launchTestKernel2(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, NewAlgoContext::newScheduleResumeCallback, new NewAlgoContext{ctx});
    cudaLock.unlock();
    range1.reset();
    co_yield StatusCode::SUCCESS;

    auto range3 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " conclusion, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    co_return StatusCode::SUCCESS;
}

NewAlgorithmBase::AlgCoInterface FirstAlgorithm::executeStraightThreadLocalStreams(NewAlgoContext ctx) const {
    auto stream = CUDAThreadLocalStream::get();    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 start, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " + ctx.info() + " stream=" + std::to_string((uint64_t)stream), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    auto output1 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output1), products()[0]));
    auto output2 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output2), products()[1]));

    // Inject error if enabled
    if (m_errorEnabled && ctx.eventNumber == m_errorEventId) {
        StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
        status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
        status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
        range1.reset();
        co_return status;
    }

    launchTestKernel1(stream);
    launchTestKernel2(stream);
    cudaLaunchHostFunc(stream, NewAlgoContext::newScheduleResumeCallback, new NewAlgoContext{ctx});
    range1.reset();
    co_yield StatusCode::SUCCESS;

    auto range2 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " conclusion, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    co_return StatusCode::SUCCESS;
}

NewAlgorithmBase::AlgCoInterface FirstAlgorithm::executeStraightThreadLocalContext(NewAlgoContext ctx) const {
    CUDAThreadLocalContext::check(); // Ensure the primary context is retained for this thread
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 start, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    auto output1 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output1), products()[0]));
    auto output2 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output2), products()[1]));

    // Inject error if enabled
    if (m_errorEnabled && ctx.eventNumber == m_errorEventId) {
        StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
        status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
        status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
        range1.reset();
        co_return status;
    }

    launchTestKernel1(ctx.stream);
    launchTestKernel2(ctx.stream);
    cudaLaunchHostFunc(ctx.stream, NewAlgoContext::newScheduleResumeCallback, new NewAlgoContext{ctx});
    range1.reset();
    co_yield StatusCode::SUCCESS;

    auto range2 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " conclusion, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    co_return StatusCode::SUCCESS;
}

NewAlgorithmBase::AlgCoInterface FirstAlgorithm::executeGraph(NewAlgoContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 start, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    auto output1 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output1), products()[0]));
    auto output2 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output2), products()[1]));

    // Inject error if enabled
    if (m_errorEnabled && ctx.eventNumber == m_errorEventId) {
        StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
        status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
        status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
        range1.reset();
        co_return status;
    }

    m_graphImpl.launchGraph(ctx.stream, new NewAlgoContext{ctx});
    range1.reset();
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 end, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    co_yield StatusCode::SUCCESS;

    auto range2 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " conclusion, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    co_return StatusCode::SUCCESS;
}

NewAlgorithmBase::AlgCoInterface FirstAlgorithm::executeGraphFullyDelegated(NewAlgoContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 start, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    auto output1 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output1), products()[0]));
    auto output2 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output2), products()[1]));

    // Inject error if enabled
    if (m_errorEnabled && ctx.eventNumber == m_errorEventId) {
        StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
        status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
        status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
        range1.reset();
        co_return status;
    }

    auto* notif = new NewAlgoContext{ctx};
    CUDAThread::post([ctx, this, notif]() {
        m_graphImpl.launchGraph(ctx.stream, notif);
    });
    range1.reset();
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 end, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    co_yield StatusCode::SUCCESS;

    auto range2 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " conclusion, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    co_return StatusCode::SUCCESS;
}

NewAlgorithmBase::AlgCoInterface FirstAlgorithm::executeCachedGraph(NewAlgoContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 start, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    auto output1 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output1), products()[0]));
    auto output2 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output2), products()[1]));

    // Inject error if enabled
    if (m_errorEnabled && ctx.eventNumber == m_errorEventId) {
        StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
        status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
        status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
        range1.reset();
        co_return status;
    }

    m_graphContainer.launchGraph(ctx.stream, new NewAlgoContext{ctx});
    range1.reset();
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 end, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    co_yield StatusCode::SUCCESS;

    auto range2 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " conclusion, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    co_return StatusCode::SUCCESS;
}

NewAlgorithmBase::AlgCoInterface FirstAlgorithm::executeCachedGraphDelegated(NewAlgoContext ctx) const {
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 start, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    auto range1 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    auto output1 = std::make_unique<int>(-1);
    assert(products().size() >= 1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output1), products()[0]));
    auto output2 = std::make_unique<int>(-1);
    SC_CHECK_YIELD(ctx.eventStore.record(std::move(output2), products()[1]));

    // Inject error if enabled
    if (m_errorEnabled && ctx.eventNumber == m_errorEventId) {
        StatusCode status{StatusCode::FAILURE, "FirstAlgorithm execute failed"};
        status.appendMsg("context event number: " + std::to_string(ctx.eventNumber));
        status.appendMsg("context slot number: " + std::to_string(ctx.slotNumber));
        range1.reset();
        co_return status;
    }

    m_graphContainer.launchGraphDelegated(ctx.stream, new NewAlgoContext{ctx});
    range1.reset();
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) + " part1 end, " << ctx.info() << " tid=" << gettid() << std::endl;
    }
    co_yield StatusCode::SUCCESS;

    auto range2 = std::make_unique<nvtx3::unique_range>(MEMBER_FUNCTION_NAME(FirstAlgorithm) + " conclusion, " + ctx.info(), nvtxcolor(ctx.eventNumber), nvtx3::payload{gettid()});
    co_return StatusCode::SUCCESS;
}

StatusCode FirstAlgorithm::finalize() {
    nvtx3::scoped_range range{MEMBER_FUNCTION_NAME(FirstAlgorithm)};
    if (m_verbose) {
        std::cout << MEMBER_FUNCTION_NAME(FirstAlgorithm) << std::endl;
    }
    return StatusCode::SUCCESS;
}
