#include "NewAlgorithmBase.hpp"
#include "EventContext.hpp"

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraight() const {
  auto exec = execute();
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraightDelegated() const {
  auto exec = execute();
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraightMutexed() const {
  auto exec = execute();
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraightThreadLocalStreams() const {
  auto exec = execute();
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraightThreadLocalContext() const {
  auto exec = execute();
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeGraph() const {
  auto exec = execute();
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeGraphFullyDelegated() const {
  auto exec = executeGraph();
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeCachedGraph() const {
  auto exec = executeGraph();
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeCachedGraphDelegated() const {
  auto exec = executeCachedGraph();
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}