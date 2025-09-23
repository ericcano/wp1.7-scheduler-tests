#include "NewAlgorithmBase.hpp"

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraight(AlgorithmContext ctx) const {
  auto exec = execute(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraightDelegated(AlgorithmContext ctx) const {
  auto exec = execute(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraightMutexed(AlgorithmContext ctx) const {
  auto exec = execute(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraightThreadLocalStreams(AlgorithmContext ctx) const {
  auto exec = execute(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraightThreadLocalContext(AlgorithmContext ctx) const {
  auto exec = execute(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeGraph(AlgorithmContext ctx) const {
  auto exec = execute(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }  
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeGraphFullyDelegated(AlgorithmContext ctx) const {
  auto exec = executeGraph(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeCachedGraph(AlgorithmContext ctx) const {
  auto exec = executeGraph(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeCachedGraphDelegated(AlgorithmContext ctx) const {
  auto exec = executeCachedGraph(ctx);
  while (exec.isResumable()) {
    // Process the coroutine execution
    co_yield exec.getYield();
    exec.resume();
  }
  co_return exec.getReturn();
}