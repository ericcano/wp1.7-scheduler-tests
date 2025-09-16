#include "NewAlgorithmBase.hpp"
#include "EventContext.hpp"

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraight(NewAlgoContext& ctx) const {
  auto exec = execute(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraightDelegated(NewAlgoContext& ctx) const {
  auto exec = execute(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraightMutexed(NewAlgoContext& ctx) const {
  auto exec = execute(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraightThreadLocalStreams(NewAlgoContext& ctx) const {
  auto exec = execute(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeStraightThreadLocalContext(NewAlgoContext& ctx) const {
  auto exec = execute(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeGraph(NewAlgoContext& ctx) const {
  auto exec = execute(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeGraphFullyDelegated(NewAlgoContext& ctx) const {
  auto exec = executeGraph(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeCachedGraph(NewAlgoContext& ctx) const {
  auto exec = executeGraph(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}

NewAlgorithmBase::AlgCoInterface NewAlgorithmBase::executeCachedGraphDelegated(NewAlgoContext& ctx) const {
  auto exec = executeCachedGraph(ctx);
  while (exec.resume()) {
    // Process the coroutine execution
    co_yield exec.getYield();
  }
  co_return exec.getReturn();
}