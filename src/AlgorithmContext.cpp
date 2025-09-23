#include "NewScheduler.hpp"
#include "AlgorithmContext.hpp"

void AlgorithmContext::newScheduleResumeCallback(void* args) {
  using AR = NewScheduler::NewRunQueue::ActionRequest;
  std::unique_ptr<AlgorithmContext> ctx(static_cast<AlgorithmContext*>(args));

  // std::cout << "In NewAlgoContext::newScheduleResumeCallback(): Scheduling resume for event " << ctx->eventNumber
  //           << ", slot " << ctx->slotNumber << ", alg " << ctx->algorithmNumber << std::endl;
  ctx->scheduler.m_runQueue.queue.push(AR{
    .type = AR::ActionType::Resume,
    .slot = ctx->slotNumber,
    .alg = ctx->algorithmNumber
  });
}
