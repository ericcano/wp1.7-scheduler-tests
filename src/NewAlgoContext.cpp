#include "NewScheduler.hpp"
#include "NewAlgoContext.hpp"

void NewAlgoContext::newScheduleResumeCallback(void* args) {
  using AR = NewScheduler::NewRunQueue::ActionRequest;
  std::unique_ptr<NewAlgoContext> ctx(static_cast<NewAlgoContext*>(args));

  // std::cout << "In NewAlgoContext::newScheduleResumeCallback(): Scheduling resume for event " << ctx->eventNumber
  //           << ", slot " << ctx->slotNumber << ", alg " << ctx->algorithmNumber << std::endl;
  ctx->scheduler.m_runQueue.queue.push(AR{
    .type = AR::ActionType::Resume,
    .slot = ctx->slotNumber,
    .alg = ctx->algorithmNumber
  });
}
