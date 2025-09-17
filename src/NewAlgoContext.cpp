#include "NewScheduler.hpp"
#include "NewAlgoContext.hpp"

void NewAlgoContext::newScheduleResumeCallback(void* args) {
  using AR = NewScheduler::NewRunQueue::ActionRequest;
  std::unique_ptr<NewAlgoContext> ctx(static_cast<NewAlgoContext*>(args));

  ctx->scheduler.m_runQueue.queue.push(AR{
    .type = AR::ActionType::Resume,
    .slot = ctx->slotNumber,
    .alg = ctx->algorithmNumber
  });
}
