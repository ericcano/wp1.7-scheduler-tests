#include "NewScheduler.hpp"
#include "AssertCuda.cuh"
#include <exception>
#include <ranges>

NewScheduler::NewScheduler(int threads, int slots, ExecutionStrategy executionStrategy)
    : m_threadsNumber{threads},
      m_eventSlotsNumber{slots},
      m_nextEventId{},
      m_remainingEvents{},
      m_executionStrategy(executionStrategy) {}


void NewScheduler::addAlgorithm(AlgorithmBase& alg) {
  if (m_runStarted) {
    throw RuntimeError("In NewScheduler::addAlgorithm(): Algorithms cannot be added after run start");
  }
  m_algorithms.push_back(alg);
}


void NewScheduler::initSchedulerState()  {
  // Ge all to initial state
  m_nextEventId = 0;
  m_remainingEvents = 0;
  m_eventSlots.clear();
  // First, populate the algorithm dependency map with the algorithms.
  new(&m_algoDependencyMap) NewAlgoDependencyMap(m_algorithms);

  // Then, create the event slots. The event slots are assigned an event number
  // in all cases, even if the event is above target and the slot will be active.
  // It will possibly be in subsequent runs.
  // NewEventSlot is not movable or copyable so we do a placement new instead of resize.
  new (&m_eventSlots) std::vector<NewEventSlot>(m_eventSlotsNumber);
  for (auto slotId: std::ranges::iota_view(0, m_eventSlotsNumber)) {
     std::ignore = slotId; // Avoid unused variable warning
     m_eventSlots[slotId].initialize(m_algoDependencyMap, m_nextEventId++);
  //m_eventSlots[slotId].eventContentManager.dumpContents(m_algoDependencyMap, std::cout);
  }
}

StatusCode NewScheduler::run(int eventsToProcess, RunStats& stats) {
  // Lock and algorithm registration.
  if (!m_runStarted) {
    initSchedulerState();
    m_runStarted = true;
  }
  m_remainingEvents.store(eventsToProcess);
  m_targetEventId += eventsToProcess;
  auto startTime = std::chrono::high_resolution_clock::now();
  // Populate the run queue with requests for each runnable algorithm in each slot.
  // (At this point, only the algorithms with no dependencies will be runnable).
  for (auto& slot: m_eventSlots) {
    if (slot.eventNumber >= m_targetEventId) continue;
    for (auto& alg: slot.algorithms) {
      std::size_t algId = &alg - &slot.algorithms[0];
      if(m_algoDependencyMap.isAlgIndependent(algId)) {
        
        int sId = &slot - &m_eventSlots[0];
        NewRunQueue::ActionRequest req{NewRunQueue::ActionRequest::ActionType::Start, sId, algId, false};
        m_runQueue.queue.push(req);
      }
    }
  }
  throw 1;
  // Launch the worker threads.
  // TODO
  // for (auto i: std::ranges::iota_view(0, m_threadsNumber - 1)) {
  //    std::ignore = i; // Avoid unused variable warning
  //    m_workerThreads.push_back(std::thread([this]() {
  //       while (true) {
  //          ActionRequest action;
  //          m_runQueue.queue.pop(action);
  //          if (action.exit) break;
  //          if (StatusCode status = executeAction(action); !status) {
  //             // Fatal error, stop everything.
  //             std::cerr << "Fatal error in worker thread: " << status.what() << std::endl;
  //             std::terminate();
  //          }
  //       }
  //    }));
  }
