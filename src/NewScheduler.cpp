#include "NewScheduler.hpp"
#include "AssertCuda.cuh"
#include <exception>
#include <ranges>

#pragma GCC optimize("O0")

NewScheduler::NewScheduler(int threads, int slots, ExecutionStrategy executionStrategy)
    : m_threadsNumber{threads},
      m_eventSlotsNumber{slots},
      m_nextEventId{},
      m_executionStrategy(executionStrategy) {}


void NewScheduler::addAlgorithm(NewAlgorithmBase& alg) {
  if (m_runStarted) {
    throw RuntimeError("In NewScheduler::addAlgorithm(): Algorithms cannot be added after run start");
  }
  m_algorithms.push_back(alg);
}


void NewScheduler::initSchedulerState()  {
  // Ge all to initial state
  m_nextEventId = 0;
  m_remainingEventsToComplete = 0;
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
  m_remainingEventsToComplete.store(eventsToProcess);
  m_targetEventId += eventsToProcess;
  auto startTime = std::chrono::high_resolution_clock::now();
  // Populate the run queue with requests for each runnable algorithm in each slot.
  // (At this point, only the algorithms with no dependencies will be runnable).
  populateRunQueue();
  startWorkerThreads();
  processRunQueue();
  joinWorkerThreads();
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  double rate = static_cast<double>(eventsToProcess) / (duration / 1000.0); // Events per second

  // Store run statistics
  stats.events = eventsToProcess;
  stats.rate = rate;
  stats.duration = duration;

  //std::cout << "Processed " << m_events << " events in " << duration << " ms (" << rate << " events/sec)" << std::endl;

  return StatusCode::SUCCESS;
}


void NewScheduler::populateRunQueue() {
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
}

NewScheduler::NewRunQueue::ActionRequest NewScheduler::scheduleNextEventInSlot(NewEventSlot& slot) {
  // Clear all coroutines in the slot.
  for (auto& alg: slot.algorithms) {
    std::scoped_lock lock(alg.mutex);
    if (alg.coroutine.empty() || alg.coroutine.isResumable()) {
      std::cerr << "In NewScheduler::scheduleNextEventInSlot(): Attempting to schedule a new event in a slot where an algorithm has not completed." << std::endl;
      std::cerr << "Event Slot State:" << std::endl;
      std::cerr << "Event Number: " << slot.eventNumber << std::endl;
      std::cerr << "Algorithms:" << std::endl;
      for (std::size_t i = 0; i < slot.algorithms.size(); ++i) {
        auto& a = slot.algorithms[i];
        std::cerr << "  Algorithm " << i << ": ";
        if (a.coroutine.empty()) {
          std::cerr << "Not started";
        } else if (a.coroutine.isResumable()) {
          std::cerr << "Resumable";
        } else {
          std::cerr << "Completed";
        }
        std::cerr << std::endl;
      }
      throw RuntimeError("In NewScheduler::scheduleNextEventInSlot(): Algorithm has not completed for previous event");
    }
    alg.coroutine.setEmpty();
  }
  // Update slot event number and clear the coroutines.
  slot.eventNumber = m_nextEventId++;
  slot.eventStore.clear();
  slot.eventContentManager.reset();
  // If we completed the last event, we need to signal the worker threads to exit.
  if (m_remainingEventsToComplete.fetch_sub(1) == 1) {
    for (int i = 0; i < m_threadsNumber; ++i) {
      NewRunQueue::ActionRequest exitReq{NewRunQueue::ActionRequest::ActionType::Exit, -1,
        std::numeric_limits<std::size_t>::max(), true};
      m_runQueue.queue.push(exitReq);
    }
  }
  // Did we reach the target event number. In this case, we do not need to schedule it.
  if (slot.eventNumber >= m_targetEventId) {
    return NewRunQueue::ActionRequest{NewRunQueue::ActionRequest::ActionType::Exit, 0, 0, true};
  }
  // We will schedule this event's first algos.
  NewRunQueue::ActionRequest firstReq;
  bool first = true;
  for (auto& alg: slot.algorithms) {
    std::size_t algId = &alg - &slot.algorithms[0];
    if(m_algoDependencyMap.isAlgIndependent(algId)) {
      
      int sId = &slot - &m_eventSlots[0];
      NewRunQueue::ActionRequest req{NewRunQueue::ActionRequest::ActionType::Start, sId, algId, false};
      if (first) {
        first = false;
        firstReq = req;
      } else {
        m_runQueue.queue.push(req);
      }
    }
  }
  return firstReq;
}

void NewScheduler::startWorkerThreads() {
  // Start worker threads if not already started.
  if (m_workerThreads.empty()) {
    for (int i = 0; i < m_threadsNumber - 1; ++i) {
      m_workerThreads.emplace_back(&NewScheduler::processRunQueue, this);
    }
  }
}

void NewScheduler::joinWorkerThreads() {
  for (auto& thread : m_workerThreads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  m_workerThreads.clear();
}

void NewScheduler::processRunQueue() {
  while (true) {
    NewRunQueue::ActionRequest req;
    if (m_runQueue.queue.try_pop(req)) {
      if (req.exit) {
        break; // Exit signal received
      }
      processActionRequest(req);
    } else {
      // If no work is available, yield to avoid busy-waiting
      std::this_thread::yield();
    }
  }
}

void NewScheduler::processActionRequest(NewRunQueue::ActionRequest& req) {
  begin:
  // std::cout << "Processing request: slot=" << req.slot << ", alg=" << req.alg << ", type=" 
  //           << (req.type == NewRunQueue::ActionRequest::ActionType::Start ? "Start" : 
  //               req.type == NewRunQueue::ActionRequest::ActionType::Resume ? "Resume" : "Exit");
  // Validate the request.
  if (req.slot < 0 || req.slot >= m_eventSlotsNumber) {
    throw RuntimeError("In NewScheduler::processActionRequest(): Invalid slot index");
  }
  auto& slot = m_eventSlots[req.slot];
  //std::cout << ", event=" << slot.eventNumber << std::endl;
  if (req.alg < 0 || req.alg >= slot.algorithms.size()) {
    throw RuntimeError("In NewScheduler::processActionRequest(): Invalid algorithm index");
  }
  auto& algSlot = slot.algorithms[req.alg];

  // Lock the algorithm mutex to ensure only one thread runs the algorithm's coroutine at a time.
  std::lock_guard<std::mutex> algLock(algSlot.mutex);

  // If the request is to start, ensure the algorithm has not already run for this event.
  if (req.type == NewRunQueue::ActionRequest::ActionType::Start) {
    // If the coroutine is not empty, dump the whole slot state for debugging.
    if (!algSlot.coroutine.empty()) {
      std::cerr << "In NewScheduler::processActionRequest(): Attempting to start an algorithm that has already started for this event." << std::endl;
      std::cerr << "Request: slot=" << req.slot << ", alg=" << req.alg << std::endl;
      std::cerr << "Event Slot State:" << std::endl;
      std::cerr << "Event Number: " << slot.eventNumber << std::endl;
      std::cerr << "Algorithms:" << std::endl;
      for (std::size_t i = 0; i < slot.algorithms.size(); ++i) {
        auto& a = slot.algorithms[i];
        std::cerr << "  Algorithm " << i << ": ";
        if (a.coroutine.empty()) {
          std::cerr << "Not started";
        } else if (a.coroutine.isResumable()) {
          std::cerr << "Resumable";
        } else {
          std::cerr << "Completed";
        }
        std::cerr << std::endl;
      }
      throw RuntimeError("In NewScheduler::processActionRequest(): Algorithm already started for this event");
    }
    assert(algSlot.coroutine.empty());
    auto & alg = m_algorithms[req.alg].get();
    NewAlgoContext ctx{
      .eventNumber = slot.eventNumber,
      .slotNumber = req.slot,
      .algorithmNumber = req.alg,
      .scheduler = *this,
      .eventStore = slot.eventStore,
      .stream = slot.stream
    };
    switch (m_executionStrategy)
    {
      case ExecutionStrategy::SingleLaunch:
          algSlot.coroutine = alg.execute(ctx);
          break;
      case ExecutionStrategy::StraightLaunches:
          algSlot.coroutine = alg.executeStraight(ctx);
          break;
      case ExecutionStrategy::StraightDelegated:
          algSlot.coroutine = alg.executeStraightDelegated(ctx);
          break;
      case ExecutionStrategy::StraightMutexed:
          algSlot.coroutine = alg.executeStraightMutexed(ctx);
          break;
      case ExecutionStrategy::StraightThreadLocalStreams:
          algSlot.coroutine = alg.executeStraightThreadLocalStreams(ctx);
          break;
      case ExecutionStrategy::StraightThreadLocalContext:
          algSlot.coroutine = alg.executeStraightThreadLocalContext(ctx);
          break;
      case ExecutionStrategy::Graph:
          algSlot.coroutine = alg.executeGraph(ctx);
          break;
      case ExecutionStrategy::GraphFullyDelegated:
          algSlot.coroutine = alg.executeGraphFullyDelegated(ctx);
          break;
      case ExecutionStrategy::CachedGraphs:
          algSlot.coroutine = alg.executeCachedGraph(ctx);
          break;
      case ExecutionStrategy::CachedGraphsDelegated:
          algSlot.coroutine = alg.executeCachedGraphDelegated(ctx);
          break;
      default:
          std::cerr << "In Scheduler::pushAction(): Unknown execution strategy:" << to_string(m_executionStrategy) << std::endl;
          abort();
    }
  } else if(req.type == NewRunQueue::ActionRequest::ActionType::Resume) {
    algSlot.coroutine.resume();
  } else {
    throw RuntimeError("In NewScheduler::processActionRequest(): Unexpected action request type");
  }

  // Let's see the outcome.
  StatusCode algStatus;
  bool done = false;
  if (algSlot.coroutine.isResumable()) {
      algStatus = algSlot.coroutine.getYield();
  } else {
      algStatus = algSlot.coroutine.getReturn();
      done = true;
  }

  // If anything went wrong, crash and burn.
  // TODO: be more subtle.
  if (!algStatus) {
    abort();
  }

  // We are done with the algorithm for now.
  algLock.~lock_guard();

  // If an algorithm completed, there might be more to execute.
  if (done) {
    // Algorithm finished processing for this event.
    // Lock the slot mutex to safely check overall slot state.
    std::lock_guard<std::mutex> slotLock(slot.schedulingMutex);
    // TODO: error handling
    std::ignore = slot.eventContentManager.setAlgExecuted(req.alg, m_algoDependencyMap);
    // Get the list of dependent algorithms that might now be ready to run.
    auto dependents = slot.eventContentManager.getDependentAndReadyAlgs(req.alg, m_algoDependencyMap);
    // If there are no dependants, we might have completed the event.
    if (dependents.empty()) {
      // Check if all algorithms are done.
      bool allDone = true;
      for (auto& alg : slot.algorithms) {
        std::scoped_lock lock(alg.mutex);
        if (alg.coroutine.empty() ||  alg.coroutine.isResumable()) {
          allDone = false;
          break;
        }
      }
      if (allDone) {
        // Schedule the next event if available.
        auto nextReq = scheduleNextEventInSlot(slot);
        if (nextReq.exit) return;
        slotLock.~lock_guard();
        req = nextReq;
        // std::cout << "GOTO on next event ";
        goto begin;
      }
    } else {
      slotLock.~lock_guard();
      req = {NewRunQueue::ActionRequest::ActionType::Start, req.slot, dependents[0], false};
      for (std::size_t i = 1; i < dependents.size(); ++i) {
        NewRunQueue::ActionRequest depReq{NewRunQueue::ActionRequest::ActionType::Start, req.slot, dependents[i], false};
        m_runQueue.queue.push(depReq);
      }
      algLock.~lock_guard();
      // std::cout << "GOTO on dependent ";
      goto begin;
    }
  }
}
