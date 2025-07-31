#include "NewScheduler.hpp"
#include "AssertCuda.cuh"
#include <exception>

NewScheduler::NewScheduler(int threads, int slots, ExecutionStrategy executionStrategy)
    : m_threadsNumber{threads},
      m_eventSlotsNumber{slots},
      m_nextEvent{},
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
  m_nextEvent = 0;
  m_remainingEvents = 0;
  m_eventSlots.clear();
  // First, populate the algorithm dependency map with the algorithms.
  new(&m_algoDependencyMap) NewAlgoDependencyMap(m_algorithms);

  // Then, create the event slots. The event slots are assigned an event number
  // in all cases, even if the event is above target and the slot will be active.
  // It will possibly be in subsequent runs.
  m_eventSlots.resize(m_eventSlotsNumber);
  for (auto & slot : m_eventSlots) {
    slot.eventNumber = m_nextEvent++;
    new (&slot.eventContentManager) NewEventContentManager(m_algoDependencyMap);

    // Create a CUDA stream for the slot
    ASSERT_CUDA(cudaStreamCreate(&slot.stream));
    // Initialize algorithms in the slot
    slot.algorithms.resize(m_algorithms.size());
  }
  // Initialize the run queue

  // Event slot need a reference to the event content manager
  // and other necessary initializations
}