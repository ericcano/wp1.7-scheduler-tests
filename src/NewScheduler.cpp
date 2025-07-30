#include "NewScheduler.hpp"
#include <exception>

NewScheduler::NewScheduler(int threads, int slots, ExecutionStrategy executionStrategy)
    : m_threads{threads},
      m_slots{slots},
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
  m_nextEvent = 0;
  m_remainingEvents = 0;
}