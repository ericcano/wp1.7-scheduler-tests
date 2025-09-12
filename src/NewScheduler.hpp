#pragma once
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>
#include <stdexcept>
#include <ranges>
#include <tbb/concurrent_queue.h>
#include <cuda_runtime_api.h>
#include "AlgorithmBase.hpp"
#include "NewAlgoDependencyMap.hpp"

namespace WP17NewScheduler {
   /**
    * @brief Base exception class
    */
   class Exception: public std::exception {
   public:
      Exception(const std::string &w): m_what(w) {}
      const char * what() const noexcept override { return m_what.c_str(); } 
   private:
      const std::string m_what;
   };
}

#define DEFINE_EXCEPTION(name) class name: public WP17NewScheduler::Exception { using WP17NewScheduler::Exception::Exception; }

/**
 * @brief A progressive build of the scheduler replacement.
 */
struct NewScheduler {
public:
    enum class ExecutionStrategy {
        SingleLaunch,
        StraightLaunches,
        StraightDelegated,
        StraightMutexed,
        StraightThreadLocalStreams,
        StraightThreadLocalContext,
        Graph,
        GraphFullyDelegated,
        CachedGraphs,
        CachedGraphsDelegated
    };

   /**
    * @brief NewScheduler constructor
    * @param threads number of threads
    * @param slots Number of slots (i.e. concurrent events being processed).
    * @param executionStrategy Execution strategy for the algorithms.
    */
   NewScheduler(int threads = 4, int slots = 4, ExecutionStrategy executionStrategy = ExecutionStrategy::SingleLaunch);

   // Forbidden constructors
   NewScheduler(const NewScheduler&) = delete;
   NewScheduler& operator=(const NewScheduler&) = delete;

   /**
    * @brief Destructor just cleans up the CUDA streams.
    */
   ~NewScheduler() {}

   /** 
    * @brief Adds an algorithm to the algorithm list. This function should be called before running. 
    * @todo: Should this be a constant reference? 
    */
   void addAlgorithm(AlgorithmBase& alg);

private:
   /**
    * @brief Assigns in initial event ids to each `slot` and creates the CUDA streams.
    */
   void initSchedulerState();

public:

   /**
    * @brief Structure holding algorithm state and interface to run it.
    */
   struct NewAlgoSlot {
      /**
       * @brief Mutex to ensure that only one thread processes the algorithm. The processing can queue
       * callbacks to GPU or other async resource, which will add the resuming of this algorithm to the 
       * run queue.
       * @note The locking model is to first hold the mutex on the algo and then briefly get the slot scheduling
       * mutex to update the slot state.
       */
      std::mutex mutex;

      /**
       * @brief Algorithm coroutine.
       */
      AlgorithmBase::AlgCoInterface coroutine; // Coroutine interface for the algorithm
   };

   /**
    * @brief Structure holding all the algos and data for one the processing of one event.
    */
   struct NewEventSlot {

      NewEventSlot() = default;
      NewEventSlot(const NewEventSlot&) = delete;
      NewEventSlot& operator=(const NewEventSlot&) = delete;
      NewEventSlot(NewEventSlot&&) = delete;
      NewEventSlot& operator=(NewEventSlot&&) = delete;
      /**
       * @brief Initializer of event content manager.
       */
      void initialize(const NewAlgoDependencyMap& depMap) {
         eventContentManager.resize(depMap);
         for (auto aidx: std::ranges::iota_view(std::size_t(0), depMap.algorithmsCount())) {
            std::ignore = algorithms[aidx];
         }
         //algorithms.resize(depMap.algorithmsCount());
      }

      /**
       * @brief Event number for the slot.
       */
      int eventNumber = -1;

      /**
       * @brief Mutex to ensure that only one thread schedules the algorithms in the slot.
       */
      std::mutex schedulingMutex;

      /**
       * @brief Algorithms in the slot, each with its own coroutine interface and mutex.
       */
      std::vector<NewAlgoSlot> algorithms; // Algorithms in the slot, each with its own coroutine interface and mutex

      /**
       * @brief Event content manager for the slot, managing data objects and dependencies.
       */
      NewEventContentManager eventContentManager;

      /**
       * @brief CUDA stream for the slot
       */
      cudaStream_t stream = nullptr;
      // Add product manager ?
      // TODO: Add helper functions
   };

   /**
    * @brief Run queue for the scheduler
    */
   struct NewRunQueue {
      struct ActionRequest {
        enum class ActionType {
          Start,  ///< Request to start the slot
          Resume  ///< Request to resume the worker thread
        };
        int slot = 0; ///< Slot index
        std::size_t alg = 0; ///< Algorithm index
        bool exit = false; ///< Exit flag for the worked thread
      };
      tbb::concurrent_queue<ActionRequest> queue; // Queue of action requests
   };
private:

   /// @brief Number of threads to use.
   int m_threadsNumber;

   /// @brief Number of slots to use (i.e. concurrent events being processed).
   int m_eventSlotsNumber;

   /// @brief Id of the next event to process.
   int m_nextEvent = 0;

   /// @brief Flag controling the switchover from configuring (registering algorithms) to running.
   bool m_runStarted = false;

   /// @brief Target event number for the current run.
   int m_targetEventId = 0;

   /// @brief Number of events remaining to be processed.
   std::atomic_int m_remainingEvents;

   /// @brief List of algorithms retistered in the scheduler.
   std::vector<std::reference_wrapper<AlgorithmBase>> m_algorithms;

   /// @brief The event content manager
   NewAlgoDependencyMap m_algoDependencyMap;

   /// @brief Vector tracking each slot's state
   std::vector<NewEventSlot> m_eventSlots;

  //  /// @brief CUDA streams for each slot, in a one-to-one relationship.
  //  /// @todo It should simply be a member of SlotState.
  //  std::vector<cudaStream_t> m_streams;

  //  /// @brief TBB task arena representing the thread pool we will run on.
  //  tbb::task_arena m_arena;

  //  /// @brief TBB task group to control the tasks.
  //  tbb::task_group m_group;

  //  /// @brief TBB concurrent bounded queue for keeping track of actions to be executed.
  //  tbb::concurrent_bounded_queue<action_type> m_actionQueue;

   /// @brief Execution strategy for the algorithms.
   ExecutionStrategy m_executionStrategy;
   /**
    * @brief Run queue for the scheduler
    */
   NewRunQueue m_runQueue;
   /**
    * @brief Worker threads for processing the slots
    */
   std::vector<std::thread> m_workerThreads;

public:

   /// @brief Exception class for scheduler errors.
   DEFINE_EXCEPTION(RuntimeError);
};
