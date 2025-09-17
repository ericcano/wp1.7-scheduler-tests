#pragma once


#include <cuda_runtime_api.h>

#include <cstdlib>
#include <string>

// Forward declarations.
class EventStore;
class NewScheduler;

/**
 * @brief Indices to event/slot context, plus pointed to global scheduler, they are the sole interface to the algorithms, via `execute()`.
 * Also contains the CUDA stream.
 */
struct NewAlgoContext {
   int eventNumber = 0;
   int slotNumber = 0;
   std::size_t algorithmNumber = 0; // Index of the current algorithm in the scheduler's list.
   NewScheduler& scheduler;
   EventStore& eventStore;
   cudaStream_t stream;

   std::string info() const {
      return "ctx.eventNumber = " + std::to_string(this->eventNumber)
           + ", ctx.slotNumber = " + std::to_string(this->slotNumber)
           + ", ctx.algorithmNumber = " + std::to_string(this->algorithmNumber);
   }
   // Function to be passed to CUDA callback.
   // it will queue the algorithm resume in the scheduler's run queue.
   static void newScheduleResumeCallback(void* args);
};

