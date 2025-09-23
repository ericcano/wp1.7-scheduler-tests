#include <cstdlib>
#include <iostream>

#include "NewScheduler.hpp"
#include "StatusCode.hpp"
#include "TracccAlgorithm.hpp"
#include "TracccAlgs.hpp"
#include "TracccCudaAlgorithm.hpp"

int main() {
   // Create the scheduler.
   NewScheduler scheduler(4, 4);

   // Create the algorithms.
   TracccCellsAlgorithm cellsAlg(10);
   TracccComputeAlgorithm computeAlg(10);

   // Add the algorithms to the scheduler.
   scheduler.addAlgorithm(cellsAlg);
   scheduler.addAlgorithm(computeAlg);

   // Run the scheduler.
   NewScheduler::RunStats stats;
   auto w = scheduler.run(1000, stats).what();
   std::cout << w << std::endl;
   return EXIT_SUCCESS;
}
