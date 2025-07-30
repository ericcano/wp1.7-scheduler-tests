find_package(GTest REQUIRED)

## Scheduler tests
add_executable(TestEventStore TestEventStore.cpp)
target_link_libraries(TestEventStore PRIVATE CoScheduler ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
add_test(NAME TestEventStore COMMAND TestEventStore)

add_executable(TestEventContentManager TestEventContentManager.cpp)
target_link_libraries(TestEventContentManager PRIVATE CoScheduler ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
add_test(NAME TestEventContentManager COMMAND TestEventContentManager)

add_executable(TestNewScheduler TestNewScheduler.cpp)
target_compile_options(TestNewScheduler PRIVATE -fno-access-control) # Allow private class members to be accessed in tests
target_link_libraries(TestNewScheduler PRIVATE CoScheduler ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
add_test(NAME TestNewScheduler COMMAND TestNewScheduler)

## TBB tests
add_executable(TestTBBTaskSuspension TestTBBTaskSuspension.cpp)
target_link_libraries(TestTBBTaskSuspension PUBLIC TBB::tbb CUDA::cudart) 

## CUDA test
add_executable(CUDAGraphTest CUDAGraphs.cu)
target_link_libraries(CUDAGraphTest PRIVATE CUDA::cudart TBB::tbb) 

