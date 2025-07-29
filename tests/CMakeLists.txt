find_package(GTest REQUIRED)

add_executable(TestEventStore TestEventStore.cpp)
target_link_libraries(TestEventStore PRIVATE CoScheduler ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
add_test(NAME TestEventStore COMMAND TestEventStore)

add_executable(TestEventContentManager TestEventContentManager.cpp)
target_link_libraries(TestEventContentManager PRIVATE CoScheduler ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
add_test(NAME TestEventContentManager COMMAND TestEventContentManager)

## TBB tests
add_executable(TestTBBTaskSuspension TestTBBTaskSuspension.cpp)
target_link_libraries(TestTBBTaskSuspension PUBLIC TBB::tbb CUDA::cudart) 


## CUDA test
add_executable(CUDAGraphTest CUDAGraphs.cu)
target_link_libraries(CUDAGraphTest PRIVATE CUDA::cudart TBB::tbb) 
add_test(NAME CUDAGraphTest COMMAND CUDAGraphTest)