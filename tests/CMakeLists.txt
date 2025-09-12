find_package(GTest REQUIRED)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g -O0 -fno-omit-frame-pointer")
set(CMAKE_CUDA_FLAGS_DEBUG "-Wall -Wextra -g -O0 -fno-omit-frame-pointer")

## Scheduler tests
add_executable(TestEventStore TestEventStore.cpp)
target_link_libraries(TestEventStore PRIVATE CoScheduler ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
add_test(NAME TestEventStore COMMAND TestEventStore)

add_executable(TestEventContentManager TestEventContentManager.cpp)
target_link_libraries(TestEventContentManager PRIVATE CoScheduler ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
add_test(NAME TestEventContentManager COMMAND TestEventContentManager)

add_executable(TestNewEventContentManager TestNewEventContentManager.cpp)
target_link_libraries(TestNewEventContentManager PRIVATE CoScheduler ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
target_compile_options(TestNewEventContentManager PRIVATE -fno-access-control) # Allow private class members to be accessed in tests
add_test(NAME TestNewEventContentManager COMMAND TestNewEventContentManager)

add_executable(TestNewScheduler TestNewScheduler.cpp)
target_compile_options(TestNewScheduler PRIVATE -fno-access-control) # Allow private class members to be accessed in tests
target_link_libraries(TestNewScheduler PRIVATE CoScheduler ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
add_test(NAME TestNewScheduler COMMAND TestNewScheduler)

add_executable(TestNewEventSlot TestNewEventSlot.cpp)
target_compile_options(TestNewEventSlot PRIVATE -fno-access-control) # Allow private class members to be accessed in tests
target_link_libraries(TestNewEventSlot PRIVATE CoScheduler ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
add_test(NAME TestNewEventSlot COMMAND TestNewEventSlot)

## TBB tests
add_executable(TestTBBTaskSuspension TestTBBTaskSuspension.cpp)
target_link_libraries(TestTBBTaskSuspension PUBLIC TBB::tbb CUDA::cudart) 

## CUDA test
add_executable(CUDAGraphTest CUDAGraphs.cu)
target_link_libraries(CUDAGraphTest PRIVATE CUDA::cudart TBB::tbb)

