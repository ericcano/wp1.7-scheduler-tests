// Mimic "assert always".
#ifdef NDEBUG
#undef NDEBUG
#endif


#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "EventStore.hpp"
#include "StatusCode.hpp"

#include <gtest/gtest.h>


class EventStoreTest : public ::testing::Test {
protected:
    EventStore store;
    std::string defaultName = "ObjectName";
    int defaultValue = 1;
    EventStoreTest() {
        EXPECT_TRUE(store.record(std::make_unique<int>(defaultValue), defaultName));
    }
    template <typename T>
    void run_contains(const std::string& name) {
        EXPECT_TRUE(store.contains<T>(name));
    }
    template <typename T>
    void run_contains_fail(const std::string& name) {
        EXPECT_FALSE(store.contains<T*>(name));
        EXPECT_FALSE(store.contains<T[]>(name));
        EXPECT_FALSE(store.contains<T>(name + " "));
        EXPECT_FALSE(store.contains<T>(name + " "));
    }
    template <typename T>
    void run_retrieve(T x, const std::string& name) {
        const T* yptr = nullptr;
        EXPECT_TRUE(store.retrieve(yptr, name));
        EXPECT_EQ(x, *yptr);
    }
    template <typename T>
    void run_retrieve_fail(const std::string& name) {
        const std::nullptr_t* null_val = nullptr;
        EXPECT_FALSE(store.retrieve(null_val, name));
        const T* yptr = nullptr;
        EXPECT_FALSE(store.retrieve(yptr, name + " "));
    }
    template <typename T>
    void run_record_repeat(const std::string& name) {
        EXPECT_TRUE(store.record(std::make_unique<T>(1), name + " "));
    }
    template <typename T>
    void run_record_repeat_fail(const std::string& name) {
        EXPECT_FALSE(store.record(std::make_unique<T>(1), name));
        EXPECT_FALSE(store.record(std::make_unique<std::nullptr_t>(nullptr), name));
    }
};

TEST_F(EventStoreTest, RecordObject) {
    // This test checks that the object is recorded in the constructor
    EventStore localStore;
    EXPECT_TRUE(localStore.record(std::make_unique<int>(1), "ObjectName"));
}

TEST_F(EventStoreTest, Contains) {
    run_contains<int>(defaultName);
}

TEST_F(EventStoreTest, ContainsFail) {
    run_contains_fail<int>(defaultName);
}

TEST_F(EventStoreTest, Retrieve) {
    run_retrieve<int>(defaultValue, defaultName);
}

TEST_F(EventStoreTest, RetrieveFail) {
    run_retrieve_fail<int>(defaultName);
}

TEST_F(EventStoreTest, RecordRepeat) {
    run_record_repeat<int>(defaultName);
}

TEST_F(EventStoreTest, RecordRepeatFail) {
    run_record_repeat_fail<int>(defaultName);
}

// Standard gtest main
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
