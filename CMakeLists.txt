cmake_minimum_required(VERSION 3.16)
project(bassinet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

file(GLOB LIB_SOURCES "src/*.cpp")
add_library(bassinet STATIC ${LIB_SOURCES})

# Config GTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parents project's compiler settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Tests
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "test/*_test.cpp" "test/*_it.cpp")
add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests PRIVATE bassinet gtest_main)

include(GoogleTest)
gtest_discover_tests(run_tests)
