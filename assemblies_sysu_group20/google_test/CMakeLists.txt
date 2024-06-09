cmake_minimum_required(VERSION 3.10)

# 项目名称
project(ParserProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加源文件
set(SOURCES
    ${CMAKE_SOURCE_DIR}/test_parse.cpp
)

# 查找 Boost 库
find_package(Boost REQUIRED COMPONENTS random math_c99 system)

# 下载并配置 Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# 添加测试
add_executable(parser_project ${SOURCES})
target_link_libraries(parser_project
    ${Boost_LIBRARIES}
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(parser_project)
