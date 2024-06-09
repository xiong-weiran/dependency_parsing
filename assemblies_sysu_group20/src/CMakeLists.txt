cmake_minimum_required(VERSION 3.10)

# 项目名称
project(ProjectName)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加源文件
set(SOURCES
    brain.cpp
    main.cpp
)

# 查找 Boost 库
find_package(Boost REQUIRED COMPONENTS random math_c99 system)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    add_executable(project ${SOURCES})
    target_link_libraries(project ${Boost_LIBRARIES})
endif()
