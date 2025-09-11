cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(MyCudaProject LANGUAGES CXX CUDA)

# Set CUDA architecture to Hopper (SM90)
set(CMAKE_CUDA_ARCHITECTURES 90)

# Enable C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")  # 启用可重定位设备代码

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cutlass/include
)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

# Build main library from src directory
file(GLOB_RECURSE SRC_FILES "src/*.cu" "src/*.cpp")
add_library(launchchain_kernel STATIC ${SRC_FILES})

# Set target properties for CUDA compilation
set_target_properties(launchchain_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON  # 确保设备符号被解析
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(launchchain_kernel PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link against CUDA libraries
target_link_libraries(launchchain_kernel PUBLIC CUDA::cudart)

# Build tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    file(GLOB_RECURSE TEST_FILES "test/*.cu" "test/*.cpp")
    
    foreach(test_file ${TEST_FILES})
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file})
        target_link_libraries(${test_name} PRIVATE launchchain_kernel)
        target_include_directories(${test_name} PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        # 为测试可执行文件也启用可重定位设备代码
        set_target_properties(${test_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()