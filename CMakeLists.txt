cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(MyCudaProject LANGUAGES CXX CUDA)

# 先设置CUDA架构（在find_package(Torch)之前）
set(CMAKE_CUDA_ARCHITECTURES 90)
set(TORCH_CUDA_ARCH_LIST "9.0")  # 告诉libtorch只使用9.0架构

# 手动下载 libtorch
set(LIBTORCH_DIR ${CMAKE_SOURCE_DIR}/thirdparty/libtorch)
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.8.0%2Bcu126.zip")
set(LIBTORCH_ZIP ${CMAKE_SOURCE_DIR}/thirdparty/libtorch.zip)

# 检查是否已经存在 libtorch
if(NOT EXISTS ${LIBTORCH_DIR})
    message(STATUS "Downloading libtorch...")
    file(DOWNLOAD 
        ${LIBTORCH_URL}
        ${LIBTORCH_ZIP}
        SHOW_PROGRESS
        STATUS download_status
    )
    
    list(GET download_status 0 status_code)
    if(NOT status_code EQUAL 0)
        message(FATAL_ERROR "Failed to download libtorch")
    endif()
    
    message(STATUS "Extracting libtorch...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${LIBTORCH_ZIP}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/thirdparty
        RESULT_VARIABLE extract_result
    )
    
    if(NOT extract_result EQUAL 0)
        message(FATAL_ERROR "Failed to extract libtorch")
    endif()
    
    # 删除下载的zip文件
    file(REMOVE ${LIBTORCH_ZIP})
endif()

# 设置 libtorch 路径
set(CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
find_package(Torch REQUIRED)

# 清理libtorch自动添加的CUDA架构flags
set(CMAKE_CUDA_FLAGS "")
set(CUDA_NVCC_FLAGS "")

# 只设置我们需要的SM90架构
set(CMAKE_CUDA_ARCHITECTURES 90)

# 设置C++标准为C++17（libtorch需要）- 但移除其CUDA flags
string(REGEX REPLACE "-gencode[^ ]*" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Enable C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add CUDA flags (只添加我们需要的)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")  # 启用可重定位设备代码

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cutlass/include
    ${TORCH_INCLUDE_DIRS}  # 添加 libtorch 头文件路径
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
    CUDA_ARCHITECTURES 90  # 明确指定只用SM90
)

target_include_directories(launchchain_kernel PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link against CUDA libraries and libtorch
target_link_libraries(launchchain_kernel PUBLIC 
    CUDA::cudart
    "${TORCH_LIBRARIES}"  # 链接 libtorch
)

# Build tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    file(GLOB_RECURSE TEST_FILES "test/*.cu" "test/*.cpp")
    
    foreach(test_file ${TEST_FILES})
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file})
        target_link_libraries(${test_name} PRIVATE 
            launchchain_kernel
            "${TORCH_LIBRARIES}"  # 测试也链接 libtorch
        )
        target_include_directories(${test_name} PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS}
            ${TORCH_INCLUDE_DIRS}
        )
        # 为测试可执行文件也启用可重定位设备代码
        set_target_properties(${test_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_ARCHITECTURES 90  # 明确指定只用SM90
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()