cmake_minimum_required(VERSION 3.6)

SET(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)
# SET(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE PATH "Path to your vcpkg")


PROJECT(stereo_reconstruction_ss22)

SET(CMAKE_BUILD_TYPE Debug)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Ceres REQUIRED)
# FIND_PACKAGE(flann REQUIRED)

SET(HEADERS
    FeatureDectector.h
    PoseOptimizer.h
    BlockMatcher.h
    Utils.h
    FivePoint.h
    Reconstruction.h
    PFMManager.h
    PointCloud.h
    DenseMatching.h
)

SET(SOURCES
    FeatureDectector.cpp
    FivePoint.cpp
    BlockMatcher.cpp
    PoseOptimizer.cpp
    Reconstruction.cpp
    PFMManager.cpp
    PointCloud.cpp
    DenseMatching.cpp
        main.cpp)

add_executable(stereo_reconstruction_ss22 ${HEADERS} ${SOURCES})

LINK_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

target_include_directories(stereo_reconstruction_ss22 PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(stereo_reconstruction_ss22 PUBLIC ${EIGEN3_INCLUDE_DIR})

# target_link_libraries(stereo_reconstruction_ss22 flann::flann flann::flann_cpp)
TARGET_LINK_LIBRARIES(stereo_reconstruction_ss22 ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(stereo_reconstruction_ss22 Eigen3::Eigen)
TARGET_LINK_LIBRARIES(stereo_reconstruction_ss22 ceres Ceres::ceres)
