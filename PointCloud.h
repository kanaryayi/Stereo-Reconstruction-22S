#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include <iostream>
#include <opencv2/core/core.hpp>

#include "Reconstruction.h"

// Print debug information when set to 1
#define POINT_CLOUD_DEBUG 1

#define EPSILON 10e-5

class PointCloud {

    public:
        // Compute a depth from from the given disperity map in `disperity` using the
        // intrinsic parameters provided by the `intrisicMat` matrix. If `normalize` is
        // true, the computed depth values are scaled to the interval [0-1].
        // The computed depth map is of type CV_32FC1, this a 1-channel 32-bit floating point matrix.
        cv::Mat depthMapFromDisperityMap(cv::Mat disperity, float baseline, float doffs, float focal, float* maxDepth, bool normalize = false);

        cv::Mat depthMapFromNormDisperity(cv::Mat disperity, float baseline, float doffs, float focal, float* maxDepth, bool normalize = false);


        // Generate a point cloud given a depth map in `depthMap` with the colors provided by `colorMap`.
        // Returns a pointer to an array of vertices which is the same size as the color/depth image.
        // The array of vertices needs to be deallocated after use.
        // `maxDepth` is the maximum depth value that is attained in the depth map.
        Vertex* generatePointCloud(cv::Mat depthMap, cv::Mat colorMap, cv::Mat depthIntrinsicMat, float maxDepth = 1.0);
};

#endif