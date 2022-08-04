#ifndef DENSE_MATCHING_H
#define DENSE_MATCHING_H

#include <iostream>
#include <stdexcept>
#include <vector>

#include "PointCloud.h"
#include "Reconstruction.h"

enum MatchingMethod {
    OPENCV_BM,
    OPENCV_SGBM,

    SAD
};

enum EvaluationMetric {
    RMS,
    BAD_1,
    BAD_2,
    BAD_5
};

class DenseMatching {
    public:
        /*
         * Use dense matching to compute a disperity map.
         */
        static cv::Mat execute(const cv::Mat& left, const cv::Mat& right, MatchingMethod method, int block_size = 19, int num_disp = 260);

        /*
         * Compare the computed disperity map with the ground truth using different metrics.
         *
         * ground_truth/computed are both assumed to be in the CV_8U format.
         */
        static float evaluate(cv::Mat ground_truth, cv::Mat computed, EvaluationMetric metric);

        static void evaluateSGBM(const ImagePair& sample, const std::string& img_name, int block_size, int num_disp);
        static void evaluateBM(const ImagePair& sample, const std::string& img_name, int block_size, int num_disp);
};

#endif