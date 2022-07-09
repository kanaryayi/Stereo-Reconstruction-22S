#pragma once
#include <iostream>
#include <vector>
#include <exception>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "Utils.h"

// Reference -> https://docs.opencv.org/3.4/d5/dde/tutorial_feature_description.html
class SURFDetector {
public: 
    SURFDetector(int minHessian, int numPoint);

    std::pair<KeyPoints,KeyPoints> findCorrespondences(cv::Mat srcImage1, cv::Mat srcImage2);
private:
    cv::Ptr<cv::xfeatures2d::SURF> detector;
    int m_numPoint = 10;
};