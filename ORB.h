#pragma once
#include <iostream>
#include <vector>
#include <exception>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

// Reference -> https://docs.opencv.org/3.4/d5/dde/tutorial_feature_description.html
class ORBDetector {
public: 
    ORBDetector(float filteringFactor);

    std::vector<cv::DMatch> findCorrespondences(cv::Mat srcImage1, cv::Mat srcImage2);
private:
    cv::Ptr<cv::ORB> detector;
    float m_filteringFactor = 0;
};