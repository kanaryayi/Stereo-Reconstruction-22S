#ifndef SIFT_H
#define SIFT_H

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "Utils.h"

class SIFTDetector {
    public: 
        SIFTDetector(int numPoint);

        std::pair<KeyPoints,KeyPoints> findCorrespondences(cv::Mat srcImage1, cv::Mat srcImage2);
    private:
        cv::Ptr<cv::SIFT> detector;
        int m_numPoint = 10;
};

#endif