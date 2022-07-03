#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

namespace factorizedEightPoint
{

using namespace std;

enum Algo {fivePoint};

int METHOD = cv::RANSAC;
double RANSAC_PROB = 0.999;
double RANSAC_THRESHOLD = 1.0;

cv::Mat deprojectPoints(vector<cv::DMatch> imagePoints, cv::Mat cameraMatrix);
cv::Mat undistortPoints(cv::Mat points, cv::Mat cameraMatrix, cv::Mat distCoeffients);
cv::Mat estimateEssentialMatrix(cv::Mat points1, cv::Mat points2, cv::Mat cameraMatrix1, cv::Mat cameraMatrix2);
void recoverRotationTranslation(cv::Mat essentialMatrix, cv::Mat &rotationMatrix, cv::Mat &translationVector);

}