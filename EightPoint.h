#pragma once
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "Utils.h"

class EightPointExecuter {

public:
	EightPointExecuter(std::pair<KeyPoints, KeyPoints> setPair, ImagePair sample);

	std::pair<R,T> getValidRT();
	std::vector<std::pair<R, T>> getAllPossibleRT();
	std::pair<bool, int> isValidRT(std::pair<R, T> RTPair, int runnerCounter);

	cv::Mat getEssentialMatrix();
	cv::Mat getFundamentalMatrix();
	std::vector<double> getGamma();

private:
	KeyPoints m_pointSet1;
	KeyPoints m_pointSet2;

	ImagePair m_sample;
	int m_numPoint;
	std::vector<double> m_gamma;

	cv::Mat m_fundamentalMatrix;
	cv::Mat m_essentialMatrix;

	std::vector<std::pair<R, T>> m_transformations;

	void initPossibleRT();
};