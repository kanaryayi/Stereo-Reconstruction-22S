#include "EightPoint.h"

EightPointExecuter::EightPointExecuter(std::pair<KeyPoints, KeyPoints> setPair, ImagePair sample) {
	std::cout << "EightPointExecuter >> Initialization Start." << std::endl;
	assert(setPair.first.size() == setPair.second.size());
	m_numPoint = setPair.first.size();

	m_pointSet1 = setPair.first;
	m_pointSet2 = setPair.second;

	m_sample = sample;
	initPossibleRT();
	std::cout << "EightPointExecuter >> Initialization Completed." << std::endl;
}

void EightPointExecuter::initPossibleRT() {
	m_fundamentalMatrix = cv::findFundamentalMat(m_pointSet1, m_pointSet2, cv::FM_8POINT);
	std::cout << "EightPointExecuter >> Fundamental Matrix Found." << std::endl;
	m_essentialMatrix = m_sample.K_img2.t() * m_fundamentalMatrix * m_sample.K_img1;
	std::cout << "EightPointExecuter >> Essential Matrix Calculated." << std::endl;

	cv::Mat R1, R2, T;
	cv::decomposeEssentialMat(m_essentialMatrix, R1, R2, T);

	m_transformations.push_back(std::make_pair(R1,  T));
	m_transformations.push_back(std::make_pair(R1, -T));

	m_transformations.push_back(std::make_pair(R2,  T));
	m_transformations.push_back(std::make_pair(R2, -T));
	std::cout << "EightPointExecuter >> 4 Possible R and T Loaded." << std::endl;
}

std::vector<std::pair<R, T>> EightPointExecuter::getAllPossibleRT() {
	return m_transformations;
}

std::pair<R,T> EightPointExecuter::getValidRT() {
	int possibleCounter = 0;
	int finalIndex = 0;
	int runnerCounter = 0;

	cv::Mat rotation;
	cv::Mat translation;
	for (std::pair<R, T> transform : m_transformations) {
		bool validationResOfTransfrom = isValidRT(transform);
		std::cout << "EightPointExecuter >> Validate Transform " << runnerCounter 
			<< (validationResOfTransfrom ? " (OK)" : " (X)") << "." << std::endl;

		if (validationResOfTransfrom) {
			possibleCounter++;
			rotation = transform.first;
			translation = transform.second;
			finalIndex = runnerCounter;
		}
		runnerCounter++;
	}

	assert(possibleCounter == 1);
	if (possibleCounter != 1) {
		perror("EightPointExecuter >> Multiple R and T possible (X).");
	}
	std::cout << "EightPointExecuter >> All Transformations validated." << std::endl;
	std::cout << "EightPointExecuter >> Only Transform " << finalIndex << " will work." << std::endl;

	return std::make_pair(rotation, translation);
}

bool EightPointExecuter::isValidRT(std::pair<R, T> RTPair) {
	cv::Mat R = RTPair.first;
	cv::Mat T = RTPair.second;

	cv::Mat K1Inv = m_sample.K_img1.inv();
	cv::Mat K2Inv = m_sample.K_img2.inv();
	cv::Mat M = cv::Mat::zeros(cv::Size(m_numPoint + 1, 3 * m_numPoint), CV_64FC1);
	for (int i = 0; i < m_numPoint; i++) {
		cv::Mat pointOfImage1 = (cv::Mat_<double>(3, 1) << m_pointSet1.at(i).x, m_pointSet1.at(i).y, 1.0);
		cv::Mat pointOfImage2 = (cv::Mat_<double>(3, 1) << m_pointSet2.at(i).x, m_pointSet2.at(i).y, 1.0);

		cv::Mat backProjectedPointOfImage1 = K1Inv * pointOfImage1;
		cv::Mat backProjectedPointOfImage2 = K2Inv * pointOfImage2;

		cv::Mat p2SkewMatrix = makeSkewMatrixFromPoint(cv::Point3f(pointOfImage2));

		cv::Mat calculationR = p2SkewMatrix * R * pointOfImage1;
		cv::Mat calculationT = p2SkewMatrix * T;

		M.at<double>(3 * i, i) = calculationR.at<double>(0, 0);
		M.at<double>(3 * i + 1, i) = calculationR.at<double>(1, 0);
		M.at<double>(3 * i + 2, i) = calculationR.at<double>(2, 0);
		M.at<double>(3 * i, m_numPoint) = calculationT.at<double>(0, 0);
		M.at<double>(3 * i + 1, m_numPoint) = calculationT.at<double>(1, 0);
		M.at<double>(3 * i + 2, m_numPoint) = calculationT.at<double>(2, 0);
	}

	cv::Mat V, D;
	cv::eigen(M.t() * M, D, V);

	cv::Mat lambda = cv::Mat::zeros(cv::Size(m_numPoint, 1), CV_64FC1);

	for (int i = 0; i < m_numPoint; i++) {
		lambda.at<double>(0, i) = V.at<double>(m_numPoint, i);
	}

	double gamma = V.at<double>(m_numPoint, m_numPoint);
	if (gamma < 0) {
		gamma = -gamma;
		lambda = -lambda;
	}

	int counter1 = 0;
	int counter2 = 0;
	for (int i = 0; i < m_numPoint; i++) {
		if (lambda.at<double>(0, i) >= 0) {
			counter1++;
		}

		cv::Mat pointOfImage1 = (cv::Mat_<double>(3, 1) << m_pointSet1.at(i).x, m_pointSet1.at(i).y, 1.0);
		cv::Mat pointOfImage2 = (cv::Mat_<double>(3, 1) << m_pointSet2.at(i).x, m_pointSet2.at(i).y, 1.0);

		cv::Mat backProjectedPointOfImage1 = K1Inv * pointOfImage1;
		backProjectedPointOfImage1 *= lambda.at<double>(0, i);

		cv::Mat recoveredPointOfImage2 = R * backProjectedPointOfImage1 + T * gamma;

		if (recoveredPointOfImage2.at<double>(2, 0)) {
			counter2++;
		}
	}

	if (counter1 / m_numPoint == 1 && counter2 / m_numPoint == 1) {
		m_scale = gamma;
		return true;
	} else {
		return false;
	}
}

cv::Mat EightPointExecuter::getEssentialMatrix() {
	return m_essentialMatrix;
}

cv::Mat EightPointExecuter::getFundamentalMatrix() {
	return m_fundamentalMatrix;
}

double EightPointExecuter::getScale() {
	return m_scale;
}