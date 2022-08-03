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

std::pair<Rotate, Translate> EightPointExecuter::tryOpenCVPiepline() {
	cv::Mat E, R, t, mask;
	cv::recoverPose(m_pointSet1, m_pointSet2,m_sample.K_img1, cv::noArray(), m_sample.K_img2,cv::noArray(),E, R, t,cv::RANSAC, 0.999, 1.0, mask);
	std::cout << "R1: " << R << std::endl;
	std::cout << "t1: " << t << " norm :: "<< cv::norm(t)<<std::endl;
	// std::cout << mask<< std::endl;
	// cv::recoverPose(E, m_pointSet1, m_pointSet2, m_sample.K_img2, R, t, mask);
	// std::cout << "R2: " << R << std::endl;
	// std::cout << "t2: " << t << std::endl;
	return std::make_pair(R, t);
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

std::vector<std::pair<Rotate, Translate>> EightPointExecuter::getAllPossibleRT() {
	return m_transformations;
}

std::pair<Rotate,Translate> EightPointExecuter::getValidRT() {
	int possibleCounter = 0;
	int finalIndex = 0;
	int runnerCounter = 0;
	bool realAnswerFound = false;
	cv::Mat rotation;
	cv::Mat translation;
	for (std::pair<Rotate, Translate> transform : m_transformations) {
		std::pair<bool, int> validationResOfTransfrom = isValidRT(transform, runnerCounter);
		std::cout << "EightPointExecuter >> Validate Transform " << runnerCounter 
			<< (validationResOfTransfrom.first ? (" (OK) with " 
			+ std::to_string(validationResOfTransfrom.second * 1.0 / m_numPoint * 1.0) + "%") : " (X)") << "." << std::endl;

		if (validationResOfTransfrom.first) {
			possibleCounter++;
			if (validationResOfTransfrom.second == m_numPoint) {
				rotation = transform.first;
				translation = transform.second;
				//translation *= m_sample.baseline / cv::norm(translation);
				finalIndex = runnerCounter;
				realAnswerFound = true;
			}
			
			if (!realAnswerFound) {
				rotation = transform.first;
				translation = transform.second;
				//translation *= m_sample.baseline / cv::norm(translation);
				finalIndex = runnerCounter;
			}
		}
		runnerCounter++;
	}

	//assert(possibleCounter == 1);
	if (possibleCounter != 1 && realAnswerFound) {
		std::cout << "EightPointExecuter >> Although multiple R and T found, but we choose perfect one." << std::endl;
	} else if (possibleCounter != 1 && !realAnswerFound) {
		perror("EightPointExecuter >> Multiple R and T possible (X).");
		std::cout << std::endl;
		std::cout << m_sample.path << " Bad result!" << std::endl;
		std::cout << std::endl;
	}

	if (!realAnswerFound) {
		std::cout << "EightPointExecuter >> Warning: Bad R and T is used." << std::endl;
	}


	std::cout << "EightPointExecuter >> All Transformations validated." << std::endl;
	std::cout << "EightPointExecuter >> Only Transform " << finalIndex << " will work." << std::endl;

	return std::make_pair(rotation, translation);
}

std::pair<bool, int> EightPointExecuter::isValidRT(std::pair<Rotate, Translate> RTPair, int runnerCounter) {
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

		cv::Mat p2SkewMatrix = makeSkewMatrixFromPoint(cv::Point3f(backProjectedPointOfImage2));

		cv::Mat calculationR = p2SkewMatrix * R * backProjectedPointOfImage1;
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

	cv::Mat lambda = cv::Mat::zeros(cv::Size(1, m_numPoint), CV_64FC1);

	for (int i = 0; i < m_numPoint; i++) {
		lambda.at<double>(i, 0) = V.at<double>(m_numPoint, i);
	}

	double gamma = V.at<double>(m_numPoint, m_numPoint);

	if (gamma < 0) {
		gamma = -gamma;
		lambda = -lambda;
	}

	// lambda = lambda / gamma;

	int counter1 = 0;
	int counter2 = 0;
	for (int i = 0; i < m_numPoint; i++) {
		if (lambda.at<double>(i, 0) >= 0.0) {
			counter1++;
		}
		
		cv::Mat pointOfImage1 = (cv::Mat_<double>(3, 1) << m_pointSet1.at(i).x, m_pointSet1.at(i).y, 1.0);
		cv::Mat pointOfImage2 = (cv::Mat_<double>(3, 1) << m_pointSet2.at(i).x, m_pointSet2.at(i).y, 1.0);

		cv::Mat backProjectedPointOfImage1 = K1Inv * pointOfImage1 * lambda.at<double>(i, 0);
		cv::Mat backProjectedPointOfImage2 = K2Inv * pointOfImage2;
		//T *= (m_sample.baseline / 1000) / (cv::norm(T));
		//std::cout << "Depth: " << lambda.at<double>(i, 0) << std::endl;
		cv::Mat recoveredPointOfImage2 = R * backProjectedPointOfImage1 + T * gamma;

		//std::cout << "Ori: " << backProjectedPointOfImage2*(recoveredPointOfImage2.at<double>(2, 0)) << std::endl;
		//std::cout << "Rec: " << recoveredPointOfImage2 << std::endl;
		
		//std::cout << (m_sample.K_img2 * recoveredPointOfImage2) / recoveredPointOfImage2.at<double>(2, 0) << std::endl;
		//std::cout << pointOfImage2 << std::endl;

		if (recoveredPointOfImage2.at<double>(2, 0) >= 0) {
			counter2++;
		}
	}

	std::cout << "Debug << " << "Counter 1 = " << counter1 << " Counter 2 = " << counter2 << std::endl;
	if (counter1 / m_numPoint == 1 && counter2 / m_numPoint == 1) {
		//m_gamma.at(runnerCounter) = gamma;
		m_gamma = gamma;
		m_lambda = lambda;
		return std::make_pair(true, m_numPoint);
	} else if (counter1 - counter2 == 0 && counter1 + counter2 >= m_numPoint) {
		//m_gamma.at(runnerCounter) = gamma;
		return std::make_pair(true, counter1);
	} else {
		return std::make_pair(false, 0);
	}
}

cv::Mat EightPointExecuter::getEssentialMatrix() {
	return m_essentialMatrix;
}

cv::Mat EightPointExecuter::getFundamentalMatrix() {
	return m_fundamentalMatrix;
}

std::pair<cv::Mat, double> EightPointExecuter::getLambdaGamma() {
	return std::make_pair(m_lambda, m_gamma);
}