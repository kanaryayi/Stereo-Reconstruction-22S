#include "SURF.h"
#include "ORB.h"
#include "SIFT.h"

#include "EightPoint.h"
#include "Utils.h"

#include <opencv2/core/utils/logger.hpp>

int main(int argc, char** argv) {
	// make OpenCV silent too urusai
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	// Load 8 Image Pairs, we have 23 image for now
	// If you want to load all
	// Use  DataLoader("Middlebury_2014") as Constructor it loads all by default
	DataLoader dataLoader = DataLoader("Middlebury_2014", 2);
	std::cout << std::endl;

	// minHessian Setting https://stackoverflow.com/a/17615172
	SURFDetector surfDectector = SURFDetector(600, 10);
	//ORBDetector orbDectector = ORBDetector(10);
	//SIFTDetector siftDectector = SIFTDetector(10);

	ImagePair randSample = dataLoader.getRandomSample();
	//orbDectector.findCorrespondences(srcImage1, srcImage2);
	std::pair<KeyPoints, KeyPoints> res = surfDectector.findCorrespondences(randSample.img1, randSample.img2);
	//siftDectector.findCorrespondences(srcImage1, srcImage2);
	std::cout << std::endl;

	EightPointExecuter eightPointExecuter = EightPointExecuter(res, randSample);
	std::pair<R,T> validRT = eightPointExecuter.getValidRT();

	cv::waitKey(0);

	return 0;
}
