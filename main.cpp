#include "SURF.h"
#include "ORB.h"
#include "SIFT.h"

#include "EightPoint.h"
#include "PoseOptimizer.h"
#include "Utils.h"

#include <opencv2/core/utils/logger.hpp>

int main(int argc, char** argv) {
	// make OpenCV silent too urusai
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	// Load 8 Image Pairs, we have 23 image for now
	// If you want to load all
	// Use  DataLoader("Middlebury_2014") as Constructor it loads all by default
	DataLoader dataLoader = DataLoader("Middlebury_2014", -1);

	// minHessian Setting https://stackoverflow.com/a/17615172
	SURFDetector surfDectector = SURFDetector(600, 8);
	//ORBDetector orbDectector = ORBDetector(10);
	//SIFTDetector siftDectector = SIFTDetector(10);
	//orbDectector.findCorrespondences(srcImage1, srcImage2);


	std::cout << std::endl;

#ifdef defined(CHECK_ALL_IMAGEPAIRS)
	// for each img pairs
	for (ImagePair ip : dataLoader.getAllImagePairs()) {
		std::pair<KeyPoints, KeyPoints> res = surfDectector.findCorrespondences(ip.img1, ip.img2);
		//siftDectector.findCorrespondences(srcImage1, srcImage2);
		std::cout << std::endl;

		EightPointExecuter eightPointExecuter = EightPointExecuter(res, ip);
		std::pair<R, T> validRT = eightPointExecuter.getValidRT();

		std::cout << std::endl;
	}
#elif defined(RANDOM_SAMPLE)

	ImagePair randSample = dataLoader.getRandomSample();
	std::cout << "DataLoader >> " << randSample.path << " is selected." << std::endl;
	std::pair<KeyPoints, KeyPoints> res = surfDectector.findCorrespondences(randSample.img1, randSample.img2);
	std::cout << std::endl;

	EightPointExecuter eightPointExecuter = EightPointExecuter(res, randSample);
	std::pair<R, T> validRT = eightPointExecuter.getValidRT();

#else 
	ImagePair specSample = dataLoader.getSpecificSample("Pipe");
	std::cout << "DataLoader >> " << specSample.path << " is selected." << std::endl;
	std::pair<KeyPoints, KeyPoints> res = surfDectector.findCorrespondences(specSample.img1, specSample.img2);
	std::cout << std::endl;
	EightPointExecuter eightPointExecuter = EightPointExecuter(res, specSample);
	std::pair<Rotate, Translate> validRT = eightPointExecuter.getValidRT();
	std::pair<cv::Mat, double> lambdaGamma = eightPointExecuter.getLambdaGamma();
	//Rotate R = validRT.first;
	//Translate T = validRT.second;
	//std::cout << "Tra " << tra << std::endl;
	//std::cout << rot << std::endl;
	//cv::Mat eulerAngles = getEulerAngleByRotationMatrix(R);
	//std::cout << eulerAngles << std::endl;
	//std::cout << getRoationMatrixByEulerAngle(eulerAngles) << std::endl;

	std::cout << std::endl;

	PoseOptimizer poseOptimizer = PoseOptimizer();
	poseOptimizer.optimizeRT(validRT, lambdaGamma, res, specSample);
#endif
	return 0;
}
