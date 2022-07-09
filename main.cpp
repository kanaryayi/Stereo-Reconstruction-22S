#include "SURF.h"
#include "ORB.h"
#include "SIFT.h"
#include "Utils.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

int main(int argc, char** argv) {
	
	
	// Load 8 Image Pairs, we have 23 image for now
	// If you want to load all
	// Use  DataLoader("Middlebury_2014") as Constructor it loads all by default
	DataLoader dataLoader = DataLoader("Middlebury_2014", 8);

	// minHessian Setting https://stackoverflow.com/a/17615172
	SURFDetector surfDectector = SURFDetector(600, 10);
	//ORBDetector orbDectector = ORBDetector(10);
	//SIFTDetector siftDectector = SIFTDetector(10);

	MiddleburyImagePair randomSample = dataLoader.getRandomSample();
	//orbDectector.findCorrespondences(srcImage1, srcImage2);
	std::pair<KeyPoints, KeyPoints> res = surfDectector.findCorrespondences(randomSample.img1, randomSample.img2);
	//siftDectector.findCorrespondences(srcImage1, srcImage2);

	//cv::Mat fundamental_mat = cv::findFundamentalMat(res.first, res.second, cv::FM_8POINT);



	cv::waitKey(0);

	return 0;
}
