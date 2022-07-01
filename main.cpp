#include "SURF.h"
#include "ORB.h"


int main(int argc, char** argv) {
	
	// minHessian Setting https://stackoverflow.com/a/17615172
	SURFDetector surfDectector = SURFDetector(600, 1);
	ORBDetector orbDectector = ORBDetector(3.5);

	cv::Mat srcImage1 = cv::imread("../data/Middlebury_2014/Adirondack-perfect/im0.png");
	cv::Mat srcImage2 = cv::imread("../data/Middlebury_2014/Adirondack-perfect/im1.png");

	orbDectector.findCorrespondences(srcImage1, srcImage2);
	surfDectector.findCorrespondences(srcImage1, srcImage2);


	cv::waitKey(0);

	return 0;
}
