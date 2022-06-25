#include "SURF.h"
#include "ORB.h"


int main(int argc, char** argv) {
	
	SURFDetector surfDectector = SURFDetector(700, true);
	ORBDetector orbDectector = ORBDetector(true);

	surfDectector.findCorrespondences("../data/Middlebury_2014/Adirondack-perfect/im0.png",
									  "../data/Middlebury_2014/Adirondack-perfect/im1.png");

	orbDectector.findCorrespondences("../data/Middlebury_2014/Adirondack-perfect/im0.png",
									  "../data/Middlebury_2014/Adirondack-perfect/im1.png");
	cv::waitKey(0);

	return 0;
}
