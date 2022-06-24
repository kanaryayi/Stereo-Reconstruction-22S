#include "SURF.h"



int main(int argc, char** argv) {
	
	SURFDetector surfDectector = SURFDetector(700, true);

	surfDectector.findCorrespondences("../data/Middlebury_2014/Adirondack-perfect/im0.png",
									  "../data/Middlebury_2014/Adirondack-perfect/im1.png");
	cv::waitKey();

	return 0;
}
