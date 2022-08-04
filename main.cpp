#include "FeatureDectector.h"
#include "EightPoint.h"
#include "PoseOptimizer.h"
#include "BlockMatcher.h"
#include "Utils.h"

#include "PFMManager.h"

#include "PointCloud.h"
#include "DenseMatching.h"
#include "Reconstruction.h"

#include "Eigen.h"

void compare_sgbm_bm() {
    std::string name{"Umbrella-perfect"};

    DataLoader dataloader = DataLoader("Middlebury_2014/" + name, 1, true);
    ImagePair sample = dataloader.getImagePairByIndex(0);

    DenseMatching::evaluateSGBM(sample, name, 7, 260);
    DenseMatching::evaluateBM(sample, name, 21, 256);

    PointCloud pc{};
    float max_depth;

    cv::Mat depth = pc.depthMapFromDisperityMap(sample.disp0, sample.baseline, sample.doffs, sample.f1, &max_depth, true);

    cv::resize(sample.img1, sample.img1, cv::Size(0.4 * sample.img1.cols, 0.4 * sample.img1.rows), 0, 0, cv::INTER_LINEAR);
    cv::resize(sample.disp0, sample.disp0, cv::Size(0.4 * sample.disp0.cols, 0.4 * sample.disp0.rows), 0, 0, cv::INTER_LINEAR);
    cv::resize(depth, depth, cv::Size(0.4 * depth.cols, 0.4 * depth.rows), 0, 0, cv::INTER_LINEAR);

    cv::imwrite(name + "_disp_ground.png", sample.disp0);
    cv::imwrite(name + "_img.png", sample.img1);

    cv::applyColorMap(sample.disp0, sample.disp0, cv::COLORMAP_JET);
    cv::imwrite(name + "_disp_ground_jet.png", sample.disp0);

    depth = 255 * depth;
    cv::imwrite(name + "_depth_ground.png", depth);
}


int main(int argc, char** argv) {
	// make OpenCV silent too urusai
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	// If you want to load all
	// Use  DataLoader("Middlebury_2014") as Constructor it loads all by default
	// Warning: will consume about 2GB memory

	// Load a sepcific num of imagePair => use like 12 for 12 points
	// DataLoader dataLoader = DataLoader("Middlebury_2014", 12);

	// Load a specific imagePair => use -1
	DataLoader dataLoader = DataLoader("Middlebury_2014", -1);


	if (SPARSE_MATCHING) {
		FeatureDectector detector = FeatureDectector(50);
		std::cout << std::endl;

#ifdef defined(ALL_SAMPLE)
	// for each img pairs
	for (ImagePair ip : dataLoader.getAllImagePairs()) {
		std::pair<KeyPoints, KeyPoints> res = detector.findCorrespondences(ip, USE_SURF);
		std::cout << std::endl;
		EightPointExecuter eightPointExecuter = EightPointExecuter(res, ip);
		std::pair<R, T> validRT = eightPointExecuter.getValidRT();
		std::cout << std::endl;
	}
#elif defined(RANDOM_SAMPLE)
	ImagePair randSample = dataLoader.getRandomSample();
	std::cout << "DataLoader >> " << randSample.path << " is selected." << std::endl;
	std::pair<KeyPoints, KeyPoints> res = surfDectector.findCorrespondences(randSample, USE_SURF);
	std::cout << std::endl;

	EightPointExecuter eightPointExecuter = EightPointExecuter(res, randSample);
	std::pair<R, T> validRT = eightPointExecuter.getValidRT();
#elif defined(SPEC_SAMPLE)
	ImagePair specSample = dataLoader.getSpecificSample("Pipe");
	std::cout << "DataLoader >> " << specSample.path << " is selected." << std::endl;
	std::pair<KeyPoints, KeyPoints> res = detector.findCorrespondences(specSample, USE_SURF);
	std::cout << std::endl;
	EightPointExecuter eightPointExecuter = EightPointExecuter(res, specSample);

	std::pair<Rotate, Translate> openCVRT = eightPointExecuter.tryOpenCVPiepline();
	cv::Mat R, t;
	R = openCVRT.first;
	t = openCVRT.second;
	t *= (specSample.baseline * 0.001);
	// scale of translation vector is uncertain but its norm is 1. so we have to scale it with an additional info
	std::cout << "Absolute Translation : "<< t << std::endl; // baseline is in mm, we want to represent in meter

	//cv::Mat eulerAngles = getEulerAngleByRotationMatrix(R);
	//std::cout << eulerAngles << std::endl;
	//std::cout << getRoationMatrixByEulerAngle(eulerAngles) << std::endl;

	//std::cout << std::endl;

	// Really bad result Why?
	// PoseOptimizer poseOptimizer = PoseOptimizer();
	// poseOptimizer.optimizeRT(openCVRT, lambdaGamma, res, specSample);

	// poseOptimizer.optimizeRT(validRT, lambdaGamma, res, specSample);

	} else if (DENSE_MATCHING) {
		ImagePair specSample = dataLoader.getSpecificSample("Pipe");
		std::cout << "DataLoader >> " << specSample.path << " is selected." << std::endl;
		BlockMatcher blockMatcher = BlockMatcher(specSample);
		blockMatcher.performBlockMatching(15, 160, USE_BM, false); // BM, NO WLS
		blockMatcher.performBlockMatching(15, 160, USE_BM, true); // BM, WLS
		blockMatcher.performBlockMatching(3, 160, USE_SGBM, false); // SGBM, NO WLS
		blockMatcher.performBlockMatching(3, 160, USE_SGBM, true); // BM, WLS	
	} else {
		std::cerr << "MAIN >> Nothing to do, check your MATCHING Marcos." << std::endl;
	}
#else
	std::cerr << "MAIN >> Nothing to do, check your SAMPLE Marcos."
#endif

	testMeshGeneration();

	return 0;
}