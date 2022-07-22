#include "FeatureDectector.h"
#include "EightPoint.h"
#include "PoseOptimizer.h"
#include "BlockMatcher.h"
#include "Utils.h"

#include "PFMManager.h"

#include "PointCloud.h"
#include "Reconstruction.h"

#include "Eigen.h"

int testMeshGeneration() {
	DataLoader dataloader = DataLoader("Middlebury_2014/Adirondack-perfect", 1, true);
	ImagePair downsampled = dataloader.getImagePairByIndex(0)
		.sampleDown(0.3);

	std::stringstream ss;
	ss << "reconstructed.off";

	PointCloud pc{};
	float maxDepth = 0.0;

	cv::Mat depthMap = pc.depthMapFromDisperityMap(downsampled.disp0, downsampled.baseline,
													downsampled.doffs, downsampled.f1, &maxDepth, false);
	

	
	Vertex* vertices = pc.generatePointCloud(depthMap, downsampled.img1, downsampled.K_img1, maxDepth);

	if (!Mesh::writeMesh(vertices, depthMap.cols, depthMap.rows, ss.str())) {
		std::cerr << "Could not write mesh " << ss.str() << "\n";
	}

	delete[] vertices;

	return 0;
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
		FeatureDectector detector = FeatureDectector(12);
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
	std::pair<Rotate, Translate> validRT = eightPointExecuter.getValidRT();
	std::pair<cv::Mat, double> lambdaGamma = eightPointExecuter.getLambdaGamma();

	std::pair<Rotate, Translate> openCVRT = eightPointExecuter.tryOpenCVPiepline();

	Rotate R = validRT.first;
	Translate T = validRT.second;
	std::cout << "R: " << R << std::endl;
	std::cout << "T: " << T << std::endl;

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