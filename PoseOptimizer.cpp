#include "PoseOptimizer.h"

PoseOptimizer::PoseOptimizer() {
	options.linear_solver_type = ceres::ITERATIVE_SCHUR;
	options.num_threads = 1;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 100;
}

std::pair<Rotate, Translate> PoseOptimizer::optimizeRT(std::pair<Rotate, Translate> RTPair, 
		std::pair<cv::Mat, double> lambdaGamma,
		std::pair<KeyPoints, KeyPoints> keyPointPairs, 
		ImagePair sample) {
	
	ceres::Problem problem;
	KeyPoints pointSet1 = keyPointPairs.first;
	KeyPoints pointSet2 = keyPointPairs.second;

	cv::Mat lambda = lambdaGamma.first;
	double gamma = lambdaGamma.second;
	cv::Mat R = RTPair.first;
	cv::Mat T = RTPair.second;

	cv::Mat eulerAngles = getEulerAngleByRotationMatrix(R);

	double eulerAngles_[3] = { eulerAngles.at<double>(0,0),
							   eulerAngles.at<double>(1,0),
							   eulerAngles.at<double>(2,0)};

	double translation_[3] = {
		T.at<double>(0,0),
		T.at<double>(1,0),
		T.at<double>(2,0)
	};

	for (int i = 0; i < pointSet1.size(); i++) {
		cv::Mat x1 = (cv::Mat_<double>(3, 1) << pointSet1.at(i).x, pointSet1.at(i).y, 1.0);
		cv::Mat x2 = (cv::Mat_<double>(3, 1) << pointSet2.at(i).x, pointSet2.at(i).y, 1.0);

		ceres::CostFunction* reprojectionError = new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
			new ReprojectionError(x1, x2, sample.K_img1, sample.K_img2, lambda.at<double>(i, 0), gamma)
		);

		ceres::CostFunction* regularizer = new ceres::AutoDiffCostFunction<Regularizer, 2, 3, 3>(
			new Regularizer(0)
		);


		problem.AddResidualBlock(reprojectionError,
			NULL /* squared loss */,
			eulerAngles_,
			translation_);

		problem.AddResidualBlock(regularizer,
			NULL /* squared loss */,
			eulerAngles_,
			translation_);
	}
	
	std::cout << "PoseOptimizer >> All blocks filled, start optimizing..." << std::endl;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << std::endl;

	cv::Mat newEulerAngles = (cv::Mat_<double>(3, 1) << eulerAngles_[0], eulerAngles_[1], eulerAngles_[2]);

	cv::Mat translation = (cv::Mat_<double>(3, 1) << translation_[0], translation_[1], translation_[2]);

	std::cout << "Before: " << eulerAngles << ", After: " << newEulerAngles << std::endl;
	std::cout << "Before: " << T << ", After: " << translation << std::endl;

	return std::make_pair(getRoationMatrixByEulerAngle(newEulerAngles), translation);
}