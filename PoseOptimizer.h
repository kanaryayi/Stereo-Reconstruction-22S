#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "Utils.h"

class ReprojectionError {
public:
	ReprojectionError(cv::Mat _x1, cv::Mat _x2, cv::Mat _K1, cv::Mat _K2, double _lambda, double _gamma) :
		x1{_x1}, x2{_x2}, K1{_K1}, K2{_K2}, lambda{_lambda}, gamma{_gamma} {}
	
	template <typename T>
	bool operator()(const T* const rotation, const T* const translation, T* residuals) const {
		Eigen::Matrix<T, 3, 1> point;
		point[0] = T(x1.at<double>(0, 0));
		point[1] = T(x1.at<double>(1, 0));
		point[2] = T(1.0);
		// cv::Mat point = (cv::Mat_<double>(3, 1) << x1.at<double>(0, 0), x1.at<double>(1, 0), 1.0);
		T lambda_ = T(lambda);
		T gamma_ = T(gamma);

		cv::Mat cvK1Inv = K1.inv();
		
		Eigen::Matrix<T, 3, 3> K1Inv;
		K1Inv << T(cvK1Inv.at<double>(0, 0)), T(cvK1Inv.at<double>(0, 1)), T(cvK1Inv.at<double>(0, 2)),
				 T(cvK1Inv.at<double>(1, 0)), T(cvK1Inv.at<double>(1, 1)), T(cvK1Inv.at<double>(1, 2)),
				 T(cvK1Inv.at<double>(2, 0)), T(cvK1Inv.at<double>(2, 1)), T(cvK1Inv.at<double>(2, 2));
		
		Eigen::Matrix<T, 3, 3> K2Re;
		K2Re << T(K2.at<double>(0, 0)), T(K2.at<double>(0, 1)), T(K2.at<double>(0, 2)),
				T(K2.at<double>(1, 0)), T(K2.at<double>(1, 1)), T(K2.at<double>(1, 2)),
				T(K2.at<double>(2, 0)), T(K2.at<double>(2, 1)), T(K2.at<double>(2, 2));

		Eigen::Matrix<T, 3, 1> deprojectedPoint = K1Inv * point * lambda_;

		const T rotation_[3] = {
			rotation[0], rotation[1], rotation[2]
		};

		const T translation_[3] = {
			translation[0], translation[1], translation[2]
		};

		Eigen::Matrix<T, 3, 1> translationMatrix;
		translationMatrix[0] = translation[0];
		translationMatrix[1] = translation[1];
		translationMatrix[2] = translation[2];

		const double kPi = 3.14159265358979323846;
		const T degrees_to_radians(kPi / 180.0);

		const T pitch(rotation_[0] * degrees_to_radians);
		const T roll(rotation_[1] * degrees_to_radians);
		const T yaw(rotation_[2] * degrees_to_radians);

		const T c1 = ceres::cos(yaw);
		const T s1 = ceres::sin(yaw);
		const T c2 = ceres::cos(roll);
		const T s2 = ceres::sin(roll);
		const T c3 = ceres::cos(pitch);
		const T s3 = ceres::sin(pitch);

		const T R00 = c1 * c2;
		const T R01 = -s1 * c3 + c1 * s2 * s3;
		const T R02 = s1 * s3 + c1 * s2 * c3;

		const T R10 = s1 * c2;
		const T R11 = c1 * c3 + s1 * s2 * s3;
		const T R12 = -c1 * s3 + s1 * s2 * c3;

		const T R20 = -s2;
		const T R21 = c2 * s3;
		const T R22 = c2 * c3;

		Eigen::Matrix<T, 3, 3> rotationMatrix;
		rotationMatrix << R00, R01, R02,
						  R10, R11, R12,
						  R20, R21, R22;


		Eigen::Matrix<T, 3, 1> transformedPoint = rotationMatrix * deprojectedPoint + translationMatrix * gamma_;
		
		Eigen::Matrix<T, 3, 1> reprojectedPoint = K2Re * transformedPoint;

		const T depth = reprojectedPoint[2];
		const T predictedX = reprojectedPoint[0] / depth;
		const T predictedY = reprojectedPoint[1] / depth;

		// std::cout << "Predicted: " << predictedX << ", " << predictedY << std::endl;

		const T observedX = T(x2.at<double>(0, 0));
		const T observedY = T(x2.at<double>(1, 0));
		
		// std::cout << "Observed: " << observedX << ", " << observedY << std::endl;

		residuals[0] = predictedX - observedX;
		residuals[1] = predictedY - observedY;

		//cv::Mat eulerAngles = (cv::Mat_<double>(3, 1) << rotation_[0], rotation_[1], rotation_[2]);
		//
		//cv::Mat Tra = (cv::Mat_<double>(3, 1) << translation_[0], translation_[1], translation_[2]);
		//
		//cv::Mat Rot = getRoationMatrixByEulerAngle(eulerAngles);
		//
		//cv::Mat transformedPoint = Rot * deprojectedPoint + Tra * gamma;

		//cv::Mat reprojectedPoint = K2 * transformedPoint;
		//reprojectedPoint /= reprojectedPoint.at<double>(2, 0);

		//double pX = reprojectedPoint.at<double>(0, 0);
		//double pY = reprojectedPoint.at<double>(1, 0);
		//T predictedX = T(pX);
		//T predictedY = T(pY);

		//double oX = x2.at<double>(0, 0);
		//double oY = x2.at<double>(1, 0);
		//T observedX = T(oX);
		//T observedY = T(oY);

		return true;
	}

private:
	cv::Mat x1; // x y 1
    cv::Mat x2; // x y 1
    cv::Mat K1;
    cv::Mat K2;
	double lambda;
	double gamma;
};

class PoseOptimizer {
public:
	PoseOptimizer();
	void optimizeRT(std::pair<Rotate, Translate> RTPair, 
		std::pair<cv::Mat, double> lambdaGamma,
		std::pair<KeyPoints, KeyPoints> keyPointPairs,
		ImagePair sample);

private:
	ceres::Solver::Options options;
};