#include "recoverPose.h"

namespace factorizedEightPoint
{

cv::Mat deprojectPoints(cv::Mat imagePoints, cv::Mat cameraMatrix)
{
    cv::Mat inverseCameraMatrix;
    cv::invert(cameraMatrix, inverseCameraMatrix);
    return inverseCameraMatrix * imagePoints;
}
cv::Mat undistortPoints(cv::Mat points, cv::Mat cameraMatrix, cv::Mat distCoeffients)
{
    cv::Mat undistortedPoints;
    cv::undistortPoints(points, undistortedPoints, cameraMatrix, distCoeffients);
    return undistortedPoints;
}
cv::Mat estimateEssentialMatrix(cv::Mat points1, cv::Mat points2, cv::Mat cameraMatrix1, cv::Mat cameraMatrix2)
{
    /*
     * @brief 
     * apply ransac and estimate EssentialMatrix  
     */
    cv::Mat outliersMask;
    return cv::findEssentialMat(points1,points2,cameraMatrix1,cameraMatrix2,cv::noArray(),cv::noArray(),METHOD,RANSAC_PROB,RANSAC_THRESHOLD, outliersMask);

}
void recoverRotationTranslation(cv::Mat essentialMatrix, cv::Mat &rotationMatrix, cv::Mat &translationVector)
{
    cv::Mat U, W, Vt;
    cv::SVDecomp(essentialMatrix,W, U, Vt, 0);
    if (cv::determinant(U * Vt) == -1){
        //W
        cv::Mat W(3, 3, CV_64F, cv::Scalar(0));
        W.at<double>(0, 0) = 1;
        W.at<double>(1, 1) = 1;
        W.at<double>(2, 2) = -1;
        rotationMatrix = U * W * Vt;
    }
    else{
        rotationMatrix = U * W *Vt;
    }
    translationVector = U.col(2);
    // TODO the transformation matrix shuold be checked
}
}