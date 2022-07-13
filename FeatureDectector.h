#include "Utils.h"

enum FeatureDectectorMethod {
    USE_SIFT,
    USE_ORB,
    USE_SURF
};

class FeatureDectector {
public:
    FeatureDectector(int numPoint);
    std::pair<KeyPoints,KeyPoints> findCorrespondences(ImagePair imgPairm, FeatureDectectorMethod fm);

private:
    int m_numPoint;
    cv::Ptr<cv::SIFT> SIFTDetector;
    cv::Ptr<cv::ORB> ORBDetector;
    cv::Ptr<cv::xfeatures2d::SURF> SURFDetector;
};