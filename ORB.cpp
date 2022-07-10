#include "ORB.h"

ORBDetector::ORBDetector(int numPoint) {
    detector = cv::ORB::create();
    m_numPoint = numPoint;
}

std::pair<KeyPoints,KeyPoints> ORBDetector::findCorrespondences(cv::Mat srcImage1, cv::Mat srcImage2) {
    if (srcImage1.empty() || srcImage2.empty()) {
        throw std::out_of_range("ORBDetector >> Fail to Load the Image.");
    }
    else {
        std::cerr << "ORBDetector >> Images successfully loaded." << std::endl;
    }

    std::vector<cv::KeyPoint>keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    detector.get()->detect(srcImage1, keypoints1);
    detector.get()->detect(srcImage2, keypoints2);

    detector.get()->compute(srcImage1, keypoints1, descriptors1);
    detector.get()->compute(srcImage2, keypoints2, descriptors2);

    if (descriptors1.empty())
        cv::error(0, "MatchFinder", "1st descriptor empty", __FILE__, __LINE__);
    if (descriptors2.empty())
        cv::error(0, "MatchFinder", "2nd descriptor empty", __FILE__, __LINE__);

    descriptors1.convertTo(descriptors1, CV_32F);
    descriptors2.convertTo(descriptors2, CV_32F);

    std::vector<cv::DMatch>matches;
    std::vector<cv::DMatch>good_matches;
    cv::FlannBasedMatcher matcher;

    matcher.match(descriptors1, descriptors2, matches);

    std::sort(matches.begin(), matches.end(), distanceSorting);

    KeyPoints image1Points;
    KeyPoints image2Points;

    if (matches.size() < m_numPoint) {
        std::cerr << "ORBDetector >> Error didn't get enough feature points." << std::endl;
    } else {
        for (int i = 0; i < m_numPoint; i++) {
            cv::KeyPoint keyPoint1 = keypoints1.at(matches[i].queryIdx);
            cv::Point2f point1 = cv::Point2f(keyPoint1.pt.x, keyPoint1.pt.y);

            cv::KeyPoint keyPoint2 = keypoints2.at(matches[i].trainIdx);
            cv::Point2f point2 = cv::Point2f(keyPoint2.pt.x, keyPoint2.pt.y);

            image1Points.push_back(point1);
            image2Points.push_back(point2);
            good_matches.push_back(matches.at(i));
        }
    }

#ifdef DRAW_DETECTOR_RESULT
    cv::Mat matchImage;
    cv::drawMatches(srcImage1, keypoints1, srcImage2, keypoints2, good_matches, matchImage);
    cv::resize(matchImage, matchImage, cv::Size(matchImage.cols * 0.3, matchImage.rows * 0.3), 0, 0, cv::INTER_LINEAR);

    cv::namedWindow("ORB_Correspondences", cv::WINDOW_AUTOSIZE);
    cv::imshow("ORB_Correspondences", matchImage);
#endif



    for (int i = 0; i < m_numPoint; i++)
    {
        printf("ORBDetector >> Good Match [%d] Keypoint 1: %d = = = Keypoint 2: %d, DIS = %f.\n", i, matches[i].queryIdx, matches[i].trainIdx, matches[i].distance);
    }
    return std::make_pair(image1Points, image2Points);
}