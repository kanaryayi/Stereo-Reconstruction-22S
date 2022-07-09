#include "SURF.h"

SURFDetector::SURFDetector(int minHessian, int numPoint) {
   detector = cv::xfeatures2d::SURF::create(minHessian);

   m_numPoint = numPoint;
}

std::pair<KeyPoints,KeyPoints> SURFDetector::findCorrespondences(cv::Mat srcImage1, cv::Mat srcImage2) {
    if (srcImage1.empty() || srcImage2.empty()) {
        throw std::length_error("SURFDectector >> Fail to Load the Image");
    }
    else {
        std::cerr << "SURFDectector >> Images successfully loaded" << std::endl;
    }

    std::vector<cv::KeyPoint>keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    detector.get()->detectAndCompute(srcImage1, cv::noArray(), keypoints1, descriptors1);
    detector.get()->detectAndCompute(srcImage2, cv::noArray(), keypoints2, descriptors2);

    std::vector<cv::DMatch>matches;
    std::vector<cv::DMatch>good_matches;
    cv::FlannBasedMatcher matcher;
    matcher.match(descriptors1, descriptors2, matches);

    std::sort(matches.begin(), matches.end(), distanceSorting);
    
    KeyPoints image1Points;
    KeyPoints image2Points;

    if (matches.size() >= m_numPoint) {
        std::cerr << "SURFDetector >> Error didn't get enough feature points." << std::endl;
    } else {
        for (int i = 0; i < m_numPoint; i++) {
            image1Points.push_back(keypoints1.at(matches[i].queryIdx));
            image2Points.push_back(keypoints2.at(matches[i].trainIdx));
            good_matches.push_back(matches.at(i));
        }
    }

#ifdef DRAW_DETECTOR_RESULT
    cv::Mat matchImage;
    //cv::drawKeypoints(srcImage1, image1Points, srcImage1,cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    //cv::drawKeypoints(srcImage2, image2Points, srcImage2,cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    cv::drawMatches(srcImage1, image1Points, srcImage2, image2Points, good_matches, matchImage, 5);
    cv::resize(matchImage, matchImage, cv::Size(matchImage.cols * 0.3, matchImage.rows * 0.3), 0, 0, cv::INTER_LINEAR);

    cv::namedWindow("SURF_Correspondences", cv::WINDOW_AUTOSIZE);
    cv::imshow("SURF_Correspondences", matchImage);
#endif



    for (int i = 0; i < m_numPoint; i++)
    {
        printf("SURFDectector>> Good Match [%d] Keypoint 1: %d = = = Keypoint 2: %d , DIS = %f\n", i, matches[i].queryIdx, matches[i].trainIdx, matches[i].distance);
    }
    
    return std::make_pair(image1Points, image2Points);
}