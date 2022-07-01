#include "ORB.h"

ORBDetector::ORBDetector(float filteringFactor) {
   detector = cv::ORB::create();
    
   m_filteringFactor = filteringFactor;
}

std::vector<cv::DMatch> ORBDetector::findCorrespondences(cv::Mat srcImage1, cv::Mat srcImage2) {
    if (srcImage1.empty() || srcImage2.empty()) {
        throw std::out_of_range("ORBDetector >> Fail to Load the Image");
    }
    else {
        std::cerr << "ORBDetector >> Images successfully loaded" << std::endl;
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
    cv::FlannBasedMatcher matcher;
    std::cout << "OKOK" << std::endl;
    matcher.match(descriptors1, descriptors2, matches);

    if (m_filteringFactor > 0) {
        double max_dist = 0;
        double min_dist = 100;
        for (int i = 0; i < descriptors1.rows; i++)
        {
            double dist = matches[i].distance;
            if (dist < min_dist)min_dist = dist;
            if (dist > max_dist)max_dist = dist;
        }

        std::vector<cv::DMatch>good_matches;
        for (int i = 0; i < descriptors1.rows; i++)
        {
            if (matches[i].distance <= std::max(m_filteringFactor * min_dist, 0.02))
            {
                good_matches.push_back(matches[i]);
            }
        }

        matches = good_matches;
    }


    //std::cout << "Max dist: " << max_dist << std::endl;
    //std::cout << "Min dist: " << min_dist << std::endl;



    cv::Mat matchImage;
    cv::drawMatches(srcImage1, keypoints1, srcImage2, keypoints2, matches, matchImage, cv::Scalar::all(-1), cv::Scalar(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::resize(matchImage, matchImage, cv::Size(matchImage.cols * 0.3, matchImage.rows * 0.3), 0, 0, cv::INTER_LINEAR);

    cv::namedWindow("ORB_Correspondences", cv::WINDOW_AUTOSIZE);
    cv::imshow("ORB_Correspondences", matchImage);
    for (int i = 0; i < (int)matches.size(); i++)
    {
        printf("ORBDetector>> Good Match [%d] Keypoint 1: %d = = = Keypoint 2: %d \n", i, matches[i].queryIdx, matches[i].trainIdx);
    }
    
    return matches;
}