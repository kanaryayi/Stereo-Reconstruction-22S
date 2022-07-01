#include "SURF.h"

SURFDetector::SURFDetector(int minHessian, bool filtering) {
   detector = cv::xfeatures2d::SURF::create(minHessian);

   m_filtering = filtering;
}

std::vector<cv::DMatch> SURFDetector::findCorrespondences(std::string srcImage1Path, std::string srcImage2Path) {
    cv::Mat srcImage1 = cv::imread(srcImage1Path);
    int image1W = srcImage1.cols;
    int image1H = srcImage1.rows;

    cv::Mat srcImage2 = cv::imread(srcImage2Path);
    int image2W = srcImage2.cols;
    int image2H = srcImage2.rows;

    if (srcImage1.empty() || srcImage2.empty()) {
        throw std::length_error("SURFDectector >> Fail to Load the Image");
    }
    else {
        std::cerr << "SURFDectector >> Images successfully loaded" << std::endl;
        cv::resize(srcImage1, srcImage1, cv::Size(image1W * 0.4, image1H * 0.4), 0, 0, cv::INTER_LINEAR);
        cv::resize(srcImage2, srcImage2, cv::Size(image2W * 0.4, image2H * 0.4), 0, 0, cv::INTER_LINEAR);
    }

    std::vector<cv::KeyPoint>keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    detector.get()->detectAndCompute(srcImage1, cv::noArray(), keypoints1, descriptors1);
    detector.get()->detectAndCompute(srcImage2, cv::noArray(), keypoints2, descriptors2);

    std::vector<cv::DMatch>matches;
    cv::FlannBasedMatcher matcher;
    matcher.match(descriptors1, descriptors2, matches);

    if (m_filtering) {
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
            if (matches[i].distance <= std::max(2 * min_dist, 0.02))
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

    cv::namedWindow("SURF_Correspondences", cv::WINDOW_AUTOSIZE);
    cv::imshow("SURF_Correspondences", matchImage);

    for (int i = 0; i < (int)matches.size(); i++)
    {
        printf("SURFDectector>> Good Match [%d] Keypoint 1: %d = = = Keypoint 2: %d \n", i, matches[i].queryIdx, matches[i].trainIdx);
    }
    
    return matches;
}