#include "SIFT.h"

SIFTDetector::SIFTDetector(int numPoint) {
    detector = cv::SIFT::create();

    m_numPoint = numPoint;
}

std::pair<KeyPoints,KeyPoints> SIFTDetector::findCorrespondences(cv::Mat img1, cv::Mat img2) {
    if (img1.empty() || img2.empty()) {
        throw std::length_error("SIFTFDectector >> Fail to Load the Image");
    }
    else {
        std::cerr << "SURFDectector >> Images successfully loaded" << std::endl;
    }

    std::vector<cv::KeyPoint> k_points1, k_points2;
    cv::Mat descr1, descr2;

    detector.get()->detectAndCompute(img1, cv::noArray(), k_points1, descr1);
    detector.get()->detectAndCompute(img2, cv::noArray(), k_points2, descr2);

    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch>good_matches;
    cv::FlannBasedMatcher matcher;

    matcher.match(descr1, descr2, matches);
    std::sort(matches.begin(), matches.end(), distanceSorting);
    
    KeyPoints image1Points;
    KeyPoints image2Points;


    if (matches.size() >= m_numPoint) {
        std::cerr << "SIFTDetector >> Error didn't get enough feature points." << std::endl;
    } else {
        for (int i = 0; i < m_numPoint; i++) {
            image1Points.push_back(k_points1.at(matches[i].queryIdx));
            image2Points.push_back(k_points2.at(matches[i].trainIdx));
            good_matches.push_back(matches.at(i));
        }
    }
    // if (m_filteringFactor > 0) {
    //     double max_dist = 0;
    //     double min_dist = 100;
    //     for (int i = 0; i < descr1.rows; i++)
    //     {
    //         double dist = matches[i].distance;
    //         if (dist < min_dist)min_dist = dist;
    //         if (dist > max_dist)max_dist = dist;
    //     }

    //     std::vector<cv::DMatch>good_matches;
    //     for (int i = 0; i < descr1.rows; i++)
    //     {
    //         if (matches[i].distance <= std::max(m_filteringFactor * min_dist, 0.02))
    //         {
    //             good_matches.push_back(matches[i]);
    //         }
    //     }

    //     matches = good_matches;
    // }
#ifdef DRAW_DETECTOR_RESULT
    cv::Mat m_img;
    cv::drawMatches(img1, k_points1, img2,
                    k_points2, matches, m_img,
                    cv::Scalar::all(-1), cv::Scalar(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::resize(m_img, m_img, cv::Size(m_img.cols * 0.3, m_img.rows * 0.3), 0, 0, cv::INTER_LINEAR);

    cv::namedWindow("SIFT_Correspondences", cv::WINDOW_AUTOSIZE);
    cv::imshow("SIFT_Correspondences", m_img);
#endif
    for (int i = 0; i < m_numPoint; i++)
    {
        printf("SIFTDectector>> Good Match [%d] Keypoint 1: %d = = = Keypoint 2: %d , DIS = %f\n", i, matches[i].queryIdx, matches[i].trainIdx, matches[i].distance);
    }
    return std::make_pair(image1Points, image2Points);
}