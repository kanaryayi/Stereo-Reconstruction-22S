#include "sift.h"

SIFTDetector::SIFTDetector(float filtering_factor) {
    detector = cv::SIFT::create();

    m_filteringFactor = filtering_factor;
}

std::vector<cv::DMatch> SIFTDetector::findCorrespondences(cv::Mat img1, cv::Mat img2) {

    std::vector<cv::KeyPoint> k_points1, k_points2;
    cv::Mat descr1, descr2;

    detector.get()->detectAndCompute(img1, cv::noArray(), k_points1, descr1);
    detector.get()->detectAndCompute(img2, cv::noArray(), k_points2, descr2);

    std::vector<cv::DMatch> matches;
    cv::FlannBasedMatcher matcher;

    matcher.match(descr1, descr2, matches);

    if (m_filteringFactor > 0) {
        double max_dist = 0;
        double min_dist = 100;
        for (int i = 0; i < descr1.rows; i++)
        {
            double dist = matches[i].distance;
            if (dist < min_dist)min_dist = dist;
            if (dist > max_dist)max_dist = dist;
        }

        std::vector<cv::DMatch>good_matches;
        for (int i = 0; i < descr1.rows; i++)
        {
            if (matches[i].distance <= std::max(m_filteringFactor * min_dist, 0.02))
            {
                good_matches.push_back(matches[i]);
            }
        }

        matches = good_matches;
    }

    cv::Mat m_img;
    cv::drawMatches(img1, k_points1, img2,
                    k_points2, matches, m_img,
                    cv::Scalar::all(-1), cv::Scalar(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::resize(m_img, m_img, cv::Size(m_img.cols * 0.3, m_img.rows * 0.3), 0, 0, cv::INTER_LINEAR);

    cv::namedWindow("SIFT Correspondences", cv::WINDOW_AUTOSIZE);
    cv::imshow("SIFT Correspondences", m_img);
    
    return matches;
}