#include "FeatureDectector.h"

FeatureDectector::FeatureDectector(int numPoint) {
    m_numPoint = numPoint;
}

std::pair<KeyPoints,KeyPoints> FeatureDectector::findCorrespondences(ImagePair imgPair, FeatureDectectorMethod fm) {
    cv::Mat img1 = imgPair.img1;
    cv::Mat img2 = imgPair.img2;

    if (img1.empty() || img2.empty()) {
        throw std::length_error("FeatureDectector >> Fail to Load the Image.");
    }
    else {
        std::cerr << "FeatureDectector >> Images successfully loaded." << std::endl;
    }

    if (fm == USE_SIFT) {
        SIFTDetector = cv::SIFT::create();
        std::cout << "FeatureDectector >> SIFTDectector loaded." << std::endl;
    } else if (fm == USE_ORB) {
        ORBDetector = cv::ORB::create();
        std::cout << "FeatureDectector >> ORBDectector loaded." << std::endl;
    } else {
        SURFDetector = cv::xfeatures2d::SURF::create(SURF_MIN_HESSIAN);
        std::cout << "FeatureDectector >> SURFDectector loaded." << std::endl;
    }

    std::vector<cv::KeyPoint> k_points1, k_points2;
    cv::Mat descr1, descr2;

    if (fm == USE_SIFT) {
        SIFTDetector.get()->detectAndCompute(img1, cv::noArray(), k_points1, descr1);
        SIFTDetector.get()->detectAndCompute(img2, cv::noArray(), k_points2, descr2);
    } else if (fm == USE_ORB) {
        ORBDetector.get()->detectAndCompute(img1, cv::noArray(), k_points1, descr1);
        ORBDetector.get()->detectAndCompute(img2, cv::noArray(), k_points2, descr2);
        if (descr1.empty())
            cv::error(0, "MatchFinder", "1st descriptor empty", __FILE__, __LINE__);
        if (descr2.empty())
            cv::error(0, "MatchFinder", "2nd descriptor empty", __FILE__, __LINE__);

        descr1.convertTo(descr1, CV_32F);
        descr2.convertTo(descr2, CV_32F);
    } else {
        SURFDetector.get()->detectAndCompute(img1, cv::noArray(), k_points1, descr1);
        SURFDetector.get()->detectAndCompute(img2, cv::noArray(), k_points2, descr2);
    }

    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch>good_matches;
    cv::FlannBasedMatcher matcher;

    matcher.match(descr1, descr2, matches);
    std::sort(matches.begin(), matches.end(), distanceSorting);

    KeyPoints image1Points;
    KeyPoints image2Points;

    if (matches.size() < m_numPoint) {
        std::cerr << "SIFTDetector >> Error didn't get enough feature points." << std::endl;
    } else {
        for (int i = 0; i < m_numPoint; i++) {
            cv::KeyPoint keyPoint1 = k_points1.at(matches[i].queryIdx);
            cv::Point2f point1 = cv::Point2f(keyPoint1.pt.x, keyPoint1.pt.y);

            cv::KeyPoint keyPoint2 = k_points2.at(matches[i].trainIdx);
            cv::Point2f point2 = cv::Point2f(keyPoint2.pt.x, keyPoint2.pt.y);

            image1Points.push_back(point1);
            image2Points.push_back(point2);
            good_matches.push_back(matches.at(i));
        }
    }

#ifdef DRAW_DETECTOR_RESULT
    cv::Mat m_img;
    cv::drawMatches(img1, k_points1, img2,
                    k_points2, matches, m_img,
                    cv::Scalar::all(-1), cv::Scalar(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::resize(m_img, m_img, cv::Size(m_img.cols * 0.3, m_img.rows * 0.3), 0, 0, cv::INTER_LINEAR);

    // cv::namedWindow("SIFT_Correspondences", cv::WINDOW_AUTOSIZE);
    cv::imshow("Correspondences" + (fm == USE_SIFT) ? "_SIFT" : ((fm == USE_ORB) ? "_ORB" : "_SURF"), m_img);
#endif
    for (int i = 0; i < m_numPoint; i++)
    {
        printf("FeatureDectector >> Good Match [%d] Keypoint 1: %d = = = Keypoint 2: %d , DIS = %f.\n", i, matches[i].queryIdx, matches[i].trainIdx, matches[i].distance);
    }
    return std::make_pair(image1Points, image2Points);
}
