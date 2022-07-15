#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>


void drawMatches(
    cv::Mat img1,
    cv::Mat img2,
    std::vector<cv::KeyPoint> key_1,
    std::vector<cv::KeyPoint> key_2,
    std::vector<cv::DMatch> matches   
) {
    cv::Mat img_matches;
    cv::drawMatches(img1, key_1, img2, key_2, matches, img_matches,
        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Good Matches", img_matches);
    cv::waitKey(0);

}

float ssd(cv::Rect& roi, cv::Mat& img1, cv::Mat& img2) {

    cv::Mat region_1{img1, roi};
    cv::Mat region_2{img2, roi};

    return cv::sum(cv::abs(region_1 - region_2))[0];
}

void calculate_disperity_map(cv::Mat img_left, cv::Mat img_right, int window_size, int search_size) {

    int h = img_left.rows;
    int w = img_left.cols;

    int half_window = std::floor(window_size / 2.0);

    cv::Mat disparity

    for (int y = half_window; y < h - half_window; y++) {
        for (int x = half_window; x < w - half_window; x++) {

            int x_start = x - half_window;
            int width = x + half_window + 1 - x_start;

            int y_start = y - half_window;
            int height = y + half_window + 1 - y_start;

            cv::Rect block{x_start, y_start, width, height};

            float ssd_val = ssd(block, img_left, img_right);
        }
    }

}

int main(int argc, char** argv) {

    srand(time(NULL));

    cv::Mat img1, img2;

    img1 = cv::imread("../data/im0_res.png");
    img2 = cv::imread("../data/im1_res.png");


    // Convert to grayscale
    cv::Mat img1_g, img2_g;

    cv::cvtColor(img1, img1_g, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_g, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> key_1, key_2;
    cv::Mat descr_1, descr_2;
    
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->detectAndCompute(img1_g, cv::noArray(), key_1, descr_1);
    sift->detectAndCompute(img2_g, cv::noArray(), key_2, descr_2);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descr_1, descr_2, knn_matches, 2);
    
    const float ratio_thresh = 0.7f;

    std::vector<cv::DMatch> good_matches;


    std::vector<cv::Point2f> best_points_1;
    std::vector<cv::Point2f> best_points_2;

    // Extract best matches
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {            
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    for (auto& match : good_matches) {
        best_points_1.push_back(key_1[match.queryIdx].pt);
        best_points_2.push_back(key_2[match.trainIdx].pt);
    }
    
    /*
    cv::circle(img1, best_points_1[0], 10, cv::Scalar(0, 255,0), -1);
    cv::namedWindow("Keypoint 1", cv::WINDOW_AUTOSIZE);
    cv::imshow("Keypoint 1", img1);
    


    cv::circle(img2, best_points_2[0], 10, cv::Scalar(0, 255,0), -1);
    cv::namedWindow("Keypoint 2", cv::WINDOW_AUTOSIZE);
    cv::imshow("Keypoint 2", img2);

    cv::waitKey(0);
    */

    cv::InputArray points1_arr{best_points_1};
    cv::InputArray points2_arr{best_points_2};

    std::vector<uchar> mask;
    
    cv::Mat fundamental_matrix = cv::findFundamentalMat(points1_arr, points2_arr,
        cv::FM_RANSAC, 3, 0.99, mask);
    
    
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i] != 0) {
            points1.push_back(best_points_1[i]);
            points2.push_back(best_points_2[i]);
        }
    }

    best_points_1.clear();
    best_points_2.clear();

    /*
    for (int i = 0; i < points1.size(); i++) {
        cv::circle(img1_g, points1[i], 5, cv::Scalar(0,255,0), -1);
    }

    cv::imshow("Points", img1_g);
    cv::waitKey(0);
    */

    cv::InputArray p1_arr{points1};
    cv::InputArray p2_arr{points2};


    cv::Mat H1, H2;

    cv::stereoRectifyUncalibrated(p1_arr, p2_arr, fundamental_matrix, img1_g.size(), H1, H2);

    cv::Mat img1_rectified{};
    cv::Mat img2_rectified{};

    cv::warpPerspective(img1, img1_rectified, H1, img1.size());
    cv::warpPerspective(img2, img2_rectified, H2, img2.size());

    cv::imwrite("../data/rectified1.png", img1_rectified);
    cv::imwrite("../data/rectified2.png", img2_rectified);
    /*
    std::vector<cv::Vec3f> left_lines{};
    cv::computeCorrespondEpilines(p1_arr, 1, fundamental_matrix, left_lines);


    for (size_t i = 0; i < std::max<int>(left_lines.size(), 20); i++) {

        int idx = rand() % left_lines.size();

        cv::Scalar color{rand() % 255, rand() % 255, rand() % 255};

        cv::Vec3f line = left_lines[idx];
        cv::line(img2, cv::Point(0, -(line)[2] / (line)[1]),
                 cv::Point(img2.cols, -((line)[2] + (line)[0] * img2.cols) / (line)[1]),
                 color);
    }

    cv::imshow("Image 2 - Epipolar", img2);
    int s = cv::waitKey(0);

    if (s == 's') {
        cv::imwrite("../data/epipolar_left.png", img2);
    }
    */

    return 0;
}
