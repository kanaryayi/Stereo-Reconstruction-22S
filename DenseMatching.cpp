#include "DenseMatching.h"

cv::Mat ssd(cv::Mat left, cv::Mat right, int window_size) {
    int width = left.size().width;
    int height = right.size().height;
    int max_offset = 79;

    cv::Mat depth(height, width, 0);
    std::vector<std::vector<int> > min_ssd; // store min SSD values

    for (int i = 0; i < height; ++i) {
        std::vector<int> tmp(width, std::numeric_limits<int>::max());
        min_ssd.push_back(tmp);
    }

    for (int offset = 0; offset <= max_offset; offset++) {
        cv::Mat tmp(height, width, 0);
        // shift image depend on type to save calculation time

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < offset; x++) {
                tmp.at<uchar>(y, x) = right.at<uchar>(y, x);
            }

            for (int x = offset; x < width; x++) {
                tmp.at<uchar>(y, x) = right.at<uchar>(y, x - offset);
            }
        }

        // calculate each pixel's SSD value
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int start_x = std::max(0, x - window_size);
                int start_y = std::max(0, y - window_size);
                int end_x = std::min(width - 1, x + window_size);
                int end_y = std::min(height - 1, y + window_size);
                int sum_sd = 0;

                for (int i = start_y; i <= end_y; i++) {
                    for (int j = start_x; j <= end_x; j++) {
                        int delta = abs(left.at<uchar>(i, j) - tmp.at<uchar>(i, j));
                        sum_sd += delta * delta;
                    }
                }

                // smaller SSD value found
                if (sum_sd < min_ssd[y][x]) {
                    min_ssd[y][x] = sum_sd;
                    // for better visualization
                    depth.at<uchar>(y, x) = (uchar)(offset * 3);
                }
            }
        }
    }

    return depth;
}

cv::Mat DenseMatching::execute(const cv::Mat& img1, const cv::Mat& img2, MatchingMethod method, int block_size, int num_disp) {

    // Convert input images to grayscale
    // This makes matching better and doesn't change the matched windows
    cv::Mat img1_gray{};
    cv::Mat img2_gray{};

    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);

    switch (method) {
        case MatchingMethod::OPENCV_BM: {
                cv::Mat disparity;
                cv::Ptr<cv::StereoMatcher> matcher = cv::StereoBM::create(num_disp, block_size);
                matcher->compute(img1_gray, img2_gray, disparity);

                // Divide by 16 because OpenCV disperity map is calculated with 4 bit of fractional precision
                disparity = disparity / 16;
                disparity.convertTo(disparity, CV_8U);
                return disparity;
            }
            break;
        case MatchingMethod::OPENCV_SGBM: {
                cv::Mat disparity;
                cv::Ptr<cv::StereoSGBM> matcher = cv::StereoSGBM::create(0, num_disp, block_size);
                matcher->setP1(24 * block_size * block_size);
                matcher->setP2(96 * block_size * block_size);
                matcher->compute(img1_gray, img2_gray, disparity);


            // Divide by 16 because OpenCV disperity map is calculated with 4 bit of fractional precision
                disparity = disparity / 16;
                disparity.convertTo(disparity, CV_8U);
                return disparity;
            }
        case MatchingMethod::SAD: {
                return ssd(img1_gray, img2_gray, block_size);
            }
        default:
            throw std::invalid_argument("[DenseMatching::execute]: Unsupported matching algorithm") ;
    }

    cv::Mat dummy{};
    return dummy;
}

float DenseMatching::evaluate(cv::Mat ground_truth, cv::Mat computed, EvaluationMetric metric) {

    if (ground_truth.size() != computed.size()) {
        std::cerr << "[DisperityMapEvaluation::evaluate] Mismatch in disperity map size\n";
        return -1.0f;
    }

    int dist = 1;

    switch (metric) {
        case EvaluationMetric::BAD_1: dist = 1; break;
        case EvaluationMetric::BAD_2: dist = 2; break;
        case EvaluationMetric::BAD_5: dist = 5; break;
    }

    switch (metric) {
        case RMS: {
                float abs_error = 0;
                for (int y = 0; y < ground_truth.rows; y++) {
                    for (int x = 0; x < ground_truth.cols; x++) {
                        float diff = computed.at<unsigned char>(y, x) - ground_truth.at<unsigned char>(y,x);
                        abs_error += diff * diff;
                    }
                }

                return std::sqrt(abs_error / (float)(ground_truth.cols * ground_truth.rows));
            }
        case BAD_1:
        case BAD_2:
        case BAD_5: {
                int width = ground_truth.cols;
                int height = ground_truth.rows;
                int bad_pixels = 0;

                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int delta = std::abs(computed.at<unsigned char>(y, x) - ground_truth.at<unsigned char>(y, x));
                        if (delta > dist && computed.at<unsigned char>(y, x) != 0) {
                            bad_pixels++;
                        }
                    }
                }

                float rate_left = float(bad_pixels) / (float)(height * width);
                return rate_left * 100;
            }
        default:
            return 0.0;
    }
}

void DenseMatching::evaluateSGBM(const ImagePair& sample, const std::string& name, int block_size, int num_disp) {
    PointCloud pc{};
    float max_depth = 0.0;

    // Compute disparity map using SGBM
    cv::Mat disp_sgbm = DenseMatching::execute(sample.img1, sample.img2, MatchingMethod::OPENCV_SGBM, block_size, num_disp);

    float bad1_sgbm = DenseMatching::evaluate(sample.disp0, disp_sgbm, EvaluationMetric::BAD_1);
    float bad2_sgbm = DenseMatching::evaluate(sample.disp0, disp_sgbm, EvaluationMetric::BAD_2);
    float bad5_sgbm = DenseMatching::evaluate(sample.disp0, disp_sgbm, EvaluationMetric::BAD_5);
    float rms_sgbm = DenseMatching::evaluate(sample.disp0, disp_sgbm, EvaluationMetric::RMS);
    std::cout << "Metrics - SGBM\n";
    std::cout << "BAD-1 " << 100.0 - bad1_sgbm << "\n";
    std::cout << "BAD-2 " << 100.0 - bad2_sgbm << "\n";
    std::cout << "BAD-5 " << 100.0 - bad5_sgbm << "\n";
    std::cout << "RMS " << rms_sgbm << "\n";

    cv::Mat depth_sgbm = pc.depthMapFromDisperityMap(disp_sgbm, sample.baseline, sample.doffs, sample.f1, &max_depth, true);

    std::stringstream ss;
    ss << name + "_sgbm.off";

    Vertex* vertices = pc.generatePointCloud(depth_sgbm, sample.img1, sample.K_img1, max_depth);

    if (!Mesh::writeMesh(vertices, depth_sgbm.cols, depth_sgbm.rows, ss.str())) {
        std::cerr << "Could not write mesh " << ss.str() << "\n";
    }

    delete[] vertices;

    cv::resize(disp_sgbm, disp_sgbm, cv::Size(0.4 * disp_sgbm.cols, 0.4 * disp_sgbm.rows), 0, 0, cv::INTER_LINEAR);
    cv::resize(depth_sgbm, depth_sgbm, cv::Size(0.4 * disp_sgbm.cols, 0.4 * disp_sgbm.rows), 0, 0, cv::INTER_LINEAR);

    depth_sgbm = 255 * depth_sgbm;
    depth_sgbm.convertTo(depth_sgbm, CV_8UC1);

    cv::imwrite(name + "_disp_normal_sgbm.png", disp_sgbm);
    cv::applyColorMap(disp_sgbm, disp_sgbm, cv::COLORMAP_JET);
    cv::imwrite(name + "_disp_jet_sgbm.png", disp_sgbm);

    cv::imwrite(name + "_depth_sgbm.png", depth_sgbm);
}

void DenseMatching::evaluateBM(const ImagePair& sample, const std::string& name, int block_size, int num_disp) {
    PointCloud pc{};
    float max_depth = 0.0;

    // Compute disparity map using SGBM
    cv::Mat disp_bm = DenseMatching::execute(sample.img1, sample.img2, MatchingMethod::OPENCV_BM, block_size, num_disp);

    float bad1_bm = DenseMatching::evaluate(sample.disp0, disp_bm, EvaluationMetric::BAD_1);
    float bad2_bm = DenseMatching::evaluate(sample.disp0, disp_bm, EvaluationMetric::BAD_2);
    float bad5_bm = DenseMatching::evaluate(sample.disp0, disp_bm, EvaluationMetric::BAD_5);
    float rms_bm = DenseMatching::evaluate(sample.disp0, disp_bm, EvaluationMetric::RMS);
    std::cout << "Metrics - BM\n";
    std::cout << "BAD-1 " << 100.0 - bad1_bm << "\n";
    std::cout << "BAD-2 " << 100.0 - bad2_bm << "\n";
    std::cout << "BAD-5 " << 100.0 - bad5_bm << "\n";
    std::cout << "RMS " << rms_bm << "\n";

    cv::Mat depth_bm = pc.depthMapFromDisperityMap(disp_bm, sample.baseline, sample.doffs, sample.f1, &max_depth, true);

    std::stringstream ss;
    ss << name + "_bm.off";

    Vertex* vertices = pc.generatePointCloud(depth_bm, sample.img1, sample.K_img1, max_depth);

    if (!Mesh::writeMesh(vertices, depth_bm.cols, depth_bm.rows, ss.str())) {
        std::cerr << "Could not write mesh " << ss.str() << "\n";
    }

    delete[] vertices;

    cv::resize(disp_bm, disp_bm, cv::Size(0.4 * disp_bm.cols, 0.4 * disp_bm.rows), 0, 0, cv::INTER_LINEAR);
    cv::resize(depth_bm, depth_bm, cv::Size(0.4 * depth_bm.cols, 0.4 * depth_bm.rows), 0, 0, cv::INTER_LINEAR);

    depth_bm = 255 * depth_bm;
    depth_bm.convertTo(depth_bm, CV_8UC1);

    cv::imwrite(name + "_disp_normal_bm.png", disp_bm);
    cv::applyColorMap(disp_bm, disp_bm, cv::COLORMAP_JET);
    cv::imwrite(name + "_disp_jet_bm.png", disp_bm);

    cv::imwrite(name + "_depth_bm.png", depth_bm);
}