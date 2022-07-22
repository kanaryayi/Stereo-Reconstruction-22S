#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <iostream>
#include <fstream>
#include <random>
#include <filesystem>

#include "PFMManager.h"

#define SPARSE_MATCHING 1
#define DENSE_MATCHING  0

#define DRAW_DETECTOR_RESULT

//#define ALL_SAMPLE
#define SPEC_SAMPLE
//#define RANDOM_SAMPLE

// minHessian Setting https://stackoverflow.com/a/17615172
#define SURF_MIN_HESSIAN 600

#define USE_MIDDLEBURY_2014
#define BIG_INTEGER 114514

typedef std::vector<cv::Point2f> KeyPoints;
typedef cv::Mat Rotate;
typedef cv::Mat Translate;

inline bool distanceSorting(cv::DMatch a, cv::DMatch b) {
    return a.distance < b.distance;
}

inline cv::Mat makeSkewMatrixFromPoint(cv::Point3f p) {
    cv::Mat skewMatrix = (cv::Mat_<double>(3, 3) <<    0,  -1,  p.y, 
                                                       1,   0, -p.x,
                                                    -p.y, p.x,   0);

    return skewMatrix;
}

inline cv::Mat getEulerAngleByRotationMatrix(cv::Mat Rotate) {
    double R32 = Rotate.at<double>(2, 1);
    double R33 = Rotate.at<double>(2, 2);
    double thetaX = std::atan2(R32, R33);
    double thetaY = std::atan2(-Rotate.at<double>(2, 0), std::sqrt(R32 * R32 + R33 * R33));
    double thetaZ = std::atan2(Rotate.at<double>(1, 0), Rotate.at<double>(0, 0));

    cv::Mat retM = (cv::Mat_<double>(3, 1) << thetaX, thetaY, thetaZ);

    return retM;
}

inline cv::Mat getRoationMatrixByEulerAngle(cv::Mat angle) {
    double thetaX = angle.at<double>(0, 0);
    double thetaY = angle.at<double>(1, 0);
    double thetaZ = angle.at<double>(2, 0);

    cv::Mat X = (cv::Mat_<double>(3, 3) << 1, 0, 0,
        0, std::cos(thetaX), -std::sin(thetaX),
        0, std::sin(thetaX), std::cos(thetaX));

    cv::Mat Y = (cv::Mat_<double>(3, 3) << std::cos(thetaY), 0, std::sin(thetaY),
        0, 1, 0,
        -std::sin(thetaY), 0, std::cos(thetaY));

    cv::Mat Z = (cv::Mat_<double>(3, 3) << std::cos(thetaZ), -std::sin(thetaZ), 0,
        std::sin(thetaZ), std::cos(thetaZ), 0,
        0, 0, 1);

    return Z * Y * X;
}

struct ImagePair {
    std::string path;

    cv::Mat img1;
    cv::Mat img2;

    cv::Mat K_img1;
    cv::Mat K_img2;

    /*
        Ground truth disperity of left image.
    */
    cv::Mat disp0;

    /*
        Ground truth disperity of right image.
    */
    cv::Mat disp1;

    float f1;
    float f2;

    float baseline;
    float doffs;

    int weight;
    int height;

    int vmin;
    int vmax;

    // Returns a new ImagePair with the same information/matrices as the
    // original one with the difference that it is downsampled by `factor`
    ImagePair sampleDown(float factor) {
        cv::Mat img1_d;
        cv::Mat img2_d;
        cv::Mat disp0_d;
        cv::Mat disp1_d;
        
        cv::resize(img1, img1_d, cv::Size(factor * img1.cols, factor * img1.rows),
            0, 0, cv::INTER_LINEAR);
        cv::resize(img1, img1_d, cv::Size(factor * img1.cols, factor * img1.rows),
            0, 0, cv::INTER_LINEAR);
        cv::resize(disp0, disp0_d, cv::Size(factor * disp0.cols, factor * disp0.rows),
            0, 0, cv::INTER_LINEAR);
        cv::resize(disp1, disp1_d, cv::Size(factor * disp1.cols, factor * disp1.rows),
            0, 0, cv::INTER_LINEAR);

        ImagePair new_pair{};

        new_pair.img1 = img1_d;
        new_pair.img2 = img2_d;
        new_pair.K_img1 = K_img1;
        new_pair.K_img2 = K_img2;
        new_pair.disp0 = disp0_d;
        new_pair.disp1 = disp1_d;
        new_pair.f1 = f1;
        new_pair.f2 = f2;
        new_pair.baseline = baseline;
        new_pair.doffs = doffs;
        new_pair.weight = weight;
        new_pair.height = height;
        new_pair.vmin = vmin;
        new_pair.vmax = vmax;
        
        return new_pair;
    }
};

class DataLoader
{
    public:
        DataLoader(std::string dataset) {
            std::cout << "DataLoader >> Loading data ..." << std::endl;
            m_dataset = dataset;
            std::filesystem::path rootDataPath = "../data/" + m_dataset;
            getFiles(rootDataPath, m_files, BIG_INTEGER);
            initImagePairs();
            std::cout << "DataLoader >> All " << getSizeOfDataset() << " scenes loaded successfully." << std::endl;
        }

        // Loads `num` many data pairs, specified by the `dataset` string. If
        // `load_specific` is true then it loads a specific pair, otherwise
        // it will randomly sample a subset of all available sets. For `load_specific`
        // to work `num = 1` must be true.
        DataLoader(std::string dataset, int num, bool load_specific = false) {
            std::cout << "DataLoader >> Loading data ..." << std::endl;
            m_dataset = dataset;
            std::filesystem::path rootDataPath = "../data/" + m_dataset;
            
            if (load_specific) {
                getSpecificFiles(rootDataPath, m_files, num);
            } else {
                getFiles(rootDataPath, m_files, num);
            }

            if (num > 0) {
                initImagePairs();
                std::cout << "DataLoader >> All " << getSizeOfDataset() << " scenes loaded successfully." << std::endl;
            }
        }
        
        void initImagePairs() {
            for (int i = 0; i < m_files.size(); i++) {
                m_imagePairs.push_back(getImagePairByIndex(i));
            }
        }

        int getSizeOfDataset() {
            return m_files.size();
        }

        std::string getDataPathByIndex(int index) {
            return m_files.at(index);
        }

        ImagePair getSpecificSample(std::string name) {
            for (int i = 0; i < m_files.size(); i++) {
                if (m_files.at(i).find(name) != std::string::npos) {
                    return getImagePairByIndex(i);
                }
            }

            return getImagePairByIndex(-1);
        }

        ImagePair getImagePairByIndex(int index) {
            if (index < 0) {
                throw std::invalid_argument("Bad index");
            }

            std::string path = m_files.at(index);

            std::string img1Path = path + "/im0.png";
            std::string img2Path = path + "/im1.png";
            std::string calibPath = path + "/calib.txt";


            std::string disp0Path = path + "/disp0.pfm";
            std::string disp1Path = path + "/disp1.pfm";

            std::cout << "PATH " << path << "\n";
            
            ImagePair imgPair;

            imgPair.path = path;

            imgPair.img1 = cv::imread(img1Path);
            imgPair.img2 = cv::imread(img2Path);

            // Calculate disperity for left image
            
            cv::Mat disp0_raw = PFMManager::loadPFM(disp0Path);
            cv::Mat disp0_mask{disp0_raw == std::numeric_limits<float>::infinity()};
            disp0_raw.setTo(0, disp0_mask);
            
            cv::Mat disp0_norm;
            cv::normalize(disp0_raw, disp0_norm, 0, 255, cv::NORM_MINMAX);
            disp0_norm.convertTo(imgPair.disp0, CV_8UC1);
            
            cv::Mat disp1_raw = PFMManager::loadPFM(disp1Path);
            cv::Mat disp1_mask{disp1_raw == std::numeric_limits<float>::infinity()};
            disp1_raw.setTo(0, disp1_mask);
            
            cv::Mat disp1_norm;
            cv::normalize(disp1_raw, disp1_norm, 0, 255, cv::NORM_MINMAX);
            disp1_norm.convertTo(imgPair.disp1, CV_8UC1);

            std::ifstream ifs(calibPath, std::ios::in);

            if (!ifs.is_open()) {
                std::cout << "DataLoader >> Failed to open file.\n";
            } else {
                std::stringstream ss;
                ss << ifs.rdbuf();
                std::string str(ss.str());
                std::vector<std::string> lines = splitStringByNewline(str);

                imgPair.baseline = std::stof(getAttrNumByName("baseline", lines));
                imgPair.doffs = std::stof(getAttrNumByName("doffs", lines));
                imgPair.weight = std::stoi(getAttrNumByName("width", lines));
                imgPair.height = std::stoi(getAttrNumByName("height", lines));
                imgPair.vmin = std::stoi(getAttrNumByName("vmin", lines));
                imgPair.vmax = std::stoi(getAttrNumByName("vmax", lines));

                //std::cout << getAttrNumByName("cam0", lines) << std::endl;
                imgPair.K_img1 = getIntrinsicOfString(getAttrNumByName("cam0", lines));
                imgPair.f1 = imgPair.K_img1.at<double>(0, 0);
                //std::cout << getAttrNumByName("cam1", lines) << std::endl;
                imgPair.K_img2 = getIntrinsicOfString(getAttrNumByName("cam1", lines));
                imgPair.f2 = imgPair.K_img2.at<double>(0, 0);

                //std::cout << imgPair.K_img1 << std::endl;
                //std::cout << imgPair.K_img2 << std::endl;
                ifs.close();
            }
            return imgPair;
        }

        std::vector<ImagePair> getAllImagePairs() {
            return m_imagePairs;
        }

        ImagePair getRandomSample() {
            std::vector<ImagePair> oneShot;
            std::sample(m_imagePairs.begin(), m_imagePairs.end(), std::back_inserter(oneShot),
                1, std::mt19937{ std::random_device{}() });
            return oneShot.at(0);
        }

        cv::Mat getIntrinsicOfString(std::string intrinsicString) {
            auto result = std::vector<std::string>{};
            auto ss = std::stringstream{ intrinsicString };

            for (std::string line; std::getline(ss, line, ';');)
                result.push_back(line);

            auto row = std::vector<double>{};

            for (std::string line : result) {
                auto ss_sub = std::stringstream{ line };
                for (std::string line_sub; std::getline(ss_sub, line_sub, ' ');) {
                    line_sub.erase(std::remove(line_sub.begin(), line_sub.end(), ' '), line_sub.end());
                    if (line_sub.length() > 0) {
                        //std::cout << line_sub.length() << " Line->" << line_sub << "<-End" << std::endl;
                        row.push_back(std::stof(line_sub));
                    }
                }
            }
            
            cv::Mat retMat = (cv::Mat_<double>(3, 3) << row[0], row[1], row[2],
                                                        row[3], row[4], row[5],
                                                        row[6], row[7], row[8]);
            return retMat;
        }

    private:
        std::string m_dataset = "Middlebury_2014";
        std::vector<ImagePair> m_imagePairs;
        std::vector<std::string> m_files;

        void getFiles(std::filesystem::path path, std::vector<std::string>& m_files, int num)
        {
            for (const auto& entry : std::filesystem::directory_iterator(path)) {
                //std::cout << entry.path().string() << std::endl;
                m_files.push_back(entry.path().string());
            }
            if (num < 0) {
                return;
            }

            if (num < m_files.size()) {
                std::vector<int> tmpIndcies, outIndices;
                for (int i = 0; i < m_files.size(); i++) {
                    tmpIndcies.push_back(i);
                }

                std::sample(tmpIndcies.begin(), tmpIndcies.end(), std::back_inserter(outIndices),
                    num, std::mt19937{ std::random_device{}() });

                std::vector<std::string> tmpFiles;
                for (int i = 0; i < outIndices.size(); i++) {
                    std::cout << "DataLoader >> " << m_files.at(outIndices.at(i)) << " is loaded." << std::endl;
                    tmpFiles.push_back(m_files.at(outIndices.at(i)));
                }

                m_files.clear();
                m_files.swap(tmpFiles);
            }

            else {
                for (int i = 0; i < m_files.size(); i++) {
                    std::cout << "DataLoader >> " << m_files.at(i) << " is loaded." << std::endl;
                }
            }
        }

        void getSpecificFiles(std::filesystem::path path, std::vector<std::string>& m_files, int num) {
            m_files.push_back(path.string());
        }

        std::string getAttrNumByName(std::string attr, std::vector<std::string> lines) {
            int offset = 1;
            int backOffset = 0;
            for (std::string line : lines) {
                if (line.find(attr) == 0) {
                    if (attr.find("cam") == 0) {
                        offset++;
                        backOffset--;
                    }
                    return line.substr(attr.length() + offset, line.length() - attr.length() - offset + backOffset);
                }
            }

            return "ERROR";
        }

        std::vector<std::string> splitStringByNewline(const std::string& str)
        {
            auto result = std::vector<std::string>{};
            auto ss = std::stringstream{ str };

            for (std::string line; std::getline(ss, line, '\n');)
                result.push_back(line);

            return result;
        }
};