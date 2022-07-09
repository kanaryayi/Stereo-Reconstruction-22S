#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

#include <filesystem>
#include <iostream>
#include <fstream>
#include <random>

//#define DRAW_DETECTOR_RESULT

#define USE_MIDDLEBURY_2014
#define BIG_INTEGER 114514

typedef std::vector<cv::KeyPoint> KeyPoints;

inline bool distanceSorting(cv::DMatch a, cv::DMatch b) {
    return a.distance < b.distance;
}

struct MiddleburyImagePair {
    cv::Mat img1;
    cv::Mat img2;

    cv::Mat K_img1;
    cv::Mat K_img2;

    float baseline;
    float doffs;

    int weight;
    int height;

    int vmin;
    int vmax;
};

class DataLoader
{
    public:
        DataLoader(std::string dataset) {
            m_dataset = dataset;
            getFiles("..\\data\\" + m_dataset, m_files, BIG_INTEGER);
            initImagePairs();
            std::cout << "DataLoader >> All " << getSizeOfDataset() << " scenes loaded successfully." << std::endl;
        }

        DataLoader(std::string dataset, int num) {
            m_dataset = dataset;
            getFiles("..\\data\\" + m_dataset, m_files, num);
            initImagePairs();
            std::cout << "DataLoader >> All " << getSizeOfDataset() << " scenes loaded successfully." << std::endl;
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

        MiddleburyImagePair getImagePairByIndex(int index) {
            std::string path = m_files.at(index);

            std::string img1Path = path + "/im0.png";
            std::string img2Path = path + "/im1.png";
            std::string calibPath = path + "/calib.txt";
            
            MiddleburyImagePair imgPair;

            imgPair.img1 = cv::imread(img1Path);
            imgPair.img2 = cv::imread(img2Path);

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

                //std::cout << getAttrNumByName("cam1", lines) << std::endl;
                imgPair.K_img2 = getIntrinsicOfString(getAttrNumByName("cam1", lines));

                //std::cout << imgPair.K_img1 << std::endl;
                //std::cout << imgPair.K_img2 << std::endl;
                ifs.close();
            }
            return imgPair;
        }

        std::vector<MiddleburyImagePair> getAllImagePairs() {
            return m_imagePairs;
        }

        MiddleburyImagePair getRandomSample() {
            std::vector<MiddleburyImagePair> oneShot;
            std::sample(m_imagePairs.begin(), m_imagePairs.end(), std::back_inserter(oneShot),
                1, std::mt19937{ std::random_device{}() });
            return oneShot.at(0);
        }

        cv::Mat getIntrinsicOfString(std::string intrinsicString) {
            auto result = std::vector<std::string>{};
            auto ss = std::stringstream{ intrinsicString };

            for (std::string line; std::getline(ss, line, ';');)
                result.push_back(line);

            auto row = std::vector<float>{};

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

            cv::Mat retMat = (cv::Mat_<float>(3, 3) << row[0], row[1], row[2],
                row[3], row[4], row[5],
                row[6], row[7], row[8]);
            return retMat;
        }

    private:
        std::string m_dataset = "Middlebury_2014";
        std::vector<MiddleburyImagePair> m_imagePairs;
        std::vector<std::string> m_files;

        void getFiles(std::string path, std::vector<std::string>& m_files, int num)
        {
            for (const auto& entry : std::filesystem::directory_iterator(path)) {
                //std::cout << entry.path().string() << std::endl;
                m_files.push_back(entry.path().string());
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
                    std::cout << m_files.at(outIndices.at(i)) << std::endl;
                    tmpFiles.push_back(m_files.at(outIndices.at(i)));
                }

                m_files.clear();
                m_files.swap(tmpFiles);
            }
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