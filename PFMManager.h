#ifndef PFM_MANAGER_H
#define PFM_MANAGER_H

#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/core/mat.hpp>

class PFMManager {
    public:
        static cv::Mat loadPFM(const std::string filePath);
};

#endif