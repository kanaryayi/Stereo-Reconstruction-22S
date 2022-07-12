#include "Utils.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>

class BlockMatcher {
public:
    BlockMatcher(ImagePair imgPair);
    void opBM(int wSize, int maxDisp);
    void opBMwithFiltering(int wSize, int maxDisp);
    void opSGBM(int wSize, int maxDisp);
    void opSGBMwithFiltering(int wSize, int maxDisp);
private:

    ImagePair m_imgPair;
};