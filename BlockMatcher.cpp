#include "BlockMatcher.h"

BlockMatcher::BlockMatcher(ImagePair imgPair) {
    m_imgPair = imgPair;
}


// https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/samples/disparity_filtering.cpp
// https://docs.opencv.org/3.4/d3/d14/tutorial_ximgproc_disparity_filtering.html
// https://stackoverflow.com/questions/45855725/does-the-stereobm-class-in-opencv-do-rectification-of-the-input-images-or-frames

void BlockMatcher::performBlockMatching(int wSize, int maxDisp, BlockMatcherMethod bm, bool enableWLS) {
    std::cout << "BlockMatcher >> Perform Block Matching window size = " << wSize << ", max disp = " << maxDisp << "." << std::endl;
    std::cout << "BlockMatcher >> Mode " << (bm == USE_BM ? "BM" : "SGBM") << "," << (enableWLS ? " WLS is enabled." : " WLS is disabled.") << std::endl;
    cv::Mat img1 = m_imgPair.img1;
    cv::Mat img1Clone = img1.clone();
    cv::Mat img2 = m_imgPair.img2;
    cv::Mat img2Clone = img2.clone();
    cv::Mat leftDisp, rightDisp, filteredDisp;

    cv::Ptr<cv::StereoMatcher> sbmRight;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wlsFilter;
    cv::Mat confMap;
    cv::Rect ROI;

    if (bm == USE_BM) {
        cv::cvtColor(img1, img1Clone, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img2, img2Clone, cv::COLOR_BGR2GRAY);
    }

    cv::Ptr<cv::StereoSGBM> sgbmLeft;
    cv::Ptr<cv::StereoBM> sbmLeft;

    if (bm == USE_SGBM) {
        sgbmLeft = cv::StereoSGBM::create(0, maxDisp, wSize);
        sgbmLeft->setP1(24 * wSize * wSize);
        sgbmLeft->setP2(96 * wSize * wSize);
        sgbmLeft->setPreFilterCap(63);
        sgbmLeft->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
        sgbmLeft->compute(img1Clone, img2Clone, leftDisp);
    }
    else {
        sbmLeft = cv::StereoBM::create(maxDisp, wSize);
        sbmLeft->setMinDisparity(0);
        sbmLeft->setNumDisparities(160);
        sbmLeft->setTextureThreshold(10);
        sbmLeft->setUniquenessRatio(8);
        sbmLeft->setSpeckleRange(32);
        sbmLeft->setSpeckleWindowSize(10);
        sbmLeft->setDisp12MaxDiff(-1);
        sbmLeft->compute(img1Clone, img2Clone, leftDisp);
    }

    if (enableWLS) {
        if (bm == USE_SGBM) {
            wlsFilter = cv::ximgproc::createDisparityWLSFilter(sgbmLeft);
            sbmRight = cv::ximgproc::createRightMatcher(sgbmLeft);
        }
        else {
            wlsFilter = cv::ximgproc::createDisparityWLSFilter(sbmLeft);
            sbmRight = cv::ximgproc::createRightMatcher(sbmLeft);
        }
        sbmRight->compute(img2Clone, img1Clone, rightDisp);

        wlsFilter->setLambda(8000.0);
        wlsFilter->setSigmaColor(1.5);
        wlsFilter->filter(leftDisp, img1Clone, filteredDisp, rightDisp);
        confMap = wlsFilter->getConfidenceMap();
        ROI = wlsFilter->getROI();
        cv::resize(filteredDisp, filteredDisp, cv::Size(filteredDisp.cols / 4, filteredDisp.rows / 4));
        cv::resize(confMap, confMap, cv::Size(confMap.cols / 4, confMap.rows / 4));

        cv::Mat visDisp;
        cv::ximgproc::getDisparityVis(filteredDisp, visDisp, 1.0);
        if (bm == USE_BM) {
            cv::imshow("DisparityMap_BM_WLS", visDisp);
            cv::imshow("ConfindenceMap_BM_WLS", confMap);
            std::cout << "BlockMatcher >> WLS filtered Block Matching Done." << std::endl;
        }
        else {
            cv::imshow("DisparityMap_SGBM_WLS", visDisp);
            cv::imshow("ConfindenceMap_SGBM_WLS", confMap);
            std::cout << "BlockMatcher >> WLS filtered Semi Block Matching Done." << std::endl;
        }
        cv::Mat depth = 16 * m_imgPair.f1 * m_imgPair.baseline / filteredDisp;
        // cv::imshow("Some1", depth);
        cv::waitKey(0);
    }
    else {
        //cv::copyMakeBorder(img1Gray, img1Gray, 0, 0, 160, 0, cv::BORDER_REPLICATE);  
        //cv::copyMakeBorder(img2Gray, img2Gray, 0, 0, 160, 0, cv::BORDER_REPLICATE);  

        // disp = disp.colRange(160, img1Gray.cols);  
        cv::resize(leftDisp, leftDisp, cv::Size(leftDisp.cols / 4, leftDisp.rows / 4));
        if (bm == USE_BM) {
            cv::imshow("DisparityMap_BM", leftDisp);
            std::cout << "BlockMatcher >> Normal Block Matching Done." << std::endl;
        }
        else {
            cv::imshow("DisparityMap_SGBM", leftDisp);
            std::cout << "BlockMatcher >> Normal Semi Block Matching Done." << std::endl;
        }
        cv::Mat depth;
        depth = 16 * m_imgPair.f1 * m_imgPair.baseline / leftDisp;
        // cv::imshow("Some1", depth);
        cv::waitKey(0);
        // disp.convertTo(disp8,CV_8U,255 / (numDisparitiesBM * 16.));
    }
}

