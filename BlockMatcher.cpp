#include "BlockMatcher.h"

BlockMatcher::BlockMatcher(ImagePair imgPair) {
    m_imgPair = imgPair;
}

void BlockMatcher::opBM(int wSize, int maxDisp) {

    cv::Mat img1 = m_imgPair.img1;
    cv::Mat img1Gray;
    cv::Mat img2 = m_imgPair.img2;
    cv::Mat img2Gray;
    cv::Mat disp;

    cv::cvtColor(img1, img1Gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2Gray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::StereoBM> sbmLeft = cv::StereoBM::create(maxDisp, wSize);
    sbmLeft->setMinDisparity(0);
    sbmLeft->setNumDisparities(160);
    sbmLeft->setTextureThreshold(10);
    sbmLeft->setUniquenessRatio(8);
    sbmLeft->setSpeckleRange(32);
    sbmLeft->setSpeckleWindowSize(10);
    sbmLeft->setDisp12MaxDiff(-1);
    sbmLeft->compute(img1Gray, img2Gray, disp);


    //cv::copyMakeBorder(img1Gray, img1Gray, 0, 0, 160, 0, cv::BORDER_REPLICATE);  
    //cv::copyMakeBorder(img2Gray, img2Gray, 0, 0, 160, 0, cv::BORDER_REPLICATE);  
    disp = cv::Mat(cv::Size(img1Gray.cols,img2Gray.rows),CV_16S);    
    sbmLeft->compute(img1Gray,img2Gray,disp);

    // disp = disp.colRange(160, img1Gray.cols);  
    cv::resize(disp, disp, cv::Size(disp.cols / 4, disp.rows / 4));
    cv::imshow("DisparityMap_BM", disp);
    cv::Mat depth;
    depth = 16 * m_imgPair.f1 * m_imgPair.baseline / disp;  
    // cv::imshow("Some1", depth);
    cv::waitKey(0);
    // disp.convertTo(disp8,CV_8U,255 / (numDisparitiesBM * 16.));
}

void BlockMatcher::opBMwithFiltering(int wSize, int maxDisp) {
    cv::Mat img1 = m_imgPair.img1;
    cv::Mat img1Gray;
    cv::Mat img2 = m_imgPair.img2;
    cv::Mat img2Gray;
    cv::Mat leftDisp, rightDisp, filteredDisp;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wlsFilter;
    cv::Mat confMap;
    cv::Rect ROI;

    cv::cvtColor(img1, img1Gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2Gray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::StereoBM> sbmLeft = cv::StereoBM::create(maxDisp, wSize);
    sbmLeft->setMinDisparity(0);
    sbmLeft->setNumDisparities(160);
    sbmLeft->setTextureThreshold(10);
    sbmLeft->setUniquenessRatio(8);
    sbmLeft->setSpeckleRange(32);
    sbmLeft->setSpeckleWindowSize(10);
    sbmLeft->setDisp12MaxDiff(-1);
    wlsFilter = cv::ximgproc::createDisparityWLSFilter(sbmLeft);
    cv::Ptr<cv::StereoMatcher> sbmRight = cv::ximgproc::createRightMatcher(sbmLeft);

    sbmLeft->compute(img1Gray, img2Gray, leftDisp);
    sbmRight->compute(img2Gray, img1Gray, rightDisp);

    wlsFilter->setLambda(8000.0);
    wlsFilter->setSigmaColor(1.5);
    wlsFilter->filter(leftDisp, img1Gray, filteredDisp, rightDisp);
    confMap = wlsFilter->getConfidenceMap();
    ROI = wlsFilter->getROI();

    cv::resize(filteredDisp, filteredDisp, cv::Size(filteredDisp.cols / 4, filteredDisp.rows / 4));

    cv::Mat visDisp;
    cv::ximgproc::getDisparityVis(filteredDisp, visDisp, 1.0);
    cv::imshow("DisparityMap_BM_WLS", visDisp);
    cv::Mat depth;
    depth = 16 * m_imgPair.f1 * m_imgPair.baseline / filteredDisp; 
    // cv::imshow("Some1", depth);
    cv::waitKey(0);
}

void BlockMatcher::opSGBM(int wSize, int maxDisp) {
    cv::Mat img1 = m_imgPair.img1;
    cv::Mat img2 = m_imgPair.img2;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wlsFilter;
    cv::Mat leftDisp, rightDisp;
    cv::Mat disp;

    cv::Ptr<cv::StereoSGBM> sgbmLeft = cv::StereoSGBM::create(0, maxDisp, wSize);


    sgbmLeft->setP1(24 * wSize * wSize);
    sgbmLeft->setP2(96 * wSize * wSize);
    sgbmLeft->setPreFilterCap(63);
    sgbmLeft->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

    sgbmLeft->compute(img1, img2, disp);

    cv::resize(disp, disp, cv::Size(disp.cols / 4, disp.rows / 4));
    cv::imshow("DisparityMap_SGBM", disp);
    cv::Mat depth;
    depth = 16 * m_imgPair.f1 * m_imgPair.baseline / disp;  
    // cv::imshow("Some1", depth);
    cv::waitKey(0);
}

void BlockMatcher::opSGBMwithFiltering(int wSize, int maxDisp) {
    cv::Mat img1 = m_imgPair.img1;
    cv::Mat img2 = m_imgPair.img2;
    cv::Mat leftDisp, rightDisp, filteredDisp;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wlsFilter;
    cv::Mat confMap;
    cv::Rect ROI;

    cv::Ptr<cv::StereoSGBM> sgbmLeft = cv::StereoSGBM::create(0, maxDisp, wSize);

    sgbmLeft->setP1(24 * wSize * wSize);
    sgbmLeft->setP2(96 * wSize * wSize);
    sgbmLeft->setPreFilterCap(63);
    sgbmLeft->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    wlsFilter = cv::ximgproc::createDisparityWLSFilter(sgbmLeft);
    cv::Ptr<cv::StereoMatcher> sgbmRight = cv::ximgproc::createRightMatcher(sgbmLeft);

    sgbmLeft->compute(img1, img2, leftDisp);
    sgbmRight->compute(img2, img1, rightDisp);

    wlsFilter->setLambda(8000.0);
    wlsFilter->setSigmaColor(1.5);
    wlsFilter->filter(leftDisp, img1, filteredDisp, rightDisp);

    confMap = wlsFilter->getConfidenceMap();
    ROI = wlsFilter->getROI();

    cv::resize(filteredDisp, filteredDisp, cv::Size(filteredDisp.cols / 4, filteredDisp.rows / 4));

    cv::Mat visDisp;
    cv::ximgproc::getDisparityVis(filteredDisp, visDisp, 1.0);
    cv::imshow("DisparityMap_SGBM_WLS", visDisp);
    cv::Mat depth;
    depth = 16 * m_imgPair.f1 * m_imgPair.baseline / filteredDisp;
    // cv::imshow("Some1", depth);
    cv::waitKey(0);
}