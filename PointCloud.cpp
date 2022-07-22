#include "PointCloud.h"

cv::Mat PointCloud::depthMapFromDisperityMap(cv::Mat disperity, float baseline, float doffs, float focal, float* maxDepth, bool normalize) {
    cv::Mat depth_map(disperity.size(), CV_32FC1);

    int rows = depth_map.rows;
    int cols = depth_map.cols;

    for (int x = 0; x < cols; x++) {
		for (int y = 0; y < rows; y++) {
			float d = disperity.at<unsigned char>(y,x);
			float depth_val = baseline * focal / (d + doffs);
			depth_map.at<float>(y,x) = depth_val;
		}
	}

    double min, max;
	cv::minMaxLoc(depth_map, &min, &max);

    *maxDepth = (float)max;

#ifdef POINT_CLOUD_DEBUG
	std::cout << "[PointCloud::depthMapFromDisperityMap] Depth map minimum value " << min << "\n";
	std::cout << "[PointCloud::depthMapFromDisperityMap] Depth map maximum value " << max << "\n";
#endif

    if (normalize) {
        cv::Mat depth_map_norm;
        cv::normalize(depth_map, depth_map_norm, 0.0, 1.0, cv::NORM_MINMAX);
        *maxDepth = 1.0;
        return depth_map_norm;
    }

    return depth_map;
}

Vertex* PointCloud::generatePointCloud(cv::Mat depthMap, cv::Mat colorMap, cv::Mat depthIntrinsicMat, float maxDepth) {

    int width = depthMap.cols;
    int height = depthMap.rows;

    float fX = depthIntrinsicMat.at<double>(0,0);
    float fY = depthIntrinsicMat.at<double>(1,1);
    float cX = depthIntrinsicMat.at<double>(0,2);
    float cY = depthIntrinsicMat.at<double>(1,2);

    Eigen::Matrix3f depthIntrinsic;
    depthIntrinsic << fX, 0, cX, 0, fY, cY, 0, 0, 1;

    Eigen::Matrix3f depthIntrinsicInv = depthIntrinsic.inverse();
    Vertex* vertices = new Vertex[width * height];

    int idx = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++, idx++) {
            float depth = depthMap.at<float>(y,x);

            if (std::abs(depth - maxDepth) < EPSILON) {
				vertices[idx].position = Eigen::Vector4f(MINF, MINF, MINF, MINF);
				vertices[idx].color = Vector4uc(0, 0, 0, 0);
			} else {
				Eigen::Vector3f img_coords = Eigen::Vector3f(x * depth, y * depth, depth);
				Eigen::Vector3f tmp_coords = depthIntrinsicInv * img_coords;

				Eigen::Vector4f world_coords = Eigen::Vector4f(tmp_coords[0], tmp_coords[1], tmp_coords[2], 1.0f);

				// Format is B,G,R
				cv::Vec3b color = colorMap.at<cv::Vec3b>(y,x);

				vertices[idx].position = world_coords;
				vertices[idx].color = Vector4uc(color[2],color[1],color[0],255);
			}
        }
    }

    return vertices;
}