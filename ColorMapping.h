#pragma once
#include <ogx/Plugins/EasyPlugin.h>
#include <ogx/Data/Clouds/CloudHelpers.h>
#include<cstdlib>
#include<functional>
#include<numeric>
#include<algorithm>
#include<vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace ogx;

class renderSilhouette
{
public:
	renderSilhouette();
	virtual ~renderSilhouette();
	cv::Point3f shapeSize;
	std::vector<cv::Point3f> CloudXYZ;

	void GetSilhouette(cv::Mat rvec, cv::Mat tvec, cv::Mat intrinsic_matrix, std::vector<cv::Point3f>& points3D, cv::Mat& silhouette);
	void GetImageToProjection(cv::Mat rvec, cv::Mat tvec, cv::Mat intrinsic_matrix, const std::vector<cv::Point3f>& points3D, const std::vector<double>& distance, cv::Mat& img, cv::Mat& z_bufer);
	cv::Mat RemoveHoles(cv::Mat& img_in);
	void similarityImg(cv::Mat& img_in1, cv::Mat& img_in2, cv::Mat& img_out);
	void pointsToCV_all(ogx::Data::Clouds::PointsRange& points_all);
	void pointsToCV_front(Math::Point3D& camera_pos, Data::Clouds::PointsRange& points_all, std::vector<double>& pointsDist, std::vector<double>& angle);
	
protected:
	cv::Point3f get3dSize(const std::vector<cv::Point3f>& location);
};

