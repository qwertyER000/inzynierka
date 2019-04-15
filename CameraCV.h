#pragma once
#include <ogx/Plugins/EasyPlugin.h>
#include <ogx/Data/Clouds/CloudHelpers.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>

class CVCamera : public ogx::Plugin::EasyMethod
{
public:
	cv::Mat camera_matrix;
	cv::Mat tvec;
	cv::Mat rvec;

	//calculate OpenCV Camera from OpenGL
	void calculateTvec(ogx::Math::Transform3D projectionMatrixGL);
	void calculateRvec(ogx::Math::Transform3D projectionMatrixGL);
	void calculateCameraMatrix(ogx::Math::Matrix4 cameraMatrixGL, double width, double height);

	//noramlize translation & rotation
	void backTranslation(cv::Mat& trans, cv::Point3f size3D);
	void normalizeTranslation(cv::Mat& trans, cv::Point3f size3D);
	void rvecToEulerAngles(cv::Mat& rvec, cv::Vec3f& euler_angles);
	void eulerAnglesToRvec(cv::Vec3f& euler_angles, cv::Mat& rvec);

	void updateView(Context& context, double f, cv::Mat& trans, cv::Mat &rvec, int height_Viewport);
	
};

