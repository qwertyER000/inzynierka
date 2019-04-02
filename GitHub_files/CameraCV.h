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

	cv::Mat tvec(ogx::Math::Transform3D projectionMatrixGL);
	cv::Mat rvec(ogx::Math::Transform3D projectionMatrixGL);
	cv::Mat camera_matrix(ogx::Math::Matrix4 cameraMatrixGL, double width, double height);

	void updateView(Context& context, double f, cv::Mat& trans, cv::Mat &rvec, int height_Viewport);

};

