#include "CameraCV.h"



cv::Mat CVCamera:: tvec(ogx::Math::Transform3D viewMatrixGL)
{

	cv::Mat transVector = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1);

	transVector.at<float>(0, 0) = viewMatrixGL(0, 3);
	transVector.at<float>(1, 0) = viewMatrixGL(1, 3);
	transVector.at<float>(2, 0) = viewMatrixGL(2, 3);

	return transVector;
}


cv::Mat CVCamera::rvec(ogx::Math::Transform3D viewMatrixGL)
{
	cv::Mat R_matrix = cv::Mat::zeros(cv::Size(3, 3), CV_32F);

	//calculate openCV rvec from openGL rotation matrix
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			R_matrix.at<float>(j, i) = viewMatrixGL(j, i);
		}
	}

	//calculate vector from 3x3 matrix
	cv::Rodrigues(R_matrix, R_matrix);

	return R_matrix;
}


cv::Mat CVCamera::camera_matrix(ogx::Math::Matrix4 projectionMatrixGL, double width, double height)
{
	cv::Mat perCam_cv = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);

	//calculate openCV camera matrix parameters
	double f = -0.5*width*projectionMatrixGL(0, 0);
	double cx = (1 - projectionMatrixGL(0, 2)) / 2 * width;
	double cy = (projectionMatrixGL(1, 2) + 1)*0.5*height;

	//openCV camera matrix;
	perCam_cv.at<float>(0, 0) = f;
	perCam_cv.at<float>(1, 1) = -f;
	perCam_cv.at<float>(0, 2) = cx;
	perCam_cv.at<float>(1, 2) = cy;
	perCam_cv.at<float>(2, 2) = 1;

	return perCam_cv;
}


void CVCamera::updateView(Context& context, double f, cv::Mat& trans, cv::Mat &rvec, int height_Viewport)
{
	//field of view
	auto FOVY = 2 * std::atan(0.5 * height_Viewport / (f));
	context.Feedback().GetCamera().GetProjection().SetFOVY(FOVY);


	//ROTATION
	cv::Mat R_matrix = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
	cv::Rodrigues(rvec, R_matrix);
	ogx::Math::Transform3D viewTransform;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			viewTransform(j, i) = R_matrix.at<float>(j, i);
		}
	}

	//TRANSLATION
	viewTransform(0, 3) = trans.at<float>(0, 0);
	viewTransform(1, 3) = trans.at<float>(1, 0);
	viewTransform(2, 3) = trans.at<float>(2, 0);

	context.Feedback().GetCamera().SetViewTransform(viewTransform);
}

//cv::Mat CVCamera::distortion()
//{
//	cv::Mat dist = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1);
//
//	return dist;
//}
