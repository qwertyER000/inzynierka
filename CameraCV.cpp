#include "CameraCV.h"



void CVCamera::calculateTvec(ogx::Math::Transform3D viewMatrixGL)
{

	cv::Mat transVector = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1);

	transVector.at<float>(0, 0) = viewMatrixGL(0, 3);
	transVector.at<float>(1, 0) = viewMatrixGL(1, 3);
	transVector.at<float>(2, 0) = viewMatrixGL(2, 3);

	tvec = transVector;
}


void CVCamera::calculateRvec(ogx::Math::Transform3D viewMatrixGL)
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

	rvec = R_matrix;
}


void CVCamera::calculateCameraMatrix(ogx::Math::Matrix4 projectionMatrixGL, double width, double height)
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

	camera_matrix = perCam_cv;
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
void CVCamera::normalizeTranslation(cv::Mat& trans, cv::Point3f size3D)
{
	trans.at<float>(0, 0) = trans.at<float>(0, 0) / abs(size3D.x);
	trans.at<float>(1, 0) = trans.at<float>(1, 0) / abs(size3D.y);
	trans.at<float>(2, 0) = trans.at<float>(2, 0) / abs(size3D.z);
}

//get original translation from normalized
void CVCamera::backTranslation(cv::Mat& trans, cv::Point3f size3D)
{
	trans.at<float>(0, 0) = trans.at<float>(0, 0) * abs(size3D.x);
	trans.at<float>(1, 0) = trans.at<float>(1, 0) * abs(size3D.y);
	trans.at<float>(2, 0) = trans.at<float>(2, 0) * abs(size3D.z);
}

// Calculate Rodriques rotation vector from Euler angles
void CVCamera::eulerAnglesToRvec(cv::Vec3f& euler_angles, cv::Mat& rvec)
{
	// Calculate rotation about x axis
	cv::Mat R_x = (cv::Mat_<float>(3, 3) <<
		1, 0, 0,
		0, cos(euler_angles[0]), -sin(euler_angles[0]),
		0, sin(euler_angles[0]), cos(euler_angles[0])
		);
	// Calculate rotation about y axis
	cv::Mat R_y = (cv::Mat_<float>(3, 3) <<
		cos(euler_angles[1]), 0, sin(euler_angles[1]),
		0, 1, 0,
		-sin(euler_angles[1]), 0, cos(euler_angles[1])
		);
	// Calculate rotation about z axis
	cv::Mat R_z = (cv::Mat_<float>(3, 3) <<
		cos(euler_angles[2]), -sin(euler_angles[2]), 0,
		sin(euler_angles[2]), cos(euler_angles[2]), 0,
		0, 0, 1);
	// Combined rotation matrix
	cv::Mat R = R_z * R_y * R_x;
	cv::Rodrigues(R, rvec);

	R_x.release();
	R_y.release();
	R_z.release();
	R.release();
}

// calculate Euler angles from Rodriques rotation vector
void CVCamera::rvecToEulerAngles(cv::Mat& rvec, cv::Vec3f& euler_angles)
{
	cv::Mat R = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
	cv::Rodrigues(rvec, R);
	//CHECK IF ROTATION MATRIX
	bool is_rotation_matrix;
	cv::Mat Rt;
	cv::transpose(R, Rt);
	cv::Mat shouldBeIdentity = Rt * R;
	cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());
	is_rotation_matrix = (norm(I, shouldBeIdentity) < 1e-6);

	assert(is_rotation_matrix);
	float sy = sqrt(R.at<float>(0, 0) * R.at<float>(0, 0) + R.at<float>(1, 0) * R.at<float>(1, 0));
	bool singular = sy < 1e-6; //
	float x, y, z;
	if (!singular)
	{
		x = atan2(R.at<float>(2, 1), R.at<float>(2, 2));
		y = atan2(-R.at<float>(2, 0), sy);
		z = atan2(R.at<float>(1, 0), R.at<float>(0, 0));
	}
	else
	{
		x = atan2(-R.at<float>(1, 2), R.at<float>(1, 1));
		y = atan2(-R.at<float>(2, 0), sy);
		z = 0;
	}

	euler_angles = { x, y, z };
}
