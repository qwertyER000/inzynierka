#include <ogx/Plugins/EasyPlugin.h>
#include <ogx/Math/Calculation.h>
#include <ogx/Math/Optimization.h>
#include <ogx/Data/Clouds/CloudHelpers.h>
#include <ogx/Data/Clouds/SearchKernels.h>



#include"CameraCV.h"
#include "ColorMapping.h"


using namespace ogx;
using namespace ogx::Data;

struct ImageMapping : public ogx::Plugin::EasyMethod, ogx::Math::ModelI
{
	// INPUT
	Data::ResourceID node_id;
	String img_path;
	int height;
	//SOLVER
	double min_dist;
	int	width, with_new_fram, m_far;
	std::vector<cv::Point3f> CloudXYZ;
	cv::Mat camera_matrix, trans_start, translation = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1);
	cv::Vec3f rotation, rotation_start;
	cv::Point2d center_of_mass;
	cv::Point3d size_of_3d_model;
	std::vector<double> img_distance;
	std::vector<double> obj_distance;
	size_t count = 0;
	double tol = 0.001;
	float f;
	cv::Mat silhuette2D;

	//cv::Mat silhuette_rotated;
	//cv::Mat rotated;
	//cv::Mat M = cv::Mat::zeros(cv::Size(2,3), CV_32FC1);

	// constructor
	ImageMapping() : EasyMethod(L"Ewa Rycaj", L"Image to geometry registration")
	{
	}

	// add input/output parameters
	virtual void DefineParameters(ParameterBank& bank)
	{
		//bank.Add(L"node_id", image_id).AsNode();
		bank.Add(L"node id", node_id).AsNode();
		bank.Add(L"image path", img_path = L"C:\\Users\\Ewa\\Desktop\\20181108_dane_do_inz_ERycaj\\black.png").AsFile();
		bank.Add(L"height", height = 600, L"Choose output image vertical resolution. If height=-1 resolution is set as input. ");
	}

	//load texture image
	virtual void LoadImageCV(String path, cv::Mat& camera_img, int& img_height)
	{
		std::string name(path.begin(), path.end());

		cv::Mat input_img = cv::imread(name);
		if (input_img.empty()) ReportError(L"image not found");



		if (img_height == -1)
		{
			img_height = input_img.rows;
			input_img.copyTo(camera_img);
		}
		else
		{
			int img_width = input_img.cols*img_height / input_img.rows;
			cv::resize(input_img, camera_img, cv::Size(img_width, img_height));
		}

		input_img.release();
	}

	//silhouette from camera image
	virtual void silhouetteFromImage(cv::Mat &img_in, cv::Mat &img_out)
	{
		cv::Mat thres;

		if (img_in.channels() == 3)
		{
			cv::cvtColor(img_in, img_in, CV_BGR2GRAY);
		}

		if (img_in.channels() == 4)
		{
			cv::cvtColor(img_in, img_in, CV_BGRA2GRAY);
		}

		cv::threshold(img_in, thres, 40, 255, cv::THRESH_BINARY);

		cv::medianBlur(thres, img_out, 3);
		thres.release();
	}
	
	//normalize translation to model size
	virtual void normalizeTranslation(cv::Mat& trans, cv::Point3f size3D)
	{
		trans.at<float>(0, 0) = trans.at<float>(0, 0) / abs (size3D.x);
		trans.at<float>(1, 0) = trans.at<float>(1, 0) / abs (size3D.y);
		trans.at<float>(2, 0) = trans.at<float>(2, 0) / abs (size3D.z);
	}
	//get original translation from normalized
	virtual void backTranslation(cv::Mat& trans, cv::Point3f size3D)
	{
		trans.at<float>(0, 0) = trans.at<float>(0, 0) * abs (size3D.x);
		trans.at<float>(1, 0) = trans.at<float>(1, 0) * abs(size3D.y);
		trans.at<float>(2, 0) = trans.at<float>(2, 0) * abs(size3D.z);
	}
	
	//get N radiuses from mass center to object's contour
	virtual void shapeDescriptor(std::vector<std::vector<cv::Point>>& contours, std::vector<double>& distance, const cv::Point2d& mass_center)
	{
		int N = 360;

		std::vector<cv::Point2d> c_polar;
		//CARTESIAN TO POLAR
		double angle, radius, x, y;
		for (auto& iter : contours[0])
		{
			x = (double)iter.x - mass_center.x;
			y = (double)iter.y - mass_center.y;
			radius = sqrt(x*x + y * y);
			if (x > 0 && y > 0)	angle = atan(y / x) * 180 / CV_PI;
			if (x < 0 && y>0) angle = atan(y / x) * 180 / CV_PI + 180;
			if (x < 0 && y < 0) angle = atan(y / x) * 180 / CV_PI + 180;
			if (x > 0 && y < 0) angle = atan(y / x) * 180 / CV_PI + 360;

			c_polar.push_back(cv::Point2d(radius, angle));
		}

		std::sort(c_polar.begin(), c_polar.end(), [](cv::Point2d const &a, cv::Point2d const &b) { return a.y < b.y; });

		double step = 360.f / N;
		double start_angle = std::round(c_polar[0].y);
		//GET RADIUS 
		for (auto& iter_point : c_polar)
		{
			if (start_angle == 360) break;

			if (start_angle <= iter_point.y)
			{
				distance.push_back(iter_point.x);
				start_angle += step;
			}
		}
	}
	
	// find center of mass
	virtual void findMassCenter(const std::vector<cv::Point>& contours, cv::Point2d& mass_center)
	{
		cv::Moments mu;
		mu = cv::moments(contours, true);
		mass_center = cv::Point2d(mu.m10 / mu.m00, mu.m01 / mu.m00);
	}

	// calculate Euler angles from Rodriques rotation vector
	virtual void rvecToEulerAngles(cv::Mat& rvec, cv::Vec3f& euler_angles)
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
	
	// Calculate Rodriques rotation vector from Euler angles
	virtual void eulerAnglesToRvec(cv::Vec3f& euler_angles, cv::Mat& rvec)
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
	//scale angles for optimization
	virtual void normalizeAngles(cv::Vec3f& input, cv::Vec3f& output)
	{
		float scaling_factor = 1.0;
		output = { input[0] * scaling_factor, input[1] * scaling_factor, input[2] * scaling_factor };
	}
	//scale angles back
	virtual void anglesFromNorm(cv::Vec3f& input, cv::Vec3f& output)
	{
		float scaling_factor = 1.0;
		output = { input[0] * scaling_factor,input[1] * scaling_factor, input[2] * scaling_factor };
	}
	
	virtual void Run(Context& context)
	{
		auto subtree = context.Project().TransTreeFindNode(node_id);
		// report error if give node was not found, this will stop execution of algorithm
		if (!subtree) ReportError(L"Node not found");

		//CAMERA
		with_new_fram = height * context.Feedback().GetCamera().GetProjection().GetAspect();
		CVCamera camera;
		camera_matrix = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
		camera_matrix = camera.camera_matrix(context.Feedback().GetCamera().GetProjection().GetMatrix(), with_new_fram, height); //camera matrix
		f = camera_matrix.at<float>(1, 1);
		trans_start = camera.tvec(context.Feedback().GetCamera().GetViewTransform()); // translation vector
		//translation = camera.tvec(context.Feedback().GetCamera().GetViewTransform()); // translation vector
		cv::Mat rvec = camera.rvec(context.Feedback().GetCamera().GetViewTransform()); //rotation vector
		rvecToEulerAngles(rvec, rotation_start);
		auto camera_position = context.Feedback().GetCamera().GetPosition(); //get current camera position

		//TEXTURE IMG
		cv::Mat imgCam;
		
		LoadImageCV(img_path, imgCam, height);
		silhouetteFromImage(imgCam, silhuette2D);
		width = silhuette2D.cols;
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(silhuette2D, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		//cv::GaussianBlur(silhuette2D, silhuette2D, cv::Size(10, 10), 0, 0);
		//GET IMAGE SHAPE DESCRIPTOR
		findMassCenter(contours[0], center_of_mass);
		shapeDescriptor(contours, img_distance, center_of_mass);
		count = height * width / 4;
		//min_dist = *std::min_element(img_distance.begin(), img_distance.end());
	
		
		//3D MODEL TO CV
		renderSilhouette obj;
		// perform calculation for each cloud
			// run with number of threads available on current machine, optional
			auto const thread_count = std::thread::hardware_concurrency();

		Clouds::ForEachCloud(*subtree, [&](Data::Clouds::ICloud & cloud, Data::Nodes::ITransTreeNode & node)
		{
			Data::Clouds::PointsRange points;
			cloud.GetAccess().GetAllPoints(points); //get all points

			obj.pointsToCV_all(points, CloudXYZ);
		}, thread_count);
		
		size_of_3d_model = obj.shapeSize;
		normalizeTranslation(trans_start, size_of_3d_model);

		rotation = rotation_start;
		 
		//START OPTIMIZATION
		auto solver = Math::SolverI::Create();
		solver->Solve(*this);

		//UPDATE VIEW
		eulerAnglesToRvec(rotation, rvec);
		//f = camera_matrix.at<float>(1, 1);
		camera.updateView(context, f, translation, rvec, height);
		delete solver;
	}

	virtual Count GetResultCount()
	{
		return count;
	}

	virtual Count GetParamCount()
	{
		/*return 6;*/
		return 3;
	}

	virtual void InitParams(Math::Vector& params)
	{

		//params[0] = camera_matrix.at<float>(1, 1)/m_far*1000;

		params[0] = 0;
		params[1] = 0;
		params[2] = 0;
		//params[3] = 0;
		//params[4] = 0;
		//params[5] = 0;
		//params[6] = 0;

	}

	virtual bool Iterate(const Math::Vector& params, Math::Vector& results)
	{

		cv::Mat framSilhouette = cv::Mat::zeros(cv::Size(with_new_fram, height), CV_8UC1);

		//camera_matrix.at<float>(0, 0) = -params[6] * 100000000 + f;
		//camera_matrix.at<float>(1, 1) = params[6] * 100000000 + f;
		rotation[0] = rotation_start[0] + 10000 * params[0];
		rotation[1] = rotation_start[1] + 10000 * params[1];
		rotation[2] = rotation_start[2] + 10000 * params[2];

		//translation.at<float>(0, 0) = (trans_start.at<float>(0, 0) + 2000000 * (float) params[0]);
		//translation.at<float>(1, 0) = (trans_start.at<float>(1, 0) + 2000000 * (float)params[1]);
		//translation.at<float>(2, 0) = (trans_start.at<float>(2, 0) + 2000000 * (float)params[2]);
		//backTranslation(translation, size_of_3d_model);

		translation.at<float>(0, 0) = trans_start.at<float>(0, 0);
		translation.at<float>(1, 0) = trans_start.at<float>(1, 0);
		translation.at<float>(2, 0) = trans_start.at<float>(2, 0);
		backTranslation(translation, size_of_3d_model);

		cv::Mat tmp = silhuette2D.clone();
		cv::Mat rvec;
		eulerAnglesToRvec(rotation, rvec);

		renderSilhouette render3d;
		render3d.GetSilhouette(rvec, translation, camera_matrix, CloudXYZ, framSilhouette);
		cv::Rect ROI(framSilhouette.cols / 2 - width / 2, 0, width, height);
		cv::Mat img3D = framSilhouette(ROI);
		cv::Mat silhuette3D = render3d.RemoveHoles(img3D);

		//std::vector < std::vector<cv::Point>> contours;
		//cv::findContours(silhuette3D, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		//shapeDescriptor(contours, obj_distance, center_of_mass);

		cv::Mat blured2D;
		cv::Mat diff;
		cv::Mat blured3D;// = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);
		
		cv::GaussianBlur(silhuette3D, blured3D, cv::Size(61, 61), 0);
		cv::GaussianBlur(tmp, blured2D, cv::Size(61, 61), 0);
		cv::absdiff(blured2D, blured3D, diff);
		cv::resize(diff, diff, cv::Size(width/2, height/2), 0, 0, CV_INTER_AREA);
		/*for (int i=0; i < count; ++i)
		{
			results(i) = abs (img_distance[i] - obj_distance[i]);
		}*/
		for (int a = 0; a < diff.rows; a++)
		{
			for (int b = 0; b < diff.cols; b++)
			{
				results(diff.cols * a + b) = float(diff.at<uchar>(a, b)) / 255.0;
			}
		}
		printf("%f \n", results[0]);
		//obj_distance.clear();
		//contours.clear();
		rvec.release();
		img3D.release();
		silhuette3D.release();

		return true;
	}

	virtual void SaveSolution(const Math::Vector& params)
	{
		//OGX_LINE.Format(ogx::Info, L"Current solver solution is: %f, %f, %f, %f, %f, %f, %f", params[0] * 500000, params[1] * 500000, params[2] * 500000); //, params[3] * 100000, params[4] * 100000, params[5] * 100000);
	}

};

OGX_EXPORT_METHOD(ImageMapping)