#include <ogx/Plugins/EasyPlugin.h>
#include <ogx/Math/Calculation.h>
#include <ogx/Math/Optimization.h>
#include <ogx/Data/Clouds/CloudHelpers.h>
#include <ogx/Data/Clouds/SearchKernels.h>



#include"CameraCV.h"
#include "ColorMapping.h"


using namespace ogx;
using namespace ogx::Data;

struct OptimizeRotation : protected ogx::Math::ModelI
{
private:
	size_t count = 0;
public:
	int scale_factor;
	cv::Mat input2D;
	//cv::Mat rvec;
	//cv::Vec3f rotation;
	//cv::Vec3f translation_start;
	//cv::Mat translation_start;
	//cv::Vec3f  rotation_start;
	//cv::Mat tvec;
	Math::Vector3D focus_point;
	Math::Vector3D axisX;
	Math::Vector3D axisY;
	Math::Vector3D axisZ;
	Math::Transform3D viewTrans;
	Math::Transform3D view;
	cv::Mat camera_matrix;
	//cv::Mat cam_pos;
	//cv::Mat focus_point;
	CVCamera camera;
	int height, width, width_new_fram;
	cv::Point3f shapeSize;
	std::vector<cv::Point3f> points3D;

	//virtual void getNewCameraTransformation(cv::Vec3f& angle, const Math::Vector3D& pivot, cv::Mat& tvec, cv::Mat& rvec)
	//{
	//	cv::Mat r, R, transform, transform_new;
	//	camera.eulerAnglesToRvec(angle, r);

	//	cv::Rodrigues(r, R);
	//	cv::Rodrigues(rvec, transform);
	//	cv::Mat row = (cv::Mat_<float>(1, 3) << tvec.at<float>(0, 0) + pivot(0), tvec.at<float>(1, 0) + pivot(1), tvec.at<float>(2, 0) + pivot(2));
	//	transform.push_back(row);
	//	cv::transpose(transform, transform);
	//	transform = R * transform;
	//	cv::transpose(transform, transform);
	//	cv::Rect rec(0, 0, 3, 3);
	//	for (int i = 0; i < 3; i++)
	//	{
	//		tvec.at<float>(i, 0) = transform.at<float>(3, i) - pivot(i);
	//	}
	//	cv::Mat R_fin = transform(rec);
	//	cv::Rodrigues(R_fin, rvec);
	//	
	//}

	//virtual void getNewCameraTransform( cv::Vec3f& angle, cv::Mat& rvec, cv::Mat& tvec)
	//{
	//	cv::Mat t, r, R, R_n;
	//	cv::add(tvec, focus_point, t);
	//	t = -t;
	//	cv::Rodrigues(rvec, R);
	//	camera.eulerAnglesToRvec(angle, r);
	//	cv::Rodrigues(r, R_n);
	//	R = R_n * R;
	//	t = R_n * t;
	//	cv::Rodrigues(R, rvec);
	//	cv::add(t, focus_point, t);
	//	tvec = -t;
	//}
	virtual void getNewCameraTransform(Math::Vector3D const pivot, Math::Vector3D const axis, Real const angle, Math::Transform3D& view_transform)
	{
		Math::Point3D const v_center = view_transform * pivot, v_axis = view_transform.rotation() * axis;
		view_transform = Eigen::Translation3d(v_center) * Eigen::AngleAxisd(angle, v_axis) * Eigen::Translation3d(-v_center) * view_transform;
	}


	virtual void Optimize()
	{
		if (scale_factor != 1)
		{
			count = height * width / (2 * abs(scale_factor));
		}
		else
		{
			count = height * width;
		}

		auto solver = Math::SolverI::Create();
		solver->Solve(*this);
		delete solver;
	}

private:

	virtual Count GetResultCount()
	{
		return count;
	}

	virtual Count GetParamCount()
	{
		return 3;
	}

	virtual void InitParams(Math::Vector& params)
	{
		params[0] = 0;
		params[1] = 0;
		params[2] = 0;
	}

	virtual bool Iterate(const Math::Vector& params, Math::Vector& results)
	{

		cv::Mat framSilhouette = cv::Mat::zeros(cv::Size(width_new_fram, height), CV_8UC1);

		Math::Vector3D solution;
		solution[0] = (-1)*scale_factor * 500000 * params[0];
		solution[1] = (-1)*scale_factor * 500000 * params[1];
		solution[2] = (-1)*scale_factor * 100000 * params[2];

		view = viewTrans;
		//rotation[0] = rotation_start[0] + solution[0];
		//rotation[1] = rotation_start[1] + solution[1];
		//rotation[2] = rotation_start[2] + solution[2];
	
		//cv::Mat rvec;
		//camera.eulerAnglesToRvec(rotation_start, rvec);

		getNewCameraTransform(focus_point, axisX, solution(0), view);
		getNewCameraTransform(focus_point, axisY, solution(1), view);
		getNewCameraTransform(focus_point, axisZ, solution(2), view);
		cv::Mat rvec, tvec;
		camera.calculateRvec(view);
		camera.calculateTvec(view);
		tvec = camera.tvec;
		rvec = camera.rvec;


		//getNewCameraPosition(solution);
		//getNewCameraTransformation(solution, focus_point, tvec, rvec);
		//camera.rvecToEulerAngles(rvec, rotation);

		renderSilhouette render3d;
		render3d.GetSilhouette(rvec, tvec, camera_matrix, points3D, framSilhouette);
		cv::Rect ROI(framSilhouette.cols / 2 - width / 2, 0, width, height);
		cv::Mat img3D = framSilhouette(ROI);
		cv::Mat silhuette3D = render3d.RemoveHoles(img3D);

		cv::Mat blured2D;
		cv::Mat diff;
		cv::Mat blured3D;

		if (width < 61 || height < 61) return false;

		cv::GaussianBlur(silhuette3D, blured3D, cv::Size(61, 61), 0);
		cv::GaussianBlur(input2D, blured2D, cv::Size(61, 61), 0);
		cv::absdiff(blured2D, blured3D, diff);
		if (abs(scale_factor)!=1)
			cv::resize(diff, diff, cv::Size(width / abs(scale_factor), height / abs(scale_factor)), 0, 0, CV_INTER_AREA);

		for (int a = 0; a < diff.rows; a++)
		{
			for (int b = 0; b < diff.cols; b++)
			{
				results(diff.cols * a + b) = float(diff.at<uchar>(a, b)) / 255.0;
			}
		}
		img3D.release();
		silhuette3D.release();
		blured2D.release();
		blured3D.release();
		diff.release();

		return true;
	}
};

struct OptimizeTranslation : public ogx::Math::ModelI
{
private:
	size_t count = 0;
public:
	int scale_factor;
	cv::Mat input2D;
	cv::Mat translation = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1);
	cv::Mat translation_start;
	//Math::Vector3D AxisX;
	//Math::Vector3D AxisY;
	//Math::Vector3D AxisZ;

	cv::Mat rvec;
	cv::Mat camera_matrix;
	CVCamera camera;
	int height, width, width_new_fram;
	cv::Point3f shapeSize;
	std::vector<cv::Point3f> points3D;

	//virtual void translate(const Math::Vector3D& solution, cv::Mat& translate)
	//{
	//	cv::Vec3f X = { (float)AxisX(0), (float)AxisX(1), (float)AxisX(2) };
	//	cv::Vec3f Y = { (float)AxisY(0), (float)AxisY(1), (float)AxisY(2) };
	//	cv::Vec3f Z = { (float)AxisZ(0), (float)AxisZ(1), (float)AxisZ(2) };
	//	cv::Vec3f t_x= { (float)solution(0), (float)solution(0), (float)solution(0) };
	//	cv::Vec3f t_y = { (float)solution(1), (float)solution(1), (float)solution(1) };
	//	cv::Vec3f t_z = { (float)solution(2), (float)solution(2), (float)solution(2) };
	//	for (int i = 0; i < 3; i++)
	//	{
	//		X[i] = t_x[i] * X[i];
	//		Y[i] = t_y[i] * Y[i];
	//		Z[i] = t_z[i] * Z[i];
	//	}

	//	
	//	translate = (cv::Mat_<float>(3, 1) << X[0] + Y[0] + Z[0], X[1] + Y[1] + Z[1], X[2] + Y[2] + Z[2]);
	//}

	virtual void Optimize()
	{
		if (scale_factor != 1)
		{
			count = height * width / (2 * abs(scale_factor));
		}
		else
		{
			count = height * width;
		}

		auto solver = Math::SolverI::Create();
		solver->Solve(*this);
		delete solver;
	}

private:

	virtual Count GetResultCount()
	{
		return count;
	}

	virtual Count GetParamCount()
	{
		return 3;
	}

	virtual void InitParams(Math::Vector& params)
	{
		params[0] = 0;
		params[1] = 0;
		params[2] = 0;
	}

	virtual bool Iterate(const Math::Vector& params, Math::Vector& results)
	{

		cv::Mat framSilhouette = cv::Mat::zeros(cv::Size(width_new_fram, height), CV_8UC1);
		//cv::Mat t = cv::Mat::zeros(translation_start.size(), translation_start.type());
		//Math::Vector3D solution;
		//solution(0)=  scale_factor * 100000 * (float) params[0];
		//solution(1) =  scale_factor * 100000 * (float) params[1];
		//solution(2) = scale_factor * 100000 * (float)params[2];
		//translate(solution, t);
		//cv::add(translation_start, t, translation);


		translation.at<float>(0, 0) = (translation_start.at<float>(0, 0) + scale_factor * 100000 * (float) params[0]);
		translation.at<float>(1, 0) = (translation_start.at<float>(1, 0) + scale_factor * 100000 * (float) params[1]);
		translation.at<float>(2, 0) = (translation_start.at<float>(2, 0) + scale_factor * 100000 * (float) params[2]);

		camera.backTranslation(translation, shapeSize);


		renderSilhouette render3d;
		render3d.GetSilhouette(rvec, translation, camera_matrix, points3D, framSilhouette);
		cv::Rect ROI(framSilhouette.cols / 2 - width / 2, 0, width, height);
		cv::Mat img3D = framSilhouette(ROI);
		cv::Mat silhuette3D = render3d.RemoveHoles(img3D);

		cv::Mat blured2D;
		cv::Mat diff;
		cv::Mat blured3D;

		if (width < 61 || height < 61) return false;
		cv::GaussianBlur(silhuette3D, blured3D, cv::Size(61, 61), 0);
		cv::GaussianBlur(input2D, blured2D, cv::Size(61, 61), 0);
		cv::absdiff(blured2D, blured3D, diff);
		if (abs(scale_factor) != 1)
			cv::resize(diff, diff, cv::Size(width/ abs(scale_factor), height / abs(scale_factor)), 0, 0, CV_INTER_AREA);

		for (int a = 0; a < diff.rows; a++)
		{
			for (int b = 0; b < diff.cols; b++)
			{
				results(diff.cols * a + b) = float(diff.at<uchar>(a, b)) / 255.0;
			}
		}
		img3D.release();
		silhuette3D.release();
		blured2D.release();
		blured3D.release();
		diff.release();

		return true;
	}

	virtual void SaveSolution(const Math::Vector& params)
	{

	}
};

struct ImageToGeometryRegistration : public ogx::Plugin::EasyMethod, OptimizeRotation, OptimizeTranslation
{

	// INPUT PARAMS
	Data::ResourceID node_id;
	String img_path;
	int h;

	//SOLVER PARAMS
	float f;
	
	//size_t count = 0;
	//cv::Mat start_translation, translation = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1);
	//renderSilhouette render3d;
	//CVCamera cam;
	int w, width_from_frames;

	//cv::Mat silhuette_rotated;
	//cv::Mat rotated;
	//cv::Mat M = cv::Mat::zeros(cv::Size(2,3), CV_32FC1);

	// constructor
	ImageToGeometryRegistration() : EasyMethod(L"Ewa Rycaj", L"Image to geometry registration")
	{
	}

	// add input/output parameters
	virtual void DefineParameters(ParameterBank& bank)
	{
		//bank.Add(L"node_id", image_id).AsNode();
		bank.Add(L"node id", node_id).AsNode();
		bank.Add(L"image path", img_path).AsFile();
		bank.Add(L"height", h = -1, L"Choose output image vertical resolution. If height=-1 resolution is set as input. ");
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
	void silhouetteFromImage(cv::Mat &img_in, cv::Mat &img_out)
	{
		//cv::Mat binary = cv::Mat::zeros(cv::Size(img_in.cols, img_in.rows), CV_8UC1);
		if (img_in.channels() == 3)
		{
			cv::cvtColor(img_in, img_in, CV_BGR2GRAY);
		}

		if (img_in.channels() == 4)
		{
			cv::cvtColor(img_in, img_in, CV_BGRA2GRAY);
		}
		cv::Mat min;

		//initialize histogram
		int histogram[256];
		for (int i = 0; i < 256; i++)
		{
			histogram[i] = 0;
		}

		for (int i = 0; i < img_in.rows; i++)
		{
			for (int j = 0; j < img_in.cols; j++)
			{
				histogram[(int)img_in.at<uchar>(i, j)]++;
			}
		}
		std::vector<int> thres;
		int prev, next;
		int tol = img_in.cols*img_in.rows*0.008;

		for (int i = 1; i < 255; i++)
		{
			if ((histogram[i - 1] > histogram[i]))
				prev = -1;
			else
				prev = 1;
			if (histogram[i + 1] > histogram[i])
				next = 1;
			else
				next = -1;
			if (prev == -1 && next == 1)
				thres.push_back(i - 2);
		}
		if (!thres.empty())
		{
			cv::threshold(img_in, min, thres[0], 255, cv::THRESH_BINARY);
		}
		cv::medianBlur(min, img_out, 3);
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(img_out, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
		int hole_counter = 0;
		cv::Vec4i current_hierarchy;
		int n = 1;
		for (int i = 0; i < contours.size(); i++)
		{
			current_hierarchy = hierarchy[i];
			if (current_hierarchy[2] == -1 && std::fabs(cv::contourArea(contours[i])) < img_out.cols*img_out.rows*0.008)
			{
				cv::drawContours(img_out, contours, i, 255, CV_FILLED);
			}
		}
		cv::Mat element = getStructuringElement(CV_SHAPE_RECT, cv::Size(5, 5));
		cv::erode(img_out, img_out, element);
		cv::dilate(img_out, img_out, element);

		min.release();
	}


	virtual void optimizeFOV(const cv::Mat& rvec, const cv::Mat& tvec, cv::Mat& matrix, renderSilhouette& obj, const cv::Mat& img2D)
	{

		cv::Mat fram = cv::Mat::zeros(cv::Size(width_from_frames, h), CV_8UC1);

		std::vector<std::vector<cv::Point>> contours2D, contours3D;
		float p_f = matrix.at<float>(1, 1);
		cv::findContours(img2D, contours2D, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		auto area2D = cv::contourArea(contours2D[0]);
		double df = 30;
		double prev;
		int n = 0;
		double tol = img2D.cols*img2D.rows*0.008;

		double f = p_f;
		obj.GetSilhouette(rvec, tvec, matrix, obj.CloudXYZ, fram);
		cv::Rect ROI(fram.cols / 2 - img2D.cols / 2, 0, img2D.cols, img2D.rows);
		cv::Mat tmp = fram(ROI);
		cv::Mat img3D = obj.RemoveHoles(tmp);
		cv::findContours(img3D, contours3D, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		auto area3D = cv::contourArea(contours3D[0]);
		double similarity = abs(area2D - area3D);

		while (similarity >= tol)
		{
			f = f - df;
			prev = similarity;
			matrix.at<float>(1, 1) = f;
			matrix.at<float>(0, 0) = -f;
			obj.GetSilhouette(rvec, tvec, matrix, obj.CloudXYZ, fram);
			tmp = fram(ROI);
			img3D = obj.RemoveHoles(tmp);
			cv::findContours(img3D, contours3D, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			area3D = cv::contourArea(contours3D[0]);
			similarity = abs(area2D - area3D);
			if (prev <= similarity)
			{
				f = p_f;
				df = df / 2;
				n++;
			}
			if (n == 10) break;
		}
		f = p_f;
		df = 30;
		n = 0;
		obj.GetSilhouette(rvec, tvec, matrix, obj.CloudXYZ, fram);
		tmp = fram(ROI);
		img3D = obj.RemoveHoles(tmp);
		cv::findContours(img3D, contours3D, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		area3D = cv::contourArea(contours3D[0]);
		similarity = abs(area2D - area3D);

		while (similarity >= tol)
		{
			f = f + df;
			prev = similarity;
			matrix.at<float>(1, 1) = f;
			matrix.at<float>(0, 0) = -f;
			obj.GetSilhouette(rvec, tvec, matrix, obj.CloudXYZ, fram);
			tmp = fram(ROI);
			img3D = obj.RemoveHoles(tmp);
			cv::findContours(img3D, contours3D, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			area3D = cv::contourArea(contours3D[0]);
			similarity = abs(area2D - area3D);
			if (prev <= similarity)
			{
				f = p_f;
				df = df / 2;
				n++;
			}
			if (n == 10) break;
		}

			

	}

	virtual void getNewCameraPosition(const cv::Mat& rvec, const Math::Vector3D& pivot, cv::Mat& tvec, const Math::Vector3D& camera_position)
	{
		cv::Rodrigues(rvec, rvec);
		Math::Matrix3 camMatrix = Math::Matrix3::Zero();
		//camMatrix <<
		//	rvec.at<float>(0, 0), rvec.at<float>(1, 0), rvec.at<float>(2, 0),
		//	rvec.at<float>(0, 0), rvec.at<float>(1, 0), rvec.at<float>(2, 0),
		//	rvec.at<float>(0, 0), rvec.at<float>(1, 0), rvec.at<float>(2, 0);
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				camMatrix(j, i) = rvec.at<float>(j, i);
			}
		}

		Math::Vector3D camFocus = camera_position - pivot;

		Math::Vector3D new_position;
		new_position = camFocus.transpose() * camMatrix;

		new_position += camFocus.transpose();
		new_position = -camMatrix * new_position;
		tvec.at<float>(0, 0) = new_position(0) - camera_position(0);
		tvec.at<float>(1, 0) = new_position(0) - camera_position(1);
		tvec.at<float>(2, 0) = new_position(0) - camera_position(2);

	}
	
	virtual void Run(Context& context)
	{

		auto subtree = context.Project().TransTreeFindNode(node_id);
		// report error if give node was not found, this will stop execution of algorithm
		if (!subtree) ReportError(L"Node not found");

		//TEXTURE IMG
		cv::Mat imgCam;
		cv::Mat silhuette2D;
		LoadImageCV(img_path, imgCam, h);
		silhouetteFromImage(imgCam, silhuette2D);
		w = silhuette2D.cols;

		//CAMERA
		cv::Vec3f rot;
		cv::Mat transVec, transVecNorm, rotVec;
		width_from_frames = h * context.Feedback().GetCamera().GetProjection().GetAspect();
		CVCamera cam;
		auto viewMatrix = context.Feedback().GetCamera().GetViewTransform();
		cam.calculateCameraMatrix(context.Feedback().GetCamera().GetProjection().GetMatrix(), width_from_frames, h); //camera matrix
		cam.calculateTvec(context.Feedback().GetCamera().GetViewTransform()); //translation vector
		cam.calculateRvec(context.Feedback().GetCamera().GetViewTransform()); //rotation vector
		auto camera_position = context.Feedback().GetCamera().GetPosition(); //get current camera position
		f = cam.camera_matrix.at<float>(1, 1);

		auto camX = context.Feedback().GetCamera().GetXAxis();
		auto camY = context.Feedback().GetCamera().GetYAxis();
		auto camZ = context.Feedback().GetCamera().GetZAxis();
		auto pivot = context.Feedback().GetFocusPoint();
		//cv::Mat focus_point = (cv::Mat_<float>(3, 1) << pivot(0), pivot(1), pivot(2));
		//cv::Mat camera_pos = (cv::Mat_<float>(3, 1) << camera_position(0), camera_position(1), camera_position(2));
	
		
		//3D MODEL TO CV
		renderSilhouette object;
		// perform calculation for each cloud
			// run with number of threads available on current machine, optional
			auto const thread_count = std::thread::hardware_concurrency();

		Clouds::ForEachCloud(*subtree, [&](Data::Clouds::ICloud & cloud, Data::Nodes::ITransTreeNode & node)
		{
			Data::Clouds::PointsRange points;
			cloud.GetAccess().GetAllPoints(points); //get all points

			object.pointsToCV_all(points);
		}, thread_count);


		
		cv::Mat center;
		object.shapeSize = object.get3dSize(object.CloudXYZ, center);
		//transVecNorm = cam.tvec.clone();
		transVec = cam.tvec.clone();
		rotVec = cam.rvec.clone();
		//camera.normalizeTranslation(transVecNorm, object.shapeSize);
		//cam.rvecToEulerAngles(rotVec, rot);
		//cv::Mat camMat = cam.camera_matrix;
		optimizeFOV(rotVec, transVec, cam.camera_matrix, object, silhuette2D);

		cam.updateView(context, cam.camera_matrix.at<float>(1, 1), transVec, rotVec, h);
		


		OptimizeTranslation optimTrans;
		optimTrans.rvec = rotVec;
		//optimTrans.translation_start = transVecNorm;
		optimTrans.camera_matrix = cam.camera_matrix;
		optimTrans.width_new_fram = width_from_frames;
		optimTrans.height = h;
		optimTrans.width = w;
		optimTrans.input2D = silhuette2D;
		optimTrans.points3D = object.CloudXYZ;
		optimTrans.shapeSize = object.shapeSize;



		OptimizeRotation optimRot;
		//optimRot.rotation_start = rotVec;
		
		optimRot.camera_matrix = cam.camera_matrix;
		optimRot.width_new_fram = width_from_frames;
		optimRot.height = h;
		optimRot.width = w;
		optimRot.input2D = silhuette2D;
		optimRot.points3D = object.CloudXYZ;
		optimRot.shapeSize = object.shapeSize;
		//optimRot.cam_pos = camera_pos;
		optimRot.focus_point = pivot;


		//START OPTIMIZATION
		for (int i = 16; i > 0; i--)
		{
			optimTrans.scale_factor = i;
			optimRot.scale_factor = i;

			//optimTrans.AxisX = context.Feedback().GetCamera().GetXAxis();
			//optimTrans.AxisY = context.Feedback().GetCamera().GetYAxis();
			//optimTrans.AxisZ = context.Feedback().GetCamera().GetZAxis();

			transVecNorm = transVec.clone();
			cam.normalizeTranslation(transVecNorm, object.shapeSize);
			optimTrans.rvec = rotVec;
			optimTrans.translation_start = transVecNorm;
			optimTrans.Optimize();
			transVec = optimTrans.translation;

			cam.updateView(context, f, transVec, rotVec, h);

			optimRot.axisX = context.Feedback().GetCamera().GetXAxis();
			optimRot.axisY = context.Feedback().GetCamera().GetYAxis();
			optimRot.axisZ = context.Feedback().GetCamera().GetZAxis();
			optimRot.viewTrans = context.Feedback().GetCamera().GetViewTransform();
			optimRot.Optimize();
			cam.calculateRvec(optimRot.view);
			rotVec = cam.rvec.clone();
			cam.calculateTvec(optimRot.view);
			transVec = cam.tvec.clone();
			//optimRot.translation_start = transVec;
			//optimRot.rotation_start = rot;
			////optimRot.rvec = rotVec;
			//optimRot.tvec = transVec;
			//optimRot.Optimize();
			////rotVec = optimRot.rvec;
			//rot = optimRot.rotation;
			//cam.eulerAnglesToRvec(rot, rotVec);
			//transVec = optimRot.tvec;

			camera_position = context.Feedback().GetCamera().GetPosition();
			cv::Mat camera_pos = (cv::Mat_<float>(3, 1) << camera_position(0), camera_position(1), camera_position(2));

			cam.updateView(context, f, transVec, rotVec, h);
		}

		optimTrans.scale_factor = 1;
		optimTrans.scale_factor = 1;
		//for (int i = 3; i > 0; i--)
		//{


		//transVecNorm = transVec.clone();
		//cam.normalizeTranslation(transVecNorm, object.shapeSize);
		//optimTrans.rvec = rotVec;
		//optimTrans.translation_start = transVecNorm;
		//optimTrans.Optimize();
		//transVec = optimTrans.translation;

		//optimRot.translation_start = transVec;
		//optimRot.rotation_start = rot;
		////optimRot.rvec = rotVec;
		//optimRot.tvec = transVec;
		//optimRot.Optimize();
		////rotVec = optimRot.rvec;
		//rot = optimRot.rotation;
		//cam.eulerAnglesToRvec(rot, rotVec);
		//transVec = optimRot.tvec;

		//camera_position = context.Feedback().GetCamera().GetPosition();
		//cv::Mat camera_pos = (cv::Mat_<float>(3, 1) << camera_position(0), camera_position(1), camera_position(2));

		//camera.updateView(context, f, transVec, rotVec, h);
		//}

		
		//UPDATE VIEW

		//cam.eulerAnglesToRvec(rot, rotVec);
		//f = camera_matrix.at<float>(1, 1);
		//camera.updateView(context, f, transVec, rotVec, h);

		imgCam.release();
		silhuette2D.release();
		transVec.release();
		//focus_point.release();
		rotVec.release();
		transVecNorm.release();
		//camera_pos.release();
	}



};

OGX_EXPORT_METHOD(ImageToGeometryRegistration)