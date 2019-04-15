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
	cv::Vec3f rotation;
	cv::Vec3f rotation_start;
	cv::Mat tvec;
	cv::Mat camera_matrix;
	CVCamera camera;
	int height, width, width_new_fram;
	cv::Point3f shapeSize;
	std::vector<cv::Point3f> points3D;

	virtual void Optimize()
	{
		if (scale_factor != 1)
		{
			count = height * width / (2 * scale_factor);
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


		rotation[0] = rotation_start[0] + scale_factor * 20000 * params[0];
		rotation[1] = rotation_start[1] + scale_factor * 20000 * params[1];
		rotation[2] = rotation_start[2] + scale_factor * 20000 * params[2];

		cv::Mat rvec;
		camera.eulerAnglesToRvec(rotation, rvec);
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
		if (scale_factor != 1)
			cv::resize(diff, diff, cv::Size(width / scale_factor, height / scale_factor), 0, 0, CV_INTER_AREA);

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
	cv::Mat rvec;
	cv::Mat camera_matrix;
	CVCamera camera;
	int height, width, width_new_fram;
	cv::Point3f shapeSize;
	std::vector<cv::Point3f> points3D;

	virtual void Optimize()
	{
		if (scale_factor != 1)
		{
			count = height * width / (2 * scale_factor);
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


		translation.at<float>(0, 0) = (translation_start.at<float>(0, 0) + scale_factor * 100000 * (float)params[0]);
		translation.at<float>(1, 0) = (translation_start.at<float>(1, 0) + scale_factor * 100000 * (float)params[1]);
		translation.at<float>(2, 0) = (translation_start.at<float>(2, 0) + scale_factor * 100000 * (float)params[2]);

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
		if (scale_factor != 1)
			cv::resize(diff, diff, cv::Size(width / scale_factor, height / scale_factor), 0, 0, CV_INTER_AREA);

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

struct ImageMapping : public ogx::Plugin::EasyMethod, OptimizeRotation, OptimizeTranslation
{

	// INPUT PARAMS
	Data::ResourceID node_id;
	String img_path;
	int h;

	//SOLVER PARAMS
	float f;

	//size_t count = 0;
	//cv::Mat start_translation, translation = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1);
	renderSilhouette render3d;
	CVCamera cam;
	int w, width_from_frames;

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
		bank.Add(L"height", h = 600, L"Choose output image vertical resolution. If height=-1 resolution is set as input. ");
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


	virtual void optimizeFOV()
	{

	}

	virtual void Run(Context& context)
	{

		auto pivot = context.Feedback().GetFocusPoint();


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
		CVCamera camera;
		cam.calculateCameraMatrix(context.Feedback().GetCamera().GetProjection().GetMatrix(), width_from_frames, h); //camera matrix
		cam.calculateTvec(context.Feedback().GetCamera().GetViewTransform()); //translation vector
		cam.calculateRvec(context.Feedback().GetCamera().GetViewTransform()); //rotation vector
		auto camera_position = context.Feedback().GetCamera().GetPosition(); //get current camera position
		f = cam.camera_matrix.at<float>(1, 1);




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


		transVecNorm = cam.tvec.clone();
		transVec = cam.tvec.clone();
		rotVec = cam.rvec.clone();

		camera.normalizeTranslation(transVecNorm, object.shapeSize);
		cam.rvecToEulerAngles(rotVec, rot);


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
		//optimRot.rotation_start = rot;
		optimRot.tvec = transVec;
		optimRot.camera_matrix = cam.camera_matrix;
		optimRot.width_new_fram = width_from_frames;
		optimRot.height = h;
		optimRot.width = w;
		optimRot.input2D = silhuette2D;
		optimRot.points3D = object.CloudXYZ;
		optimRot.shapeSize = object.shapeSize;


		//START OPTIMIZATION
		for (int i = 16; i >= 1; i = i / 2)
		{
			optimTrans.scale_factor = i;
			optimRot.scale_factor = i;

			optimRot.rotation_start = rot;
			optimRot.tvec = transVec;
			optimRot.Optimize();
			rot = optimRot.rotation;
			cam.eulerAnglesToRvec(rot, rotVec);

			transVecNorm = transVec.clone();
			cam.normalizeTranslation(transVecNorm, object.shapeSize);
			optimTrans.rvec = rotVec;
			optimTrans.translation_start = transVecNorm;
			optimTrans.Optimize();
			transVec = optimTrans.translation;


		}

		optimTrans.scale_factor = 1;
		optimTrans.scale_factor = 1;
		for (int i = 3; i > 0; i--)
		{
			transVecNorm = transVec.clone();
			cam.normalizeTranslation(transVecNorm, object.shapeSize);
			optimTrans.rvec = rotVec;
			optimTrans.translation_start = transVecNorm;
			optimTrans.Optimize();
			transVec = optimTrans.translation;

			optimRot.rotation_start = rot;
			optimRot.tvec = transVec;
			optimRot.Optimize();
			rot = optimRot.rotation;
			cam.eulerAnglesToRvec(rot, rotVec);
		}

		//UPDATE VIEW

		//cam.eulerAnglesToRvec(rot, rotVec);
		//f = camera_matrix.at<float>(1, 1);
		camera.updateView(context, f, transVec, rotVec, h);
	}



};

OGX_EXPORT_METHOD(ImageMapping)