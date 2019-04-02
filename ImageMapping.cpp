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
	//// parameters
	Data::ResourceID node_id;
	String img_path;
	int height, width, with_new_fram, m_far;
	cv::Mat silhuette2D;
	std::vector<cv::Point3f> CloudXYZ;
	cv::Mat camera_matrix, translation, rotation;
	cv::Mat solver_rot;

	// constructor
	ImageMapping() : EasyMethod(L"Ewa Rycaj", L"Image to geometry registration")
	{
	}

	// add input/output parameters
	virtual void DefineParameters(ParameterBank& bank)
	{
		//bank.Add(L"node_id", image_id).AsNode();
		bank.Add(L"node id", node_id).AsNode();
		bank.Add(L"image path", img_path).AsFile();
		bank.Add(L"height", height = -1, L"Choose output image vertical resolution. If height=-1 resolution is set as input. ");
	}

	//loads texture image
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


	virtual void Run(Context& context)
	{
		auto subtree = context.Project().TransTreeFindNode(node_id);
		// report error if given node was not found, this will stop execution of algorithm
		if (!subtree) ReportError(L"Node not found");
		//camera_matrix = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
		
		//CAMERA
		with_new_fram = height * context.Feedback().GetCamera().GetProjection().GetAspect();
		CVCamera camera;
		camera_matrix = camera.camera_matrix(context.Feedback().GetCamera().GetProjection().GetMatrix(), with_new_fram, height); //camera matrix
		float f = camera_matrix.at<float>(1, 1);
		translation = camera.tvec(context.Feedback().GetCamera().GetViewTransform()); // translation vector
		rotation = camera.rvec(context.Feedback().GetCamera().GetViewTransform()); //rotation vector
		auto camera_position = context.Feedback().GetCamera().GetPosition(); //get current camera position
		int far_pl = context.Feedback().GetCamera().GetProjection().GetFarClippingPlane();

		//camera.updateView(context, viewMatrix.at<float>(1, 1), tvec, rvec, height);
		
		 //double height_fram = context.Feedback().GetViewportRect().h;


		//TEXTURE IMG
		cv::Mat imgCam;
		LoadImageCV(img_path, imgCam, height);
		silhouetteFromImage(imgCam, silhuette2D);
		width = silhuette2D.cols;

		
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

		auto solver = Math::SolverI::Create();
		solver->Solve(*this);

		//cv::Mat rvec = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1); //rotation vector
		camera.updateView(context, f, translation, rotation, height);
		delete solver;
	}

	virtual Count GetResultCount()
	{
		return 6;
	}

	virtual Count GetParamCount()
	{
		return 6;
	}

	virtual void InitParams(Math::Vector& params)
	{

		//params[0] = camera_matrix.at<float>(1, 1)/m_far*1000;

		params[0] = rotation.at<float>(0, 0);
		params[1] = rotation.at<float>(1, 0);
		params[2] = rotation.at<float>(2, 0);
		params[3] = translation.at<float>(0, 0);
		params[4] = translation.at<float>(1, 0);
		params[5] = translation.at<float>(2, 0);
	}

	virtual bool Iterate(const Math::Vector& params, Math::Vector& results)
	{
		//cv::Mat tvec = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1); // translation vector
		//cv::Mat rvec = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1); //rotation vector
		//cv::Mat matrix = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1); 

		cv::Mat framSilhouette = cv::Mat::zeros(cv::Size(with_new_fram, height), CV_8UC1);
		cv::Mat XOR;

		//camera_matrix.at<float>(0, 0) = -params(0)*m_far/1000;
		//camera_matrix.at<float>(1, 1) = params(0)*m_far/1000;
		rotation.at<float>(0, 0) = params[0];
		rotation.at<float>(1, 0) = params[1];
		rotation.at<float>(2, 0) = params[2];
		translation.at<float>(0, 0) = params[3];
		translation.at<float>(1, 0) = params[4];
		translation.at<float>(2, 0) = params[5];
		
		//cv::Mat rvec = rotation;

		renderSilhouette render3d;
		render3d.GetSilhouette(rotation, translation, camera_matrix, CloudXYZ, framSilhouette);
		cv::Rect ROI(framSilhouette.cols / 2 - width / 2, 0, width, height);
		cv::Mat img3D = framSilhouette(ROI);
		cv::Mat silhuette3D = render3d.RemoveHoles(img3D);

		render3d.similarityImg(silhuette3D, silhuette2D, XOR);

		double pixels_all = XOR.rows*XOR.cols;
		for (int i = 0; i < 6; i++)
		{
			results(i) = (double)cv::countNonZero(XOR) / pixels_all;
		}

		//tvec.release();
		rvec.release();
		img3D.release();
		silhuette3D.release();
		XOR.release();

		return true;
	}

	virtual void SaveSolution(const Math::Vector& params)
	{
		OGX_LINE.Format(ogx::Debug, L"Current solver solution is: %f, %f, %f, %f, %f, %f", params(0));
		//OGX_LINE.Format(ogx::Debug, L"Current solver solution is: %f, %f, %f, %f, %f, %f", params[0], params[1], params[2], params[3], params[4], params[5], params[6]);

	}

};

OGX_EXPORT_METHOD(ImageMapping)
