#include <ogx/Plugins/EasyPlugin.h>
#include <ogx/Data/Clouds/CloudHelpers.h>
#include"CameraCV.h"
#include "ColorMapping.h"


using namespace ogx;
using namespace ogx::Data;

struct ImageMapping : public ogx::Plugin::EasyMethod
{
	//// parameters
	Data::ResourceID node_id;
	String img_path;
	int height, width, with_new_fram;
	cv::Mat silhuette2D, silhuette3D;
	std::vector<cv::Point3f> CloudXYZ;
	Context n_context;

	// constructor
	ImageMapping() : EasyMethod(L"Ewa Rycaj", L"cos.")
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
	void silhouetteFromImage(cv::Mat &img_in, cv::Mat &img_out)
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

	double estimateCamera(std::vector<float> params)
	{
		//cv::Mat intrinsic_matrix = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1); //camera matrix
		cv::Mat tvec = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1); // translation vector
		cv::Mat rvec = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1); //rotation vector
		cv::Mat framSilhouette = cv::Mat::zeros(cv::Size(with_new_fram, height), CV_8UC1);
		cv::Mat XOR;


		rvec.at<float>(0, 0) = params[1];
		rvec.at<float>(1, 0) = params[2];
		rvec.at<float>(2, 0) = params[3];
		tvec.at<float>(0, 0) = params[4];
		tvec.at<float>(1, 0) = params[5];
		tvec.at<float>(2, 0) = params[6];

		CVCamera cam;
		cam.updateView(n_context, params[0], tvec, rvec, height);
		auto matrix = cam.camera_matrix(n_context.Feedback().GetCamera().GetProjection().GetMatrix(), with_new_fram, height);
		matrix.at<float>(0, 0) = params[0];
		matrix.at<float>(1, 1) = params[0];

		renderSilhouette render3d;
		render3d.GetSilhouette(rvec, tvec, matrix, CloudXYZ, framSilhouette);
		cv::Rect ROI(framSilhouette.cols / 2 - width / 2, 0, width, height);
		cv::Mat img3D = framSilhouette(ROI);
		silhuette3D = render3d.RemoveHoles(img3D);

		render3d.similarityImg(silhuette3D, silhuette2D, XOR);

		float misalignment = cv::countNonZero(XOR)/(XOR.rows*XOR.cols);

		return misalignment;
	}


	virtual void Run(Context& context)
	{
		auto subtree = context.Project().TransTreeFindNode(node_id);
		// report error if given node was not found, this will stop execution of algorithm
		if (!subtree) ReportError(L"Node not found");
		
		//CAMERA
		with_new_fram = height * context.Feedback().GetCamera().GetProjection().GetAspect();
		CVCamera camera;
		cv::Mat viewMatrix = camera.camera_matrix(context.Feedback().GetCamera().GetProjection().GetMatrix(), with_new_fram, height); //camera matrix
		cv::Mat tvec = camera.tvec(context.Feedback().GetCamera().GetViewTransform()); // translation vector
		cv::Mat rvec = camera.rvec(context.Feedback().GetCamera().GetViewTransform()); //rotation vector
		auto camera_position = context.Feedback().GetCamera().GetPosition(); //get current camera position
		
																			 //double height_fram = context.Feedback().GetViewportRect().h;
		//OPTIMIZATION PARAMS
		std::vector<float> parameters = { 0, 0, 0, 0, 0, 0, 0 };
		parameters[0] = viewMatrix.at<float>(1, 1); //focal length
		//rotation
		parameters[1] = rvec.at<float>(0, 0);
		parameters[2] = rvec.at<float>(1, 0);
		parameters[3] = rvec.at<float>(2, 0);
		//translation
		parameters[4] = tvec.at<float>(0, 0);
		parameters[5] = tvec.at<float>(1, 0);
		parameters[6] = tvec.at<float>(2, 0);

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
		
		estimateCamera(parameters);
	}


};

OGX_EXPORT_METHOD(ImageMapping)