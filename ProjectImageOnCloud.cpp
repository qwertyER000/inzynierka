#include <ogx/Plugins/EasyPlugin.h>
#include <ogx/Math/Calculation.h>
#include <ogx/Math/Optimization.h>
#include <ogx/Data/Clouds/CloudHelpers.h>
#include <ogx/Data/Clouds/SearchKernels.h>



#include"CameraCV.h"
#include "ColorMapping.h"



struct ProjectImageonCloud : public ogx::Plugin::EasyMethod
{


	// INPUT PARAMS
	Data::ResourceID node_id;
	String img_path;
	bool series_of_photos;
	//int h;



	// constructor
	ProjectImageonCloud() : EasyMethod(L"Ewa Rycaj", L"Project image on clouds")
	{
	}

	// add input/output parameters
	virtual void DefineParameters(ParameterBank& bank)
	{
		//bank.Add(L"node_id", image_id).AsNode();
		bank.Add(L"series of photos", series_of_photos = true);
		bank.Add(L"node id", node_id).AsNode();
		bank.Add(L"image path", img_path).AsFile();
		//bank.Add(L"height", h = 600, L"Choose output image vertical resolution. If height=-1 resolution is set as input. ");
	}

	//load texture image
	virtual void LoadImageCV(String path, cv::Mat& camera_img)
	{
		std::string name(path.begin(), path.end());

		camera_img = cv::imread(name);
		if (camera_img.empty()) ReportError(L"image not found");

		cv::resize(camera_img, camera_img, cv::Size(camera_img.cols/2, camera_img.rows / 2));
		
		if (camera_img.channels() == 4)
			cv::cvtColor(camera_img, camera_img, CV_BGRA2BGR);

		if (camera_img.depth() != CV_8U || camera_img.channels() != 3) ReportError(L"Load RGB image");

		//if (img_height == -1)
		//{
		//	img_height = input_img.rows;
		//	input_img.copyTo(camera_img);
		//}
		//else
		//{
		//	int img_width = input_img.cols*img_height / input_img.rows;
		//	cv::resize(input_img, camera_img, cv::Size(img_width, img_height));
		//}

	}
	
	virtual void Run(Context& context)
	{
		renderSilhouette render3d;
		CVCamera cam;
		int width, height;



		auto subtree = context.Project().TransTreeFindNode(node_id);
		// report error if give node was not found, this will stop execution of algorithm
		if (!subtree) ReportError(L"Node not found");

		//TEXTURE IMG
		cv::Mat imgCam;
		LoadImageCV(img_path, imgCam);
		height = imgCam.rows;
		width = imgCam.cols;

		//CAMERA
		cv::Mat transVec, transVecNorm, rotVec;
		int width_from_frames = height * context.Feedback().GetCamera().GetProjection().GetAspect();
		CVCamera camera;
		cam.calculateCameraMatrix(context.Feedback().GetCamera().GetProjection().GetMatrix(), width_from_frames, height); //camera matrix
		cam.calculateTvec(context.Feedback().GetCamera().GetViewTransform()); //translation vector
		cam.calculateRvec(context.Feedback().GetCamera().GetViewTransform()); //rotation vector
		auto camera_position = context.Feedback().GetCamera().GetPosition(); //get current camera position
		int f = cam.camera_matrix.at<float>(1, 1);


			   		 
		//3D MODEL TO CV
		// perform calculation for each cloud
			// run with number of threads available on current machine, optional

		std::vector<double> distance;
		auto const thread_count = std::thread::hardware_concurrency();

		Data::Clouds::PointsRange points;
		std::vector<double> angle_cos;

		ogx::Data::Clouds::ForEachCloud(*subtree, [&](Data::Clouds::ICloud & cloud, Data::Nodes::ITransTreeNode & node)
		{
			cloud.GetAccess().GetAllPoints(points); //get all points
			render3d.pointsToCV_front(camera_position, points, distance, angle_cos);
		}, thread_count);

		cv::Mat renderView = cv::Mat::zeros(cv::Size (width_from_frames, height), CV_8UC1);
		cv::Mat zBuffer = cv::Mat::zeros(renderView.size(), CV_32FC1);
		std::vector<cv::Point2f> points2D;
		//std::vector<cv::Point3f> points3D = render3d.CloudXYZ;
		cv::Mat rvec = cam.rvec;
		cv::Mat tvec = cam.tvec;
		cv::Mat intrinsic_matrix = cam.camera_matrix;
		cv::projectPoints(render3d.CloudXYZ, rvec, tvec, intrinsic_matrix, cv::Mat(), points2D);
		//cv::projectPoints(render3d.CloudXYZ, cam.rvec, cam.tvec, cam.camera_matrix, cv::Mat(), points2D);

		int i = 0;
		for (auto& ipt : points2D)
		{

			renderView.at<uchar>(ipt.y, ipt.x) = 255;
			if (zBuffer.at<float>(ipt.y, ipt.x) == 0 || (zBuffer.at<float>(ipt.y, ipt.x) > distance[i]))
				zBuffer.at<float>(ipt.y, ipt.x) = distance[i];
			i++;

		}
		cv::Rect ROI(renderView.cols / 2 - width / 2, 0, width, height);
		cv::Mat img3D = renderView(ROI);

		int k = -1;
		int j = 0;
		double cos;
		cv::Point2f tmp;
		cv::Vec3b current_color;

		Data::Clouds::RangeColor pointsRGB(points);

		if (!series_of_photos)
		{
			for (auto& ipt : pointsRGB)
			{
				k++;
				if (j == points2D.size())  break;

				if (angle_cos[k] < 0) continue;
				else
				{
					
					//if (zBuffer.at<float>(points2D[k].y, points2D[k].x) > distance[k])
					//	zBuffer.at<float>(points2D[k].y, points2D[k].x) = 0.5;
					//  continue;
					tmp = { points2D[j].x - (width_from_frames - width) / 2, points2D[j].y};
					
					if (tmp.x >= width || tmp.y >= height || tmp.x < 0 || tmp.y < 0)
						continue;

					ipt.z() = imgCam.at<cv::Vec3b>(tmp.y, tmp.x)[0];
					ipt.y() = imgCam.at<cv::Vec3b>(tmp.y, tmp.x)[1];
					ipt.x() = imgCam.at<cv::Vec3b>(tmp.y, tmp.x)[2];
					j++;

				}
			}
		}
		else
		{
			for (auto& ipt : pointsRGB)
			{
				k++;
				if (j == points2D.size())  break;

				if (angle_cos[k] < 0) continue;
				else
				{

					//if (zBuffer.at<float>(points2D[k].y, points2D[k].x) > distance[k])
					//	zBuffer.at<float>(points2D[k].y, points2D[k].x) = 0.5;
					//  continue;
					tmp = { points2D[j].x - (width_from_frames - width) / 2, points2D[j].y };

					if (tmp.x >= width || tmp.y >= height || tmp.x < 0 || tmp.y < 0)
						continue;
					current_color[0] = ipt.z()*(1 - angle_cos[k]) + angle_cos[k] * imgCam.at<cv::Vec3b>(tmp.y, tmp.x)[0];
					current_color[1] = ipt.y()*(1 - angle_cos[k]) + angle_cos[k] * imgCam.at<cv::Vec3b>(tmp.y, tmp.x)[1];
					current_color[2] = ipt.x()*(1 - angle_cos[k]) + angle_cos[k] * imgCam.at<cv::Vec3b>(tmp.y, tmp.x)[2];


					ipt.z() = (0.4* current_color[0] + 0.6 * imgCam.at<cv::Vec3b>(tmp.y, tmp.x)[0]);
					ipt.y() = (0.4* current_color[1] + 0.6 * imgCam.at<cv::Vec3b>(tmp.y, tmp.x)[1]);
					ipt.x() = (0.4* current_color[2] + 0.6 *  imgCam.at<cv::Vec3b>(tmp.y, tmp.x)[2]);

				}
			}
		}
		
		zBuffer.release();
		imgCam.release();
		img3D.release();

		////cv::Mat zBuffer = cv::Mat::zeros(renderView.size(), CV_32FC1);
		//render3d.GetImageToProjection(cam.rvec, cam.tvec, cam.camera_matrix, render3d.CloudXYZ, distance, renderView, zBuffer);
		//render3d.GetSilhouette(cam.rvec, cam.tvec, cam.camera_matrix, render3d.CloudXYZ, renderView);

		//if (zBuffer.at<float>(ipt.y, ipt.x) == 0 || (z_buffer.at<float>(ipt.y, ipt.x) > distance[k]))
		//	z_buffer.at<float>(ipt.y, ipt.x) = distance[k];
		//k++

		//img_render = cv::Mat::zeros(cv::Size(widthFrames, height), CV_8UC1);
		//z_buffer = cv::Mat::zeros(cv::Size(widthFrames, height), CV_32FC1);



	}

};

OGX_EXPORT_METHOD(ProjectImageonCloud)