#include <ogx/Plugins/EasyPlugin.h>
#include <string>
#include <comdef.h> 
#include <ogx/Data/Clouds/CloudHelpers.h>
#include <ogx/Data/Clouds/KNNSearchKernel.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace ogx;
using namespace ogx::Data;

struct Render : public ogx::Plugin::EasyMethod
{
	//// parameters
	Data::ResourceID node_id;
	String img_path;
	int height;

	// constructor
	Render() : EasyMethod(L"Ewa Rycaj", L"cos.")
	{
	}

	// add input/output parameters
	virtual void DefineParameters(ParameterBank& bank)
	{
		//bank.Add(L"node_id", image_id).AsNode();
		bank.Add(L"node id", node_id).AsNode();
		bank.Add(L"image path", img_path).AsFile();
		bank.Add(L"width", height = -1, L"Choose output image vertical resolution. If height=-1 resolution is set as input. ");
	}

	//converts IImage2CV - NOT USED
	virtual void IImage2CV(Images::IImage& p_image, cv::Mat& img)
	{
		int height = p_image.GetHeight();
		int width = p_image.GetWidth();


		auto pixels = p_image.GetPixels();


		int channels = p_image.GetPixelType();
		std::vector<Real> pixels_out;

		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				for (int ch = 0; ch < channels; ++ch)
				{
					Real tmp = pixels(j, i*channels + ch);
					pixels_out.push_back(tmp);
				}
			}
		}

		int k = 0;
		if (channels == 4)
		{
			img = cv::Mat::zeros(cv::Size(width, height), CV_32FC4);
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{

					img.at<cv::Vec4f>(j, i)[2] = pixels_out[k++];
					img.at<cv::Vec4f>(j, i)[1] = pixels_out[k++];
					img.at<cv::Vec4f>(j, i)[0] = pixels_out[k++];
					img.at<cv::Vec4f>(j, i)[3] = pixels_out[k++];
				}
			}
		}
		else if (channels == 3)
		{
			img = cv::Mat::zeros(cv::Size(width, height), CV_32FC4);
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					img.at<cv::Vec3f>(j, i)[2] = pixels_out[k++];
					img.at<cv::Vec3f>(j, i)[1] = pixels_out[k++];
					img.at<cv::Vec3f>(j, i)[0] = pixels_out[k++];
				}
			}
		}
		else
		{
			ReportError(L"Wrong image type");
		}
	}

	//silhouette from texture camera image
	void silhouetteFromImage(cv::Mat &img_in, cv::Mat &img_out)
	{
		if (img_in.channels() == 3)
		{
			cv::cvtColor(img_in, img_in, CV_BGR2GRAY);
		}

		if (img_in.channels() == 4)
		{
			cv::cvtColor(img_in, img_in, CV_BGRA2GRAY);
		}

		cv::threshold(img_in, img_out, 40, 255, cv::THRESH_BINARY);
	}

	//silhouette from 3d model
	cv::Mat removeHoles(cv::Mat& img_in)
	{
		cv::Mat img_out = cv::Mat::zeros(cv::Size(img_in.cols, img_in.rows), img_in.type());

		bool are_holes = true;

		//detect holes
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(img_in, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
		int hole_counter = 0;
		cv::Vec4i current_hierarchy;
		int n = 1;

		for (int i = 0; i < contours.size(); i++)
		{
			current_hierarchy = hierarchy[i];
			if (current_hierarchy[2] == -1 && std::fabs(cv::contourArea(contours[i])) < 20)
			{
				hole_counter++;

			}
		}

		if (hole_counter < 8)
		{
			are_holes = false;
			img_out = img_in.clone();
			return img_out;
		}
		else
		{

			cv::Mat blured;
			int madian_size = 2 * n + 1;
			cv::medianBlur(img_in, blured, madian_size);

			// get structuring element
			cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

			//perform closing operation
			cv::dilate(blured, img_out, element);
			cv::erode(img_out, img_out, element);

			n++;
			return removeHoles(img_out);
		}
	}

	
	//Camera from OpenGL to OpenCV
	virtual void CameraToCV(Context& context, cv::Mat& perCam_cv, cv::Mat &tvec, cv::Mat &rvec, int height_Viewport)
	{
		
		auto camera_matrix = context.Feedback().GetCamera().GetProjection().GetMatrix();
		//INTRINSIC CAMERA MATRIX
		//calculate parameters
		auto width_Viewport = height_Viewport * context.Feedback().GetCamera().GetProjection().GetAspect();
		double fx = -0.5*width_Viewport*camera_matrix(0, 0);
		double fy = -fx; //0.5*height_Viewport*camera_matrix(1, 1);
		double cx = (1 - camera_matrix(0, 2)) / 2 * width_Viewport;
		double cy = (camera_matrix(1, 2) + 1)*0.5*height_Viewport;
		//create matrix
		perCam_cv.at<float>(0, 0) = fx; 
		perCam_cv.at<float>(1, 1) = fy;
		perCam_cv.at<float>(0, 2) = cx;
		perCam_cv.at<float>(1, 2) = cy;
		perCam_cv.at<float>(2, 2) = 1;
		auto view_matrix = context.Feedback().GetCamera().GetViewTransform();

		//ROTATION
		cv::Mat R_matrix = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				R_matrix.at<float>(j, i) = view_matrix(j, i);
			}
		}
		cv::Rodrigues(R_matrix, rvec);

		//TRANSLATION
		tvec = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1);
		tvec.at<float>(0, 0) = view_matrix(0, 3);
		tvec.at<float>(1, 0) = view_matrix(1, 3);
		tvec.at<float>(2, 0) = view_matrix(2, 3);
	}


	//Cloud points from Frames to OpenCV
	virtual void FramesPointToCV(Data::Nodes::ITransTreeNode* subtree ,Math::Point3D& camera_pos, std::vector<cv::Point3f>& pointsCoord, std::vector<float>& pointsDist)
	{
		//perform calculation for each cloud
			// run with number of threads available on current machine, optional
		auto const thread_count = std::thread::hardware_concurrency();

		Clouds::ForEachCloud(*subtree, [&](Data::Clouds::ICloud & cloud, Data::Nodes::ITransTreeNode & node)
		{
			Data::Clouds::PointsRange points_all;
			cloud.GetAccess().GetAllPoints(points_all); //get all points
			cv::Point3f tmp_coord;
			float tmp_dist = 0;

			for (auto& point : Data::Clouds::RangePointConst(points_all))
			{
				//get coordinates of current object point
				tmp_coord.x = point->GetXYZ()[0];
				tmp_coord.y = point->GetXYZ()[1];
				tmp_coord.z = point->GetXYZ()[2];

				//dot product of two vectors - to find if they point in the same direction
				if ((camera_pos - point->GetXYZ().cast<Real>()).normalized().dot(point->GetNormal().cast<Real>()) < 0)
				{
					continue;
				}
				//calculate distance between camera and points
				std::vector<double> camVec = { camera_pos[0] - tmp_coord.x, camera_pos[1] - tmp_coord.y, camera_pos[2] - tmp_coord.z };
				tmp_dist = sqrt(camVec[0] * camVec[0] + camVec[1] * camVec[1] + camVec[2] * camVec[2]);

				pointsCoord.push_back(tmp_coord);
				pointsDist.push_back(tmp_dist);
			}
		}, thread_count); // run with given number of threads, optional parameter, if not given will run in current thread
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


	virtual void Run(Context& context)
	{

		auto subtree = context.Project().TransTreeFindNode(node_id);
		// report error if given node was not found, this will stop execution of algorithm
		if (!subtree) ReportError(L"Node not found");

		//3D MODEL PARAMETERS
		std::vector<cv::Point3f> vecCoord; //3D points coordinates
		std::vector<float>vecDist; //distance between object point and camera
		
		//CAMERA PARAMETERS
		cv::Mat intrinsic_matrix = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1); //camera matrix
		cv::Mat tvec; // translation vector
		cv::Mat rvec; //rotation vector
		auto camera_position = context.Feedback().GetCamera().GetPosition(); //get current camera position

		//IMAGE PARAMETERS
		std::vector<cv::Point2f> vecImgPoints;
		int width, widthFrames = height * context.Feedback().GetCamera().GetProjection().GetAspect(); //widt-based on camera img width, widthFrames - based on Frames window's aspect
		cv::Mat imgCamera, imgFrames, img_render, z_buffer, silhouetteFrames, silhouetteCamera, XOR_result;

		LoadImageCV(img_path, imgCamera, height);
		width = imgCamera.cols;

		CameraToCV(context, intrinsic_matrix, tvec, rvec, height);

		FramesPointToCV(subtree, camera_position, vecCoord, vecDist);
				
		cv::projectPoints(vecCoord, rvec, tvec, intrinsic_matrix, cv::Mat(), vecImgPoints);//OpenCV projectpoints from 3D to 2D
		
		img_render = cv::Mat::zeros(cv::Size(widthFrames, height), CV_8UC1);
		z_buffer = cv::Mat::zeros(cv::Size(widthFrames, height), CV_32FC1);
		
		int k = 0;
		for (auto& ipt : vecImgPoints)
		{


			if (ipt.x >= widthFrames || ipt.y >= height || ipt.x <0 || ipt.y <0)
			{
				continue;
			}

			if (z_buffer.at<float>(ipt.y, ipt.x) == 0 || (z_buffer.at<float>(ipt.y, ipt.x) > vecDist[k]))
			{
				z_buffer.at<float>(ipt.y, ipt.x) = vecDist[k];
				img_render.at<unsigned char>(ipt.y, ipt.x) = 255;
			}

			k++;
		}

		cv::Rect ROI(img_render.cols / 2 - width / 2, 0, width, height);
		imgFrames = img_render(ROI);
		
		//silhouettes
		silhouetteFrames = removeHoles(imgFrames);
		silhouetteFromImage(imgCamera, silhouetteCamera);

		//similarity measure
		XOR_result = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
		for (int i = 0; i < XOR_result.rows; i++)
		{
			for (int j = 0; j < XOR_result.cols; j++)
			{
				if (!silhouetteCamera.at<unsigned char>(i, j) != !silhouetteFrames.at<unsigned char>(i, j))
				{
					XOR_result.at<unsigned char>(i, j) = 255;
				}
			}
		}

		int similarity = cv::countNonZero(XOR_result);

		imgCamera.release();
		imgFrames.release();
		silhouetteFrames.release();
		silhouetteCamera.release();
		XOR_result.release();
		z_buffer.release();
		img_render.release();

	}
};


/*
trash

// fill translation vector:
Math::Vector3D camera_translate = Math::Vector3D::Zero();
for (auto i = 0; i < 3; ++i) camera_translate(i) = cam_center[i];

// fill rotation matrix:
Math::Matrix3 eigen_rot = Math::Matrix3::Identity();
for (auto i = 0; i < 9; ++i) eigen_rot(i) = cam_rot[i];

auto node_transform = img_node->GetLocalTransformation();
node_transform.rotate(eigen_rot);
node_transform.pretranslate(camera_translate);
img_node->SetLocalTransformation(node_transform);


// cameras visualization:
// v_focal ~ 50mm
// vis_width ~ 36mm = v_focal * 0.72
// vis_height ~ 24mm = width * 0.67
double width = v_focal * 0.72;
double height = width * 0.67;

// load image data:
if (m_load_image_data && m_reduce_size_factor >= 1)
{
	auto img_vis = img_node->CreateChild();
	img_vis->Rename(v_img_name + L"_vis");

	img_vis->SetElement(LaunchImageImport(context, v_pathwin.Location() + v_img_name));
	auto img = img_vis->GetElement()->GetData<Data::Images::IImage>();

	img->SetDPI(v_focal * 25.4);
	//img->SetDPI(v_focal * 25.4 / m_reduce_size_factor);
	width = Real(img->GetWidth());
	height = Real(img->GetHeight());

	auto node_transform = img_vis->GetLocalTransformation();
	node_transform.translate(Math::Point3D(-width / 2.0 / v_focal, -height / 2.0 / v_focal, 1.0));
img_vis->SetLocalTransformation(node_transform);
}*/

OGX_EXPORT_METHOD(Render)