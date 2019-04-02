#include "ColorMapping.h"

using namespace ogx;

renderSilhouette::renderSilhouette()
{
}


renderSilhouette::~renderSilhouette()
{
}

void renderSilhouette::pointsToCV_all(ogx::Data::Clouds::PointsRange& points_all, std::vector<cv::Point3f>& pointsCoord)
{
		cv::Point3f tmp_coord;
		//float tmp_dist = 0;
		for (auto& point : Data::Clouds::RangeLocalXYZ(points_all))
		{
			//get coordinates of current object point
			tmp_coord.x = point[0];
			tmp_coord.y = point[1];
			tmp_coord.z = point[2];
			pointsCoord.push_back(tmp_coord);
		}
}
void renderSilhouette::pointsToCV_front(Math::Point3D& camera_pos, Data::Clouds::PointsRange points_all, std::vector<cv::Point3f>& pointsCoord, std::vector<double>& pointsDist)
{
	cv::Point3f tmp_coord;
	double tmp_dist = 0;

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
}

void renderSilhouette::similarityImg(cv::Mat& img_in1, cv::Mat& img_in2, cv::Mat& img_out)
{
	img_out = cv::Mat::zeros(cv::Size(img_in1.cols, img_in1.rows), CV_8UC1);
	for (int i = 0; i < img_out.rows; i++)
	{
		for (int j = 0; j < img_out.cols; j++)
		{
			if (!img_in1.at<unsigned char>(i, j) != !img_in2.at<unsigned char>(i, j))
			{
				img_out.at<unsigned char>(i, j) = 255;
			}
		}
	}
}

void renderSilhouette::GetSilhouette(cv::Mat rvec, cv::Mat tvec, cv::Mat intrinsic_matrix, std::vector<cv::Point3f>& points3D, cv::Mat& silhouette)
{
	std::vector<cv::Point2f> points2D;


		cv::projectPoints(points3D, rvec, tvec, intrinsic_matrix, cv::Mat(), points2D);

		int k = 0;
		for (auto& ipt : points2D)
		{


			if (ipt.x >= silhouette.cols || ipt.y >= silhouette.rows || ipt.x < 0 || ipt.y < 0)
			{
				continue;
			}
			silhouette.at<unsigned char>(ipt.y, ipt.x) = 255;

			k++;
		}

		points2D.clear();
	
}

cv::Mat renderSilhouette::RemoveHoles(cv::Mat& img_in)
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

	if (hole_counter < 8 || n > 7)
	{
		are_holes = false;
		img_out = img_in.clone();
		return img_out;
	}
	else
	{

		// get structuring element
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * n + 1, 2 * n + 1));

		//perform closing operation
		cv::dilate(img_in, img_out, element);
		cv::erode(img_out, img_out, element);

		n++;
		return RemoveHoles(img_out);
	}
}