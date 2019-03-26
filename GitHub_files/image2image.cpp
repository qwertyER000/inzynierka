#include<cstdlib>
#include<vector>
#include<cstring>
#include<opencv2/opencv.hpp>

cv::Mat detected_edges;

std::string name_window = "Canny contours";

void unsharpMask(cv::Mat& input_img, cv::Mat& output_img)
{
	//unsharp mask
	cv::Mat gaussian, gaussian_inv, sub_img_blur, add_img_inv, tmp;
	cv::GaussianBlur(input_img, gaussian, cv::Size(5, 5), 1, 0);
	cv::subtract(input_img, gaussian, sub_img_blur);
	cv::bitwise_not(gaussian, gaussian_inv);
	cv::add(input_img, gaussian_inv, add_img_inv);
	cv::bitwise_not(add_img_inv, add_img_inv);
	cv::add(input_img, sub_img_blur, tmp);
	cv::subtract(tmp, add_img_inv, output_img);
}



cv::Mat removeHoles(cv::Mat& img_in)
{
	cv::Mat img_out = cv::Mat::zeros(cv::Size(img_in.cols, img_in.rows), img_in.type());

	bool are_holes = true;

	//detect holes
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(img_in, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	int hole_counter=0;
	cv::Vec4i current_hierarchy;
	int n=1;

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

	cv::threshold(img_in, img_out, 40, 255, cv::THRESH_BINARY);
}



int main()
{

	cv::Mat input_frames, input_camera, imgFrames, imgCamera, grayCamera;


	input_frames = cv::imread("C:\\Users\\Ewa\\Desktop\\20181108_dane_do_inz_ERycaj\\render_back.png");

	input_camera = cv::imread("C:\\Users\\Ewa\\Desktop\\20181108_dane_do_inz_ERycaj\\black.png");
	int height = input_frames.rows;
	int width = input_camera.cols*height/input_camera.rows;

	cv::resize(input_camera, imgCamera, cv::Size(width, height));

	cv::Rect ROI(input_frames.cols / 2 - width /2, 0, width, height);
	imgFrames = input_frames(ROI);


	
	//cv::Mat thresFrames = cv::Mat::zeros(cv::Size(imgFrames.cols, imgFrames.rows), CV_32FC1);
	cv::cvtColor(imgFrames, imgFrames, CV_BGR2GRAY);
	//cv::threshold(imgFrames, thresFrames, 20, 255, cv::THRESH_BINARY);
	cv::Mat silhouetteFrames, silhouetteCamera;

	// get silhouettes
	silhouetteFrames = removeHoles(imgFrames);
	silhouetteFromImage(imgCamera, silhouetteCamera);

	//convert to 32 bit
	silhouetteFrames.convertTo(silhouetteFrames, CV_32FC1);
	silhouetteCamera.convertTo(silhouetteCamera, CV_32FC1);

	for (int i = 0; i < silhouetteCamera.rows; i++)
	{
		for (int j = 0; j < silhouetteCamera.cols; j++)
		{
			silhouetteCamera.at<float>(i, j) = silhouetteCamera.at<float>(i, j) / 255;
			silhouetteFrames.at<float>(i, j) = silhouetteFrames.at<float>(i, j) / 255;
		}
	}




	cv::Mat XOR_result = cv::Mat::zeros(cv::Size(silhouetteFrames.cols, silhouetteCamera.rows), CV_32FC1);
	
	for (int i = 0; i < XOR_result.rows; i++)
	{
		for (int j = 0; j < XOR_result.cols; j++)
		{
			if (!silhouetteCamera.at<float>(i, j) != !silhouetteFrames.at<float>(i, j))
			{
				XOR_result.at<float>(i, j) = 1;
			}
		}
	}

	int similarity = cv::countNonZero(XOR_result);

	//cv::GaussianBlur(silhouetteFrames, silhouetteFrames, cv::Size(41, 41), 0);
	//cv::GaussianBlur(silhouetteCamera, silhouetteCamera, cv::Size(41, 41), 0);

	//cv::Mat diff1 = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);
	//cv::Mat	diff2 = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);
	//cv::Mat sqrt_dist = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);
	//cv::Mat combined = cv::Mat::zeros(cv::Size(width, height), CV_32FC2);

	//cv::absdiff(silhouetteFrames, silhouetteCamera, diff1);
	//cv::absdiff(silhouetteCamera, silhouetteFrames, diff2);
	//std::vector<cv::Mat> channels = { diff1, diff2 };
	//cv::merge(channels, combined);

	//cv::MatND hist;
	//int diff_bins = 32;
	//int hist_Size[] = { diff_bins };
	//int dims[] = { 0, 1 };
	//float diff_ranges[] = { 0, 1 };
	//const float* ranges[] = { diff_ranges, diff_ranges };
	//
	//cv::calcHist(&combined, 1, dims, cv::Mat(), hist, 1, hist_Size, ranges, true, true);


	input_frames.release();
	input_camera.release();
	imgFrames.release();
	imgCamera.release();


	return 0;
}