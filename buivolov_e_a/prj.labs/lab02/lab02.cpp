#include <opencv2/opencv.hpp>
#include <iostream>

int main() {

	cv::Mat img = cv::imread("D:/SandBox/buivolov_e_a/data/cross_0256x0256.png",1);
	cv::imwrite("badquality.jpg",img, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY,25});
	cv::imshow("badquality.jpg", img);
	cv::Mat img_25q = cv::imread("D:/SandBox/buivolov_e_a/build.vs.2019/prj.labs/lab02/badquality.jpg");
	//cv::imshow("badquality.jpg",img_25q);

	cv::Mat a_channels[3];
	cv::Mat b_channels[3];

	cv::split(img,a_channels);
	cv::split(img_25q, b_channels);

	cv::Mat blue_channel = a_channels[0];
	cv::Mat green_channel = a_channels[1];
	cv::Mat red_channel = a_channels[2];
	cv::Mat blue_channel_jpg = b_channels[0];
	cv::Mat green_channel_jpg = b_channels[1];
	cv::Mat red_channel_jpg = b_channels[2];

	cv::Mat mod_channel_b = a_channels[0].clone();
	cv::Mat mod_channel_g = a_channels[1].clone();
	cv::Mat mod_channel_r = a_channels[2].clone();
	cv::Mat mod_channel_b_jpg = a_channels[0].clone();
	cv::Mat mod_channel_g_jpg = a_channels[1].clone();
	cv::Mat mod_channel_r_jpg = a_channels[2].clone();

	int hist_size = 256;

	float range[] = { 0,256 };
	const float* hist_range[] = { range };

	bool uniform = true, accumulate = false;

	int const hist_height = 256;

	cv::Mat _b_g_hist = cv::Mat::zeros(img.rows, img.cols, img.type());
	cv::Mat _b_b_hist = cv::Mat::zeros(img.rows, img.cols, img.type());
	cv::Mat _b_r_hist = cv::Mat::zeros(img.rows, img.cols, img.type());
	cv::Mat full_hist = cv::Mat::zeros(img.rows, img.cols,CV_8UC1);

	cv::Mat hist_g_img = cv::Mat::zeros(img.rows, img.cols, img.type());
	cv::Mat hist_r_img = cv::Mat::zeros(img.rows, img.cols, img.type());
	cv::Mat hist_b_img = cv::Mat::zeros(img.rows, img.cols, img.type());
	cv::Mat hist_img = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

	cv::Mat _b_g_hist_jpg = cv::Mat::zeros(img_25q.rows, img_25q.cols, img.type());
	cv::Mat _b_b_hist_jpg = cv::Mat::zeros(img_25q.rows, img_25q.cols, img.type());
	cv::Mat _b_r_hist_jpg = cv::Mat::zeros(img_25q.rows, img_25q.cols, img.type());
	cv::Mat full_hist_jpg = cv::Mat::zeros(img_25q.rows, img_25q.cols, CV_8UC1);

	cv::Mat hist_g_img_jpg = cv::Mat::zeros(img_25q.rows, img_25q.cols, img.type());
	cv::Mat hist_r_img_jpg = cv::Mat::zeros(img_25q.rows, img_25q.cols, img.type());
	cv::Mat hist_b_img_jpg = cv::Mat::zeros(img_25q.rows, img_25q.cols, img.type());
	cv::Mat hist_img_jpg = cv::Mat::zeros(img_25q.rows, img_25q.cols, CV_8UC1);





	cv::Mat temp_img;
	img.copyTo(temp_img);
	cv::cvtColor(temp_img,temp_img,cv::COLOR_BGR2GRAY);
	std::cout << blue_channel.channels();
	cv::Mat temp_img_jpg;
	img_25q.copyTo(temp_img_jpg);
	cv::cvtColor(temp_img_jpg, temp_img_jpg, cv::COLOR_BGR2GRAY);
	cv::calcHist(&blue_channel_jpg, 1, 0, cv::Mat(), _b_b_hist_jpg, 1, &hist_size, hist_range, uniform, accumulate);
	cv::calcHist(&green_channel_jpg, 1, 0, cv::Mat(), _b_g_hist_jpg, 1, &hist_size, hist_range, uniform, accumulate);
	cv::calcHist(&red_channel_jpg, 1, 0, cv::Mat(), _b_r_hist_jpg, 1, &hist_size, hist_range, uniform, accumulate);
	cv::calcHist(&temp_img_jpg, 1, 0, cv::Mat(), full_hist_jpg, 1, &hist_size, hist_range, uniform, accumulate);

	

	cv::calcHist(&blue_channel, 1, 0, cv::Mat(), _b_b_hist, 1, &hist_size, hist_range, uniform, accumulate);
	cv::calcHist(&green_channel, 1, 0, cv::Mat(), _b_g_hist, 1, &hist_size, hist_range, uniform, accumulate);
	cv::calcHist(&red_channel, 1, 0, cv::Mat(), _b_r_hist, 1, &hist_size, hist_range, uniform, accumulate);
	cv::calcHist(&temp_img, 1, 0, cv::Mat(), full_hist, 1, &hist_size, hist_range, uniform, accumulate);

	double max_val = 0;		
	cv::minMaxLoc(_b_b_hist, 0, &max_val);

	for (int k = 0; k < hist_size; k++) {
		float const bin_val = _b_b_hist.at<float>(k);
		int const height = cvRound(bin_val * hist_height / max_val);
		cv::line(hist_b_img, cv::Point(k, hist_height - height), cv::Point(k, hist_height), cv::Scalar(255,0,0));
	}
	//cv::imshow("blue_hist.png", hist_b_img);

	for (int k = 0; k < hist_size; k++) {
		float const bin_val = _b_r_hist.at<float>(k);
		int const height = cvRound(bin_val * hist_height / max_val);
		cv::line(hist_r_img, cv::Point(k, hist_height - height), cv::Point(k, hist_height), cv::Scalar(0,0,255));
	}
	//cv::imshow("red_hist.png", hist_r_img);

	for (int k = 0; k < hist_size; k++) {
		float const bin_val = _b_g_hist.at<float>(k);
		int const height = cvRound(bin_val * hist_height / max_val);
		cv::line(hist_g_img, cv::Point(k, hist_height - height), cv::Point(k, hist_height), cv::Scalar(0,255,0));
	}
	//cv::imshow("green_hist.png", hist_g_img);

	for (int k = 0; k < hist_size; k++) {
		float const bin_val = full_hist.at<float>(k);
		int const height = cvRound(bin_val * hist_height / max_val);
		cv::line(hist_img, cv::Point(k, hist_height - height), cv::Point(k, hist_height), cv::Scalar::all(255));
	}
	//cv::imshow("img_hist.png", hist_img);
	

	// JPG 
	
	for (int k = 0; k < hist_size; k++) {
		float const bin_val = _b_b_hist_jpg.at<float>(k);
		int const height = cvRound(bin_val * hist_height / max_val);
		cv::line(hist_b_img_jpg, cv::Point(k, hist_height - height), cv::Point(k, hist_height), cv::Scalar(255, 0, 0));
	}
	

	for (int k = 0; k < hist_size; k++) {
		float const bin_val = _b_r_hist_jpg.at<float>(k);
		int const height = cvRound(bin_val * hist_height / max_val);
		cv::line(hist_r_img_jpg, cv::Point(k, hist_height - height), cv::Point(k, hist_height), cv::Scalar(0, 0, 255));
	}
	//cv::imshow("red_hist.png", hist_r_img);

	for (int k = 0; k < hist_size; k++) {
		float const bin_val = _b_g_hist_jpg.at<float>(k);
		int const height = cvRound(bin_val * hist_height / max_val);
		cv::line(hist_g_img_jpg, cv::Point(k, hist_height - height), cv::Point(k, hist_height), cv::Scalar(0, 255, 0));
	}
	//cv::imshow("green_hist.png", hist_g_img);

	for (int k = 0; k < hist_size; k++) {
		float const bin_val = full_hist_jpg.at<float>(k);
		int const height = cvRound(bin_val * hist_height / max_val);
		cv::line(hist_img_jpg, cv::Point(k, hist_height - height), cv::Point(k, hist_height), cv::Scalar::all(255));
	}
	
	// JPG
	cv::Mat mosaic_img_jpg(img_25q.cols * 2, img_25q.rows * 2, img_25q.type());
	mosaic_img_jpg = 0;
	cv::cvtColor(hist_img_jpg, hist_img_jpg, cv::COLOR_GRAY2BGR);
	cv::Rect2d rect_jpg(0, 0, img_25q.cols, img_25q.rows);
	hist_img_jpg.copyTo(mosaic_img_jpg(rect_jpg));
	rect_jpg.x += rect_jpg.width;
	hist_r_img_jpg.copyTo(mosaic_img_jpg(rect_jpg));
	rect_jpg.y += rect_jpg.height;
	hist_b_img_jpg.copyTo(mosaic_img_jpg(rect_jpg));
	rect_jpg.x -= rect_jpg.width;
	hist_g_img_jpg.copyTo(mosaic_img_jpg(rect_jpg));
	cv::imwrite("mosaic_hist_jpg.jpg", mosaic_img_jpg);
	//cv::imshow("mosaic_hist_jpg.jpg", mosaic_img_jpg);

	cv::Mat mosaic_img(img.cols * 2, img.rows * 2,img.type());
	mosaic_img = 0;	
	cv::cvtColor(hist_img, hist_img, cv::COLOR_GRAY2BGR);
	cv::Rect2d rect(0, 0, img.cols, img.rows);
	hist_img.copyTo(mosaic_img(rect));
	rect.x += rect.width;
	hist_r_img.copyTo(mosaic_img(rect));
	rect.y += rect.height;
	hist_b_img.copyTo(mosaic_img(rect));
	rect.x -= rect.width;
	hist_g_img.copyTo(mosaic_img(rect));
	cv::imwrite("mosaic_hist.png", mosaic_img);
	cv::imshow("mosaic_hist.png",mosaic_img);





	mod_channel_b.setTo(0);
	mod_channel_g.setTo(0);
	mod_channel_r.setTo(0);

	mod_channel_b_jpg.setTo(0);
	mod_channel_g_jpg.setTo(0);
	mod_channel_r_jpg.setTo(0);

	cv::Mat _b_r_color[3] = {mod_channel_b,mod_channel_g,red_channel};
	cv::Mat _b_g_color[3] = { mod_channel_b,green_channel,mod_channel_r };
	cv::Mat _b_b_color[3] = { blue_channel,mod_channel_g,mod_channel_r };

	cv::Mat _b_r_color_jpg[3] = { mod_channel_b_jpg,mod_channel_g_jpg,red_channel_jpg };
	cv::Mat _b_g_color_jpg[3] = { mod_channel_b_jpg,green_channel_jpg,mod_channel_r_jpg };
	cv::Mat _b_b_color_jpg[3] = { blue_channel_jpg,mod_channel_g_jpg,mod_channel_r_jpg };


	cv::Mat _b_r_img;
	cv::Mat _b_g_img;
	cv::Mat _b_b_img;

	cv::Mat _b_r_img_jpg;
	cv::Mat _b_g_img_jpg;
	cv::Mat _b_b_img_jpg;



	cv::merge(_b_r_color, 3, _b_r_img);
	cv::merge(_b_g_color, 3, _b_g_img);
	cv::merge(_b_b_color, 3, _b_b_img);

	cv::merge(_b_r_color_jpg, 3, _b_r_img_jpg);
	cv::merge(_b_g_color_jpg, 3, _b_g_img_jpg);
	cv::merge(_b_b_color_jpg, 3, _b_b_img_jpg);


	cv::Mat result_img(img.rows*2,img.cols*2,img.type());
	result_img = 0;
	cv::Rect2d rc(0,0,img.cols,img.rows);
	img.copyTo(result_img(rc));
	rc.x += rc.width;
	_b_r_img.copyTo(result_img(rc));
	rc.y += rc.height;
	_b_b_img.copyTo(result_img(rc));
	rc.x -= rc.width;
	_b_g_img.copyTo(result_img(rc));
	cv::imwrite("mosaic_img.png", result_img);
	//cv::imshow("mosaic_img.png", result_img);

	cv::Mat result_img_jpg(img_25q.rows * 2, img_25q.cols * 2, img_25q.type());
	result_img_jpg = 0;
	cv::Rect2d rc_jpg(0, 0, img_25q.cols, img_25q.rows);
	img_25q.copyTo(result_img_jpg(rc_jpg));
	rc_jpg.x += rc_jpg.width;
	_b_r_img_jpg.copyTo(result_img_jpg(rc_jpg));
	rc_jpg.y += rc_jpg.height;
	_b_b_img_jpg.copyTo(result_img_jpg(rc_jpg));
	rc_jpg.x -= rc_jpg.width;
	_b_g_img_jpg.copyTo(result_img_jpg(rc_jpg));
	cv::imwrite("mosaic_img.jpg", result_img_jpg);
	//cv::imshow("mosaic_img.jpg", result_img_jpg);

	cv::Mat combined_mosaics(mosaic_img.rows*2,mosaic_img.cols,mosaic_img.type());
	cv::Rect rect2(0, 0, mosaic_img.cols, mosaic_img.rows);
	mosaic_img.copyTo(combined_mosaics(rect2));
	rect2.y += rect2.height;
	mosaic_img_jpg.copyTo(combined_mosaics(rect2));
	//cv::imwrite("combined_mosaics.png",combined_mosaics);
	
	//imshow("monochrome_b_r_img.png", _b_r_img);
	//imshow("monochrome_b_g_img.png", _b_g_img);
	//imshow("monochrome_b_b_img.png", _b_b_img);

	cv::waitKey(0);
	return 0;
}