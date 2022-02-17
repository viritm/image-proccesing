#include <opencv2/opencv.hpp>

int main() {

	cv::Mat img = cv::imread("D:/SandBox/buivolov_e_a/data/cross_0256x0256.png",1);
	cv::imwrite("badquality.jpg",img, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY,25});

	cv::Mat img_25q = cv::imread("D:/SandBox/buivolov_e_a/build.vs.2019/prj.labs/lab02/badquality.jpg");
	//cv::imshow("badquality.jpg",img_25q);

	cv::Mat a_channels[3];

	cv::split(img,a_channels);
	cv::Mat blue_channel = a_channels[0];
	cv::Mat green_channel = a_channels[1];
	cv::Mat red_channel = a_channels[2];

	cv::Mat mod_channel_b = a_channels[0].clone();
	cv::Mat mod_channel_g = a_channels[1].clone();
	cv::Mat mod_channel_r = a_channels[2].clone();


	mod_channel_b.setTo(0);
	mod_channel_g.setTo(0);
	mod_channel_r.setTo(0);
	cv::Mat _b_r_color[3] = {mod_channel_b,mod_channel_g,red_channel};
	cv::Mat _b_g_color[3] = { mod_channel_b,green_channel,mod_channel_r };
	cv::Mat _b_b_color[3] = { blue_channel,mod_channel_g,mod_channel_r };


	cv::Mat _b_r_img;
	cv::Mat _b_g_img;
	cv::Mat _b_b_img;


	cv::merge(_b_r_color, 3, _b_r_img);
	cv::merge(_b_g_color, 3, _b_g_img);
	cv::merge(_b_b_color, 3, _b_b_img);

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


	int hist_size = 256;

	float range[] = { 0,256 };
	const float*  hits_range[] = { range };

	bool uniform = true, accumulate = false;






	cv::imshow("img_mosaic.png", result_img);


    
	/*imshow("monochrome_b_r_img.png", _b_r_img);
	imshow("monochrome_b_g_img.png", _b_g_img);
	imshow("monochrome_b_b_img.png", _b_b_img);*/

	cv::waitKey(0);
	return 0;
}