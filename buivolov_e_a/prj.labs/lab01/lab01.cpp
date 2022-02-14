#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <chrono>

int main() {

	cv::Mat img(180, 768, CV_8UC1);

	// create an gradient
	for (int k = 0; k < 180; k++) {
		for (int i = 0; i < 768; ++i) {
			img.at<uint8_t>(k, i) = i / 3;
		}
	}
					
	// create an image with gamma-correction
	
	auto start = std::chrono::high_resolution_clock::now();
	 
	cv::Mat second_img(180, 768, CV_32FC1);				
	img.convertTo(img, CV_32FC1,1.0F/255.0F);	        // convert IMG to CV_32FC1 format using scaling
	cv::pow(img, 2.2, second_img);					   // apply gamma-correction to newImg
	second_img.convertTo(second_img, CV_8UC1,255);    // reverse scaling and converting to CV_8UC1
	img.convertTo(img, CV_8UC1,255);                 // reverse scaling and converting to CV_8UC1

	auto stop = std::chrono::high_resolution_clock::now();
	auto time_res1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	
   



	cv::Mat third_img(180,768, CV_8UC1);
	img.copyTo(third_img);


	start = std::chrono::high_resolution_clock::now();
    // apply gamma-correction to each pixel 
	for (int k = 0; k < 180; k++) {
		for (int i = 0; i < 768; ++i) {
			third_img.at<uint8_t>(k, i) = (pow((third_img.at<uint8_t>(k, i) / 255.0F), 2.2F) * 255);
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	auto time_res2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	cv::Mat result(180, 768, CV_8UC1);

	const cv::Rect2d rc(0,0,768,60);
	cv::Rect2d move_rc(0, 0, 768, 60);

	// copy rectangle matrix fro img to result
	img(rc).copyTo(result(move_rc));
	cv::rectangle(result, move_rc, { 100 }, 1);
	move_rc.y += rc.height;
	second_img(rc).copyTo(result(move_rc));
	cv::rectangle(result, move_rc, { 250 }, 1);
	move_rc.y += rc.height;
	third_img(rc).copyTo(result(move_rc));
	cv::rectangle(result, move_rc, { 150 }, 1);

  //cv::imwrite("lab01.png", img);
  //cv::imshow("lab01.png", img);

  //cv::imwrite("lab01_2.png", second_img);
  //cv::imshow("lab01_2.png", second_img);

  //cv::imwrite("lab01_3.png", third_img);
  //cv::imshow("lab01_3.png", third_img);

  
  std::cout << "Time to GC through POW = " << time_res1.count() << " ms"<<  '\n';
  std::cout << "Time to GC through manipulation each pixel = " << time_res2.count() << " ms" << '\n';

  cv::imwrite("lab01_4.png", result);
  cv::imshow("lab01_4.png", result);

  cv::waitKey(0);
}
