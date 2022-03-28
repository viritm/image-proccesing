#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

uint8_t setBrightness(const uint8_t index) {
	return cv::saturate_cast<uint8_t>(pow(index,(index + 1) / 255.0));
}

void  changeBrightness(const cv::Mat& inputArray, cv::Mat& outputArray) {
	
	if (inputArray.type() != CV_8UC1) {
		std::cout << "InputArray type is not CV_8UC1 ";
		return;
	}
	if (inputArray.type() != outputArray.type()) {
		std::cout << "InputArray type is not match outputArray type";
		return;
	}
		
	inputArray.copyTo(outputArray);
	for (int i = 0; i < outputArray.cols; i++) {
		for (int j = 0; j < outputArray.rows; j++) {
			outputArray.at<uint8_t>(j, i) = setBrightness(i/3);
		}
	}
}

int main() {
	
	cv::Mat testImg = cv::imread("D:/SandBox/buivolov_e_a/data/cross_0256x0256.png");
	cv::imwrite("testImg.png",testImg);

	cv::Mat gradient(180,768, CV_8UC1);
	cv::Mat res;
	for (int i = 0; i < 768; i++) {
		for (int j = 0; j < 180; j++) {
			gradient.at<uint8_t>(j,i) = i/3;
		}
	}
	cv::imshow("tst1.png", gradient);
	changeBrightness(gradient, res);
	cv::imshow("tst.png",res);

	cv::Mat graph(testImg.rows,testImg.cols ,CV_8UC1);
	graph.setTo(255);
	
	for (int i = 0; i < graph.cols; ++i) {
		graph.at<uint8_t>(255 - setBrightness(i), i) = 0;
	}
	cv::Rect2d rc2(2, 0, graph.rows, graph.cols);
	cv::Mat graph1(graph.rows+3,graph.cols+3, CV_8UC1);
	graph1 = 255;
	graph.copyTo(graph1(rc2));
	for (int i = 0; i < graph1.cols; i++) {
		graph1.at<uint8_t>(graph1.rows - 2,i) = 0;
		graph1.at<uint8_t>(graph1.rows - 3, i) = 0;
	}
	for (int i = 0; i < graph1.rows; i++) {
		graph1.at<uint8_t>(i, 1) = 0;
		graph1.at<uint8_t>(i, 2) = 0;
	}

	graph = graph1;
	cv::resize(graph, graph, cv::Size(512, 512), 0, 0, cv::InterpolationFlags::INTER_CUBIC);
	cv::imwrite("graph.png", graph);
	cv::imshow("graph.png", graph);
	
	cv::Mat testImgGray(testImg.rows, testImg.cols, testImg.type());
	cv::cvtColor(testImg, testImgGray, cv::COLOR_BGR2GRAY);
	cv::imwrite("testImgGray.png",testImgGray);
	
	cv::Mat lut(1, 256, CV_8UC1);
	for (int k = 0; k < lut.cols; k++) {
		lut.at<uint8_t>(0, k) = setBrightness(k);
	}
	cv::Mat grayImg(testImgGray.rows,testImgGray.cols,testImgGray.type());
	cv::LUT(testImgGray, lut, grayImg);
	cv::imwrite("grayImg.png", grayImg);
	//cv::imshow("grayImg.png", grayImg);
	
	cv::Mat colorArray[3];

	cv::split(testImg, colorArray);
	//cv::imshow("test.png", testImg);

	cv::Mat arrayBGR[3];
	colorArray[0].copyTo(arrayBGR[0]);
	colorArray[1].copyTo(arrayBGR[1]);
	colorArray[2].copyTo(arrayBGR[2]);
		
	cv::Mat finalArrayBGR[3];

	cv::LUT(arrayBGR[0], lut, finalArrayBGR[0]);
	cv::LUT(arrayBGR[1], lut, finalArrayBGR[1]);
	cv::LUT(arrayBGR[2], lut, finalArrayBGR[2]);

	//cv::imshow("channelB.png",finalArrayBGR[0]);
	//cv::imshow("channelG.png", finalArrayBGR[1]);
	//cv::imshow("channelR.png", finalArrayBGR[2]);
	
	cv::Mat channelsImg(testImgGray.rows*2,testImgGray.cols*2,testImgGray.type());
	cv::Rect2d rc(0,0,testImgGray.cols,testImgGray.rows);
	grayImg.copyTo(channelsImg(rc));
	rc.x += rc.width;
	finalArrayBGR[2].copyTo(channelsImg(rc));
	rc.y += rc.height;
	finalArrayBGR[0].copyTo(channelsImg(rc));
	rc.x -= rc.width;
	finalArrayBGR[1].copyTo(channelsImg(rc));
	cv::imwrite("channelsImg.png",channelsImg);
	//cv::imshow("channelsImg.png",channelsImg);

	cv::waitKey(0);
	
	return 0;
}