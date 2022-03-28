#include <opencv2/opencv.hpp>
#include <iostream>


void getFrames(cv::VideoCapture& _video, cv::Mat _frames[3]) {

	cv::Mat frame;

	double framesCount =_video.get(cv::CAP_PROP_FRAME_COUNT);

	uint16_t i = 0;
	uint16_t k = 0;

	while (1) {
		if (i == static_cast<int>((framesCount / 5) * 2) ||
			i == static_cast<int>((framesCount / 5) * 3) ||
			i == static_cast<int>((framesCount / 5) * 4)) {
			_video >> _frames[k];
			i++;
			k++;
			continue;
		}
		_video >> frame;
		if (frame.empty()) break;
		i++;
	}

}

void colorReduction(cv::Mat& inputArray, cv::Mat& outputArray) {

	cv::Mat temp(inputArray.rows, inputArray.cols, CV_8UC1);
	cv::cvtColor(inputArray, temp, cv::COLOR_BGR2GRAY);
	temp.copyTo(outputArray);

}

void colorReuctionAll(cv::Mat in_frames[], cv::Mat out_frames[], int size) {
	for (int i = 0; i <size ; ++i) {

		colorReduction(in_frames[i], out_frames[i]);
	}
}

void invColor(cv::Mat& inputArray,cv::Mat& outputArray, int maxValue) {
	for (int x = 0; x < inputArray.size().width; ++x) {
		for (int y = 0; y < inputArray.size().height; ++y) {
			if ((int)inputArray.at<uint8_t>(y, x) == maxValue) outputArray.at<uint8_t>(y, x) = 0;
			else outputArray.at<uint8_t>(y, x) = maxValue;
		}
	}
}

void binArray_OTSU(cv::Mat _array[], const int size) {
	for (int i = 0; i < size; ++i) {
		cv::threshold(_array[i], _array[i], 0, 250, cv::THRESH_OTSU | cv::THRESH_BINARY);
	}
}

void binArray_OTSU_INV(cv::Mat _array [],const int size) {
	for (int i = 0; i < size; ++i) {
		cv::threshold(_array[i], _array[i], 0, 250, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);
	}
}

void binArray_INV(cv::Mat _array[], const int size, int threshold, int maxValue) {
	for (int i = 0; i < size; ++i) {
		cv::threshold(_array[i], _array[i], threshold, maxValue, cv::THRESH_BINARY_INV);
	}
}

void binArray(cv::Mat _array[], const int size, int threshold, int maxValue) {
	for (int i = 0; i < size; ++i) {
		cv::threshold(_array[i], _array[i], threshold, maxValue, cv::THRESH_BINARY);
	}
}

void morphologyExToAll(cv::Mat inputArray [], cv::Mat outputArray [], const int size, cv::MorphTypes type, cv::Mat mask) {
		
		for (int i = 0; i < size; ++i) {
		cv::morphologyEx(inputArray[i], outputArray[i], type, mask);
		}
}

void highlightingComponents(cv::Mat& inputArray, cv::Mat& outputArray,int connectivity) {

	cv::Mat labels, stats, centroids;
	cv::Mat result(inputArray.size(), CV_8UC3);
	int nLabels = cv::connectedComponentsWithStats(inputArray, labels, stats, centroids, connectivity);

	std::vector<cv::Vec3b> colors(nLabels);
	colors[0] = cv::Vec3b(0, 0, 0);

	for (int currLabel = 1; currLabel < nLabels; ++currLabel) {
		colors[currLabel] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
	}

	for (int r = 0; r < result.rows; ++r) {
		for (int c = 0; c < result.cols; ++c) {
			int label = labels.at<int>(r, c);
			cv::Vec3b& pixel = result.at<cv::Vec3b>(r, c);
			pixel = colors[label];
		}
	}
	result.copyTo(outputArray);

}

void deletePreMaxAreaComponentAndTrash(cv::Mat& inputArray, cv::Mat& outputArray, int connectivity) {

	cv::Mat labels, stats, centroids;
	cv::Mat result(inputArray.size(), CV_8UC3);
	int nLabels = cv::connectedComponentsWithStats(inputArray, labels, stats, centroids, connectivity);

	std::vector<cv::Vec3b> colors(nLabels);
	colors[0] = cv::Vec3b(0, 0, 0);


	int currArea = 0;
	int preMaxIndex = -1;
	for (int currLabel = 1; currLabel < nLabels; ++currLabel) {
		if (stats.at<int>(currLabel, cv::CC_STAT_AREA) < 170) {
			colors[currLabel] = cv::Vec3b(0, 0, 0);
			continue;
		}
		if (stats.at<int>(currLabel, cv::CC_STAT_AREA) > currArea) {
			currArea = stats.at<int>(currLabel, cv::CC_STAT_AREA);
			preMaxIndex = currLabel;
		}
		colors[currLabel] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
	}
	int index = preMaxIndex;
	preMaxIndex = -1;
	currArea = 0;
	for (int currLabel = 1; currLabel < nLabels; ++currLabel) {
		if (currLabel == index) continue;
		if (stats.at<int>(currLabel, cv::CC_STAT_AREA) > currArea) {
			currArea = stats.at<int>(currLabel, cv::CC_STAT_AREA);
			preMaxIndex = currLabel;
		}
	}
	index = preMaxIndex;

	for (int r = 0; r < result.rows; ++r) {
		for (int c = 0; c < result.cols; ++c) {
			int label = labels.at<int>(r, c);
			if (label == index) {
				cv::Vec3b& pixel = result.at<cv::Vec3b>(r, c);
				pixel = cv::Vec3b(0,0,0);
				continue;
			}
			cv::Vec3b& pixel = result.at<cv::Vec3b>(r, c);
			pixel = colors[label];
		}
	}
	result.copyTo(outputArray);

}


void highlightingComponentsEachImg(cv::Mat inputArray [],cv::Mat outputArray [],const int size, int connectivity) {
	for (int i = 0; i < size; ++i) {
		highlightingComponents(inputArray[i], outputArray[i], connectivity); 
	}
}

cv::Mat mergeImg(const cv::Mat& first,const cv::Mat& second) {

	if (first.type() != second.type()) exit(-1);

	if (first.rows == second.rows || first.rows > second.rows) {

		cv::Mat result(first.rows, first.cols + second.cols, first.type());
		result = 0;
		cv::Rect2d rc(0, 0, first.cols, first.rows);
		first.copyTo(result(rc));
		rc.x +=second.cols;
		second.copyTo(result(rc));
		return cv::Mat(result);
	}
	else
	{
		cv::Mat result(second.rows, first.cols + second.cols, first.type());
		result = 0;
		cv::Rect2d rc1(0, 0, first.cols, first.rows);
		first.copyTo(result(rc1));
		cv::Rect2d rc2(first.cols, 0, second.cols, second.rows);
		second.copyTo(result(rc2));
		return cv::Mat(result);

	}

}

int main() {

	cv::VideoCapture myVideo1 ("D:/SandBox/buivolov_e_a/data/video1.mp4");
	cv::VideoCapture myVideo2 ("D:/SandBox/buivolov_e_a/data/video2.mp4");
	cv::VideoCapture myVideo3 ("D:/SandBox/buivolov_e_a/data/video3.mp4");
	cv::VideoCapture myVideo4 ("D:/SandBox/buivolov_e_a/data/video4.mp4");
	cv::VideoCapture myVideo5 ("D:/SandBox/buivolov_e_a/data/video5.mp4");

	cv::Mat frames1 [3];
	cv::Mat frames2 [3];
	cv::Mat frames3 [3];
	cv::Mat frames4 [3];
	cv::Mat frames5 [3];

	getFrames(myVideo1, frames1);
	getFrames(myVideo2, frames2);
	getFrames(myVideo3, frames3);
	getFrames(myVideo4, frames4);
	getFrames(myVideo5, frames5);


	cv::imwrite("frames1[1].png", frames1[1]);
	cv::imwrite("frames2[0].png", frames2[0]);
	cv::imwrite("frames3[2].png", frames3[2]);
	

	cv::Mat binFrames1[3];
	cv::Mat binFrames2[3];
	cv::Mat binFrames3[3];
	cv::Mat binFrames4[3];
	cv::Mat binFrames5[3];

	colorReuctionAll(frames1, binFrames1, 3);
	colorReuctionAll(frames2, binFrames2, 3);
	colorReuctionAll(frames3, binFrames3, 3);
	colorReuctionAll(frames4, binFrames4, 3);
	colorReuctionAll(frames5, binFrames5, 3);


	cv::Mat test;
	binFrames1[1].copyTo(test);

	cv::Mat binFrame2_0;
	binFrames2[0].copyTo(binFrame2_0);
	cv::threshold(binFrame2_0, binFrame2_0, 150, 250, cv::THRESH_BINARY);

	binArray_OTSU_INV(binFrames1, 3);
	binArray_OTSU(binFrames2, 3);
	binArray_OTSU(binFrames3, 3);
	binArray(binFrames4, 3, 150, 250);
	binArray(binFrames5, 3, 150, 250);

	
	cv::Mat res1(mergeImg(test, binFrames1[1]));
	cv::imwrite("test.png",res1);

	cv::Mat res2(mergeImg(binFrames2[0], binFrame2_0));
	
	cv::rectangle(res2, cv::Rect2d(0, 0, binFrames2[0].cols, binFrames2[0].rows), {220},1);
	cv::imwrite("test1.png",res2);


	cv::Mat morfFrames1 [3];
	cv::Mat morfFrames2 [3];
	cv::Mat morfFrames3 [3];
	cv::Mat morfFrames4 [3];
	cv::Mat morfFrames5 [3];

 	cv::Mat elementRect = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3));
	cv::Mat elementCross = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(2, 2));
	
	morphologyExToAll(binFrames1, morfFrames1, 3, cv::MorphTypes::MORPH_OPEN, elementRect);
	morphologyExToAll(binFrames2, morfFrames2, 3, cv::MorphTypes::MORPH_OPEN, elementRect);
	morphologyExToAll(binFrames3, morfFrames3, 3, cv::MorphTypes::MORPH_OPEN, elementRect);
	morphologyExToAll(binFrames4, morfFrames4, 3, cv::MorphTypes::MORPH_OPEN, elementRect);
	morphologyExToAll(binFrames5, morfFrames5, 3, cv::MorphTypes::MORPH_OPEN, elementRect);


	cv::Mat res3 = mergeImg(binFrames3[2],morfFrames3[2]);
	cv::rectangle(res3, cv::Rect2d(0, 0, binFrames3[2].cols, binFrames3[2].rows), { 220 }, 1);
	cv::imwrite("test2.png",res3);

	
	cv::Mat highlightedFrames1[3];
	cv::Mat highlightedFrames2[3];
	cv::Mat highlightedFrames3[3];
	cv::Mat highlightedFrames4[3];
	cv::Mat highlightedFrames5[3];

	highlightingComponentsEachImg(binFrames1, highlightedFrames1, 3, 4);
	highlightingComponentsEachImg(binFrames2, highlightedFrames2, 3, 4);
	highlightingComponentsEachImg(binFrames3, highlightedFrames3, 3, 4);
	highlightingComponentsEachImg(binFrames4, highlightedFrames4, 3, 4);
	highlightingComponentsEachImg(binFrames5, highlightedFrames5, 3, 4);

	cv::Mat morfHigh3_2;
	highlightingComponents(morfFrames3[2], morfHigh3_2, 4);

	cv::Mat res4(mergeImg(highlightedFrames3[2],morfHigh3_2));	
	cv::rectangle(res4, cv::Rect2d(0, 0, highlightedFrames3[2].cols, highlightedFrames3[2].rows), { 0,0,220 }, 1);

	highlightingComponents(binFrame2_0,binFrame2_0,4);

	cv::Mat res5;
		
	deletePreMaxAreaComponentAndTrash(binFrames3[2], res5, 4);
	cv::imwrite("test5.png",res5);


	cv::imwrite("highlightedFrames1[1].png", highlightedFrames1[1]);
	cv::imwrite("binFrame2_0.png", binFrame2_0);


	cv::waitKey(0);

	return 0;
}