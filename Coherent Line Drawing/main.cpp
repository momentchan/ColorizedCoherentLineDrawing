#include "imatrix.h"
#include "ETF.h"
#include "fdog.h"
#include "myvec.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "FunctionDecalration.h"

using namespace std;
using namespace cv;
Mat color;
int colorThres=250;


Mat GrayCoherentLine(Mat src){
	// We assume that you have loaded your input image into an imatrix named "img"
	int rows = src.rows;
	int cols = src.cols;

	imatrix img(rows, cols);

	// copy from dst (unsigned char) to img (int)
	for (int y = 0; y < rows; y++)
	for (int x = 0; x < cols; x++)
		img[y][x] = src.at<uchar>(y, x);
	int image_x = img.getRow();
	int image_y = img.getCol();
	//////////////////////////////////////////////////


	//////////////////////////////////////////////////
	ETF e;
	e.init(image_x, image_y);
	e.set(img); // get gradients from input image

	//e.set2(img); // get gradients from gradient map

	e.Smooth(4, 2);
	//////////////////////////////////////////////////////

	///////////////////////////////////////////////////
	double tao = 0.99;
	double thres = 0.7;
	GetFDoG(img, e, 1.0, 3.0, tao);
	GrayThresholding(img, thres);

	/////////////////////////////////////////////
	// copy result from img (int) to dst (unsigned char)
	Mat dst = Mat::zeros(src.size(), 0);
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			dst.at<uchar>(y, x) = (uchar)img[y][x];
		}
	}

	//imshow(" ", dst); waitKey(0);
	cout << dst.type();

	/*Mat canny;
	Canny(src, canny, 90, 180, 3);
	canny = ~canny;
	imshow("", canny); waitKey(0);
	imwrite("Canny.jpg", canny);*/
	imwrite("Coherent.jpg", dst);
	return dst;
}
Mat SimpleColor(Mat src){
	Mat dst = color.clone();
	for (int y = 0; y < src.rows; y++){
		for (int x = 0; x < src.cols; x++){
			if ((int)src.at<uchar>(y, x) >colorThres){
				dst.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
			}

		}
	}
	return dst;

}
int main()
{
	Mat gray = imread("R2.jpg",0);
	color = imread("R2.jpg");
	Mat GL = GrayCoherentLine(gray);
	Mat CL = SimpleColor(GL);
	imshow("", CL); waitKey(0);
}
