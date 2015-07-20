#include "imatrix.h"
#include "ETF.h"
#include "fdog.h"
#include "myvec.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	// We assume that you have loaded your input image into an imatrix named "img"
	Mat R2 = imread("R2.jpg", 0);
	int rows = R2.rows;
	int cols = R2.cols;

	imatrix img(rows, cols);

	// copy from dst (unsigned char) to img (int)
	for (int y = 0; y < rows; y++)
		for (int x = 0; x < cols; x++) 
			img[y][x] = R2.at<uchar>(y, x);
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
	
	Mat dst = Mat::zeros(R2.size(), 0);
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			dst.at<uchar>(y, x) = (uchar)img[y][x];
		}
	}
	imshow(" ", dst); 
	cout << dst.type();

	Mat canny;
	Canny(R2, canny, 90, 180, 3);
	canny = ~canny;
	imshow("", canny); waitKey(0);
	imwrite("Canny.jpg", canny);
	imwrite("Coherent.jpg", dst);
}

