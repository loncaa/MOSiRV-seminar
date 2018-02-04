#include "stdafx.h"
#include <opencv2/video/tracking.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

char *cascadePath = "haarcascade_frontalface_default.xml";
CascadeClassifier faceCascade;
vector<Rect> faces;

/* MeaShift variables */
MatND backproj;
Rect trackWindow;
int trackObject = 1;

/* Histogram variables */
float range[] = { 0, 255 };
const float* ranges[] = { range, range };
int channels[] = { 0, 1 };
int histSize[] = { 64, 64 };
Mat objectHistogram;
Mat globalHistogram;
Mat maskBackproj;

bool clicked = false;
bool startSetup = true;
clock_t begin_time = 0;

/* Mask Update */
Mat updateMask(Mat hsvFrameRoi, Scalar lowS, Scalar highS)
{
	Mat maskUpdate, hsvRoi, face;

	GaussianBlur(hsvFrameRoi, hsvRoi, Size(3, 3), 1.5, 1.5);
	inRange(hsvRoi, lowS, highS, maskUpdate);
	//morphologyEx(maskUpdate, maskUpdate, CV_MOP_ERODE, Mat(3, 3, 1), Point(-1, -1), 1);
	medianBlur(maskUpdate, maskUpdate, 5);

	imshow("mask", maskUpdate);
	return maskUpdate;
}

/* Pronalazi lice */
bool detectFace(Mat imagFrame, Scalar lowS, Scalar highS)
{
	Mat grayFrame, hsvFace, faceMask;
	cvtColor(imagFrame, grayFrame, CV_BGR2GRAY);
	equalizeHist(grayFrame, grayFrame);
	faceCascade.detectMultiScale(grayFrame, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(25, 25));

	for (int i = 0; i < faces.size(); i++)
	{
		cvtColor(imagFrame(faces[i]), hsvFace, CV_BGR2HSV);
		faceMask = updateMask(hsvFace, lowS, highS);
		int pix = countNonZero(faceMask);
		int imgPix = faceMask.rows * faceMask.cols;

		if (((imgPix * 10) / 100) <= pix) return true;
	}
	return false;
}

/* Resetiraj algoritam na klik miša itd.*/
void onMouse(int event, int x, int y, int, void*)
{
	if (event == EVENT_LBUTTONDOWN){

		trackObject = 0;
		clicked = false;
		startSetup = false;

		cout << "FACE DETECTINO -> CAMSHIFT" << endl;
	}

	if (event == EVENT_MBUTTONDOWN)
	{
		/* resetira */
		startSetup = true;
		trackObject = 1;
		clicked = false;

		cout << "RESET -> MASK SETUP" << endl;
	}

	if (event == EVENT_RBUTTONDOWN && !startSetup)
	{
		begin_time = clock();
		clicked = true;

		cout << "BEGIN HISTOGRAM UPDATE" << endl;
	}
}

void updateHistogram(Mat image, Rect trackRect, Scalar lowS, Scalar highS)
{
	Mat img, frameRoi, mask;

	GaussianBlur(image, img, Size(3, 3), 21, 21);
	cvtColor(img, img, CV_BGR2HSV);
	frameRoi = img(trackRect);
	mask = updateMask(frameRoi, lowS, highS);

	calcHist(&frameRoi, 1, channels, mask, objectHistogram, 2, histSize, ranges, true, false);
	calcHist(&img, 1, channels, noArray(), globalHistogram, 2, histSize, ranges, true, true);

	for (int y = 0; y < objectHistogram.rows; y++) {
		for (int x = 0; x < objectHistogram.cols; x++) {
			objectHistogram.at<float>(y, x) /= globalHistogram.at<float>(y, x);
		}
	}

	normalize(objectHistogram, objectHistogram, 0, 255, NORM_MINMAX);
}

int main(int argc, const char** argv)
{
	VideoCapture cam;
	Mat frame, image;

	#pragma region Basic Init

	if (!faceCascade.load(cascadePath))
	{
		cout << "Wrong cascade path!!" << endl;
		return false;
	}

	cout << "Camera Setup: " << endl;
	if (cam.open(0))
		cout << "We're connected to camera" << endl << endl;

	/* Windows design */
	int vmin = 70, vmax = 255, //vmin = 98
		smin = 0, smax = 255,
		hmin = 0, hmax = 180;

	namedWindow("Window", 1);
	namedWindow("SlideBar", 1);
	setMouseCallback("Window", onMouse, 0);
	createTrackbar("Vmin", "SlideBar", &vmin, 255, 0);
	createTrackbar("Vmax", "SlideBar", &vmax, 255, 0);
	createTrackbar("Smin", "SlideBar", &smin, 255, 0);
	createTrackbar("Smax", "SlideBar", &smax, 255, 0);
	createTrackbar("Hmin", "SlideBar", &hmin, 180, 0);
	createTrackbar("Hmax", "SlideBar", &hmax, 180, 0);

	#pragma endregion

	int _vmin, _vmax,
		_smin, _smax,
		_hmin, _hmax;

	while (true){
		#pragma region MaskSetup

		while (trackObject && startSetup)
		{
			cam >> frame;
			if (frame.empty())
				break;

			frame.copyTo(image);
			cvtColor(image, image, CV_BGR2HSV);

			/* podešavanje maske */
			_vmin = vmin; _vmax = vmax;
			_smin = smin; _smax = smax;
			_hmin = hmin; _hmax = hmax;

			/* raèunanje maske */
			GaussianBlur(image, image, Size(3, 3), 1.5, 1.5);
			inRange(image, Scalar(MIN(_hmin, _hmax), MIN(_smin, _smax), MIN(_vmin, _vmax)),
						   Scalar(MAX(_hmin, _hmax), MAX(_smin, _smax), MAX(_vmin, _vmax)), maskBackproj);
			//morphologyEx(maskBackproj, maskBackproj, CV_MOP_ERODE, Mat(3, 3, 1), Point(-1, -1), 1);
			medianBlur(maskBackproj, maskBackproj, 5);

			imshow("Mask", maskBackproj);
			imshow("Window", frame);
			waitKey(33);

		}


		#pragma endregion

		#pragma	region FaceDetection

		/* face detection petlja */
		while (!trackObject)
		{
			cam >> frame;
			if (frame.empty())
				break;

			if (detectFace(frame, Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)), 
								  Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax))))
			{
				trackWindow = faces.at(0);
				trackObject = 1;
				break;
			}

			imshow("Window", frame);
			waitKey(33);
		}

		#pragma endregion

		updateHistogram(frame, trackWindow, Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)), 
											Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)));
		#pragma region CamShift

		while (trackObject && !startSetup)
		{
			/* Podesavanje maske */
			_vmin = vmin; _vmax = vmax;
			_smin = smin; _smax = smax;
			_hmin = hmin; _hmax = hmax;

			try
			{
				/* Histogram update */
				float time_t = round(float(clock() - begin_time) / CLOCKS_PER_SEC);
				if (time_t >= 5 && clicked)
				{
					updateHistogram(frame, trackWindow, Scalar(MIN(_hmin, _hmax), MIN(_smin, _smax), MIN(_vmin, _vmax)), 
														Scalar(MAX(_hmin, _hmax), MAX(_smin, _smax), MAX(_vmin, _vmax)));
					begin_time = clock();
					cout << "Histogram update!" << endl;
				}
			
				frame.copyTo(image);
				cvtColor(image, image, CV_BGR2HSV);

				/* racunanje backprojection slike */
				GaussianBlur(image, image, Size(3, 3), 1.5, 1.5);
				inRange(image, Scalar(MIN(_hmin, _hmax), MIN(_smin, _smax), MIN(_vmin, _vmax)),
							   Scalar(MAX(_hmin, _hmax), MAX(_smin, _smax), MAX(_vmin, _vmax)), maskBackproj);
				//morphologyEx(maskBackproj, maskBackproj, CV_MOP_ERODE, Mat(3, 3, 1), Point(-1, -1), 1);
				medianBlur(maskBackproj, maskBackproj, 5);
				imshow("Mask", maskBackproj);

				/* Racunanje BackProjectiona za hue i saturation */
				calcBackProject(&image, 1, channels, objectHistogram, backproj, ranges);
				//backproj &= maskBackproj;
				imshow("backproj", backproj);

				/* CamShift algoritam*/
				CamShift(backproj, trackWindow, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 0.01));
				rectangle(frame, trackWindow, Scalar(255, 255, 0));
				
			}
			catch (Exception e){
				trackObject = 0;
			}

			imshow("Window", frame);
			waitKey(33);

			cam >> frame;
			if (frame.empty())
				break;
		}
		#pragma endregion

	}
	return 0;
}
