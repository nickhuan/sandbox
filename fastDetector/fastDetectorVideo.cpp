#include <stdio.h>
#include <iostream>	//cout
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>	//fast
#include <opencv2/imgcodecs.hpp>	//imread
#include <opencv2/highgui.hpp>	//VideoCapture
#include <opencv2/xfeatures2d.hpp>	//Brief


using namespace std;
using namespace cv;

int threshold_FAST = 10;

int main( int argc, char** argv )
{
	cout << "Enter threshold for FAST Feature Detector:";
	cin >> threshold_FAST;
	Mat img_raw;

	VideoCapture cap(0); //open default camera
	if( ! cap.isOpened () )  // check if we succeeded
	return -1;

	// Fast detector and brief descriptor
	// Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(32);
	Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(threshold_FAST, true);

	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	while(true)
	{
		clock_t time_start = clock(); //start of clock
		cap >> img_raw;
		fast->detect(img_raw,keypoints_1);
		clock_t time_detect = clock()-time_start; //end of clock, counting time
		cout << "time for detect: " <<((double)time_detect) / CLOCKS_PER_SEC * 1000 << "ms"<< endl; //print out time
		cout << keypoints_1.size() << endl;
	}

	return 0;
}
