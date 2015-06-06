#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
	// Fast detector and brief descriptor
	// Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(32);
	Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(10, true);

	clock_t time_start = clock(); //start of clock

	Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );

	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;

	fast->detect(img_1,keypoints_1);
	clock_t time_detect = clock()-time_start; //end of clock, counting time
	cout << "time for detect: " <<((double)time_detect) / CLOCKS_PER_SEC * 1000 << "ms"<< endl; //print out time
	cout << keypoints_1.size() << endl;

	return 0;
}
