/*
 * @ 3D-to-2D approach for pose estimation using perspective n-points 
 * @ Software Requirement: OpenCV 3.0.0 rc1
 * @ To compile, run:
 * @ g++ keypointVO.cpp -o keypointVO `pkg-config --cflags --libs opencv`
 */

#include <stdio.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/line_descriptor.hpp>

#include "tcpServer.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::line_descriptor;

/* define constances here */
int image_width = 640;
int image_height = 480;
int minHessian = 1000; //~200ms when it's 400; ~100ms when 1000
int NoKP_event = 100;
Mat camera_matrix = (Mat_<double>(3,3) << 334.6936985593561,                 0, 158.8031671890114,
                                                          0, 329.5380406075342, 113.1013024157152,
                                                          0,                 0,                 1); //Microsoft HD3000 intrinsics
/* constant definitions */
double focal = (camera_matrix.at<double>(0,0) + camera_matrix.at<double>(1,1))/2; //get focal length of the camera from camera_matrix
Point2d principle_point = Point2d(camera_matrix.at<double>(0,2), camera_matrix.at<double>(1,2)); //get principle point from camera_matrix
// cout << focal << principle_point << endl;

//RANSAC for finding Essential Matrix
double prob = 0.99; //desirable level of confidence (probability) that the estimated matrix is correct.
double epipolarThreshold = 1.0;
//the maximum distance from a point to an epipolar line in pixels, 
//beyond which the point is considered an outlier and is not used for computing the final fundamental matrix. 
//It can be set to something like 1-3, depending on the accuracy of the point localization, image resolution, and the image noise.

/* function declaration */
std::vector<DMatch> MatchesFilter(std::vector<DMatch> raw_matches);
std::vector<DMatch> MatchesHomoFilter(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, std::vector<DMatch> good_matches);
std::vector<DMatch> MatchesEMatFilter(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, std::vector<DMatch> good_matches);
//Function MatchesFilter to filter out bad matches (>2*minDistance)
// void convert4Dto3D(Mat, vector<Point3f>);

int main( int argc, char** argv )
{
	/* variable declaration */
	Mat img_raw;
	Mat img_frame_previous(image_width, image_height, CV_8UC1, Scalar::all(0));
	Mat img_events(image_width, image_height, CV_8UC1, Scalar::all(0));
	Mat img_frame_current(image_width, image_height, CV_8UC1, Scalar::all(0)); //image frames
	vector<KeyPoint> keypoints_frame_previous, keypoints_events, keypoints_frame_current; //vector of keypoints
	Mat descriptors_frame_previous, descriptors_events, descriptors_frame_current; //descriptors
	std::vector<DMatch> matches; //matching vector
	Mat points4D_current; //triangulated 3D points in homogenous coordinates
	vector<Point3f> points3D_current; //3D points

	/* detection, description matching methods*/
	// // Option Package 1 -- SURF detector + SURF descriptor + FLANN matcher
	// Ptr<SURF> keypointDetector = SURF::create(minHessian);
	// Ptr<SURF> descriptorExtractor = SURF::create(minHessian);
	// FlannBasedMatcher flannMatcher;//not a pointer, need to define a pointer below
	// Ptr<DescriptorMatcher> descriptorMatcher = &flannMatcher; 
	// cout << "Minimum Hessian Number:";
	// cin >> minHessian;
	// // Option Package 2 -- FAST detector + BRIEF descriptor + binary Matcher
	Ptr<FastFeatureDetector> keypointDetector = FastFeatureDetector::create(10, true);
	// Ptr<BriefDescriptorExtractor> descriptorExtractor = BriefDescriptorExtractor::create(32);
	Ptr<ORB> descriptorExtractor = ORB::create();
	// Ptr<BinaryDescriptorMatcher> descriptorMatcher = cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
	BFMatcher bfMatcher = BFMatcher(NORM_HAMMING,true); //Problem:: cannot put TRUE here
	//NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4
	Ptr<DescriptorMatcher> descriptorMatcher = &bfMatcher;
	// Ptr<DescriptorMatcher> descriptorMatcher = cv::DescriptorMatcher::create(BruteForce_Hamming);

	/* media input */
	//VideoCapture cap ( argv[1] ); //open video file from address specified in argument
	VideoCapture cap(0); //open default camera
	if( ! cap.isOpened () )  // check if we succeeded
	return -1;
	
	// /* TCP Server Setup */
	// tcpServer server(9999);
	// bool flagKP; //tell TCP receiver whether keypoints are from frame or event: 1 for keyframes, and 0 for events
	clock_t time_b4loop = clock(); //start of clock

	// while(cap.read(img_events) && cv::waitKey(30) != 27)
	for(int fIdx=0;;fIdx++)
	{
		cout << "\n [Frame No. : " << fIdx << "]" << endl;
		// cout << "is empty? " << descriptors_frame_current.empty() << endl;

		clock_t time_start = clock(); //start of clock
		//This is only executed for normal frame (event)
		if (fIdx%30 != 0 && !descriptors_frame_current.empty())
		{
			cap >> img_raw;
			cv::cvtColor(img_raw, img_events, cv::COLOR_BGR2GRAY);

			//-- Step 1: Detect the keypoints using SURF detector, compute the descriptors
			keypointDetector->detect(img_events, keypoints_events);
			KeyPointsFilter::retainBest(keypoints_events, NoKP_event); //only keep the best 100 keypoints
			cout << "No. of Keypoints in events: " << keypoints_events.size() << endl;
			if(keypoints_events.size() < 5) continue;

			descriptorExtractor->compute(img_events, keypoints_events, descriptors_events);

			//-- Step 2: Matching descriptor vectors using FLANN matcher
			descriptorMatcher->match(descriptors_frame_current, descriptors_events, matches);
			cout << "Number of matches: " << matches.size() << endl;

			//-- Step 3: Only keep the good matches
			vector<DMatch> better_matches = MatchesFilter(matches);

			if(better_matches.size() < 5)
			{
				cout << "Number of matches: " << matches.size() << endl;
			}
			else
			{
				std::vector<DMatch> filtered_matches = MatchesEMatFilter(keypoints_frame_current, keypoints_events, better_matches);
				// cout << "filtered Matches " << filtered_matches.size() << endl;

				//-- Draw only "good" matches
				Mat img_matches;
				drawMatches(img_frame_current, keypoints_frame_current, img_events, keypoints_events,
							filtered_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
							vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
				//-- Show detected matches
				imshow( "Matched Ketpoints", img_matches );

				// -- TCP socket 
				// TODO: learn the example to send correct data
				// flagKP = 0;
				// for(int i=0; i<keypoints_frame_current.size()+1;i++)
				// {
				// 	ostringstream oss;
				// 	if(i=0) //packet header with frame ID and flag
				// 	{
				// 		oss << "FID:" << fIdx << " Flag:" << flagKP << endl;
				// 		string toSend = oss.str();
				// 	}
				// 	else
				// 	{
				// 		int x = keypoints_frame_current->at(i).pt.x;
				// 		int y = keypoints_frame_current->at(i).pt.y;
				// 		oss <<" KID:" << i << " X:" << x << " Y:" << y << endl;
				// 		string toSend = oss.str();
				// 	}
				// 	if (sendto(s, toSend.c_str(), toSend.length(), 0, (struct sockaddr*) &si_other, slen) == -1) die("sendto()");
				// }

				// -- to be migrated into ROS -- begin --//
				//TODO: condition needs to be changed to if the triangulated points are non-empty
				//TODO: restructure the program to 2 steps: 
				//1. Initialisation: Triangulation 
				// if(fIdx > 60)
				// {
				// 	Mat intrinsics;
				// 	vector<double> distCoeffs;
				// 	vector<Point2f> points_event;
				// 	vector<double> rvec, tvec;
				// 	solvePnP(points3D_current, points_event, intrinsics, distCoeffs,rvec, tvec);
				// 	cout << "Rotation Vector: " << rvec << endl;
				// 	cout << "Translation Vector" << tvec << endl;
				// }
				// -- to be migrated into ROS -- end --//
			}

		}

		//TODO: filter keyframes e.g. : set threshold for numbers of stable repeatable keypoints 
		//this is only done for the key frames, starts from 30, and then 60, 90 and etc.
		// else if (fIdx != 0)
		// {
		// 	cap >> img_raw; //load frame
		// 	cv::cvtColor(img_raw, img_frame_current, cv::COLOR_BGR2GRAY);
		// 	cout << "[Keyframe No. : " << fIdx/30 << "]" << endl; //print keyframe number
		// 	keypointDetector->detect(img_frame_current, keypoints_frame_current); 
		// 	descriptorExtractor->compute(img_frame_current, keypoints_frame_current, descriptors_frame_current);
		// 	cout << "No. of Keypoints detected in Keyframe: " << keypoints_frame_current.size() << endl;
		// 	descriptorMatcher->match(descriptors_frame_previous, descriptors_frame_current, matches); //match keyframes
		// 	std::vector<DMatch> filtered_matches = MatchesFilter(keypoints_frame_previous ,keypoints_frame_current , matches); //filter matches
		// 	cout << "filtered Matches " << filtered_matches.size() << endl;
		// 	if(filtered_matches.size() < 8 || filtered_matches.size() > 90) continue;

		// 	vector<Point2f> points_frame_current(filtered_matches.size()), points_frame_previous(filtered_matches.size()); //array of feature points
		// 	for( int i = 0; i < filtered_matches.size(); i++ )
		// 	{
		// 		//-- Get the keypoints from the good matches
		// 		points_frame_previous[i] =keypoints_frame_previous[filtered_matches[i].queryIdx].pt;
		// 		points_frame_current[i] =keypoints_frame_current[filtered_matches[i].trainIdx].pt;				 
		// 		// points_frame_previous.push_back( keypoints_frame_previous[ filtered_matches[i].queryIdx ].pt );
		// 		// points_frame_current.push_back( keypoints_frame_current[ filtered_matches[i].trainIdx ].pt );
		// 		cout << "keypoints_frame_previous: " << keypoints_frame_previous[ filtered_matches[i].queryIdx ].pt << endl;
		// 		cout << "keypoints_frame_current: " << keypoints_frame_current[ filtered_matches[i].trainIdx ].pt << endl;
		// 	}

		// 	// -- TCP socket 
		// 	// TODO: learn the example to send correct data
		// 	// flagKP = 1;
		// 	// for(int i=0; i<keypoints_frame_current.size()+1;i++)
		// 	// {
		// 	// 	ostringstream oss;
		// 	// 	if(i=0) //packet header with frame ID and flag
		// 	// 	{
		// 	// 		oss << "FID:" << fIdx << " Flag:" << flagKP << endl;
		// 	// 		string toSend = oss.str();
		// 	// 	}
		// 	// 	else
		// 	// 	{
		// 	// 		int x = keypoints_frame_current->at(i).pt.x;
		// 	// 		int y = keypoints_frame_current->at(i).pt.y;
		// 	// 		oss <<" KID:" << i << " X:" << x << " Y:" << y << endl;
		// 	// 		string toSend = oss.str();
		// 	// 	}
		// 	// 	if (sendto(s, toSend.c_str(), toSend.length(), 0, (struct sockaddr*) &si_other, slen) == -1) die("sendto()");
		// 	// }

		// 	// // -- to be migrated into ROS -- begin --//
		// 	// Mat rotation_matrix, translation_matrix, mask;
		// 	// clock_t time_preEMat = clock();
		// 	// Mat essential_matrix = findEssentialMat(points_frame_previous, points_frame_current, focal, principle_point, RANSAC, prob, epipolarThreshold, mask);
		// 	// clock_t time4EMat = clock() - time_preEMat;
		// 	// cout<< "Time for Essential Matrix calculation:" << ((double)time4EMat) / CLOCKS_PER_SEC * 1000 << " ms" << endl;
		// 	// recoverPose(essential_matrix, points_frame_previous, points_frame_current, rotation_matrix, translation_matrix, focal, principle_point, mask);
		// 	// // cout << "rotation_matrix:" << endl << rotation_matrix << endl << "translation_matrix:" << endl << translation_matrix << endl;
		// 	// // cout << "mask:" << endl << mask << endl;

		// 	// Mat projection_matrix_origin = (Mat_<double>(3,4) << 1, 0, 0, 0,
		// 	// 												0, 1, 0, 0,
		// 	// 												0, 0, 1, 0);
		// 	// Mat relative_projMat_prevToCurrent;
		// 	// hconcat(rotation_matrix, translation_matrix, relative_projMat_prevToCurrent);
		// 	// //TODO: only triangulate filtered matched keypoints (use the mask generated by essential matrix RANSAC)
		// 	// //TODO: make it run continuously (now only first 2 keyframes)
		// 	// triangulatePoints(projection_matrix_origin, relative_projMat_prevToCurrent, points_frame_previous, points_frame_current, points4D_current); //output 4xN reconstructed points
		// 	// // cout << "triangulated points in homogenous coordinates: " << points4D_current << endl;

		// 	// //TODO: convert 4D points into 3D points
		// 	// // convert4Dto3D(points4D_current, points3D_current);
		// 	// // -- to be migrated into ROS -- end --//
		// }

		//this is only done for the first frame [Frame 0] only
		else
		{
			cap >> img_raw;
			cv::cvtColor(img_raw, img_frame_current, cv::COLOR_BGR2GRAY);
			cout << "[This is the first frame]" << endl;
			keypointDetector->detect(img_frame_current, keypoints_frame_current);
			if(keypoints_frame_current.size() < 5) continue;
			descriptorExtractor->compute(img_frame_current, keypoints_frame_current, descriptors_frame_current);
			cout << "No. of Keypoints detected in frame_current: " << keypoints_frame_current.size() << endl;
			// Mat img_kps;
			// 	drawKeypoints(img_frame_current, keypoints_frame_current, img_kps);
			// 	imshow("Keypoints", img_kps);
			// 	waitKey(1);
			// -- TCP socket 
			// TODO: learn the example to send correct data
		}
		waitKey(1); //non-zero value waitkey to make drawMatch run automatically without pressing any key
		// -- store all "current" data from buffer to "previous" in archive for comparison
		// img_frame_previous = img_frame_current;
		// keypoints_frame_previous = keypoints_frame_current; //put current keypoints from buffer to archive
		// descriptors_frame_previous = descriptors_frame_current; //load current descriptors from buffer to archive
		clock_t time_end = clock()-time_start; //end of clock, counting time
		cout<< "Total Time taken:" << ((double)time_end) / CLOCKS_PER_SEC * 1000 << " ms" << endl; //print out time
	}
	return 0;
}

std::vector<DMatch> MatchesFilter(std::vector<DMatch> raw_matches)
{
  //only keep the good matches
  double max_dist = 0; double min_dist = 100;
  std::vector<DMatch> good_matches;
  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < raw_matches.size(); i++ )
  { double dist = raw_matches[i].distance;
  if( dist < min_dist ) min_dist = dist;
  if( dist > max_dist ) max_dist = dist;
  }
  // printf("-- Max dist : %f \n", max_dist );
  // printf("-- Min dist : %f \n", min_dist );
  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  //-- PS.- radiusMatch can also be used here.
  for( int i = 0; i < raw_matches.size(); i++ )
  {
  	if( raw_matches[i].distance <= max(2*min_dist, 0.02) )
  		{
  			good_matches.push_back( raw_matches[i]);
  		}
  }
  cout << "Number of good matches: " << good_matches.size() << endl;

  return good_matches;
};

std::vector<DMatch> MatchesHomoFilter(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, std::vector<DMatch> good_matches)
{
	//RANSAC using Homography
	clock_t time_ransac = clock(); //start of clock
	vector<Point2f> points_1(good_matches.size()), points_2(good_matches.size()); //array of feature points
	for( int i = 0; i < good_matches.size(); i++ )
	{
		//-- Get the keypoints from the good matches
		points_1[i] =keypoints_1[good_matches[i].queryIdx].pt;
		points_2[i] =keypoints_2[good_matches[i].trainIdx].pt;
	}

	Mat mask;
	Mat essential_matrix = findHomography(points_1, points_2, RANSAC, 3, mask, 2000, 0.995);

	std::vector<DMatch> matches_ransac;
	for(int i=0; i < good_matches.size();i++)
	{
		if(mask.at<bool>(0,i))
		{
			matches_ransac.push_back(good_matches[i]);
		}
	}
	cout << "Number of matches after RANSAC: " << matches_ransac.size() << endl;
	clock_t ransac_duration = clock()-time_ransac; //end of clock, counting time
	cout<< "Time Homography:" << ((double)ransac_duration) / CLOCKS_PER_SEC * 1000 << " ms" << endl; //print out time

	return matches_ransac;
};

std::vector<DMatch> MatchesEMatFilter(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, std::vector<DMatch> good_matches)
{
	//RANSAC using Essential Matrix 
	clock_t time_ransac = clock(); //start of clock
	vector<Point2f> points_1(good_matches.size()), points_2(good_matches.size()); //array of feature points
	for( int i = 0; i < good_matches.size(); i++ )
	{
		//-- Get the keypoints from the good matches
		points_1[i] =keypoints_1[good_matches[i].queryIdx].pt;
		points_2[i] =keypoints_2[good_matches[i].trainIdx].pt;
	}

	Mat mask;
	Mat essential_matrix = findEssentialMat(points_1, points_2, focal, principle_point, RANSAC, prob, epipolarThreshold, mask);

	std::vector<DMatch> matches_ransac;
	for(int i=0; i < good_matches.size();i++)
	{
		if(mask.at<bool>(0,i))
		{
			matches_ransac.push_back(good_matches[i]);
		}
	}
	cout << "Number of matches after RANSAC: " << matches_ransac.size() << endl;
	clock_t ransac_duration = clock()-time_ransac; //end of clock, counting time
	cout<< "Time EMat:" << ((double)ransac_duration) / CLOCKS_PER_SEC * 1000 << " ms" << endl; //print out time

	return matches_ransac;
};
