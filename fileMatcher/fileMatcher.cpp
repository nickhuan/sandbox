/*
 * @ fileMatcher 
 * @ To compile, run:
 * @ g++ ../fileMatcher/fileMatcher.cpp -o fileMatcher `pkg-config --cflags --libs opencv`
 */

#include <stdio.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

//Define constances here
Mat camera_matrix = (Mat_<double>(3,3) << 334.6936985593561,                 0, 158.8031671890114,
                                                          0, 329.5380406075342, 113.1013024157152,
                                                          0,                 0,                 1); //Microsoft HD3000 intrinsics

// Caution for picking values from Mat: to get value of projection_matrix at row 2, column 3 (113.1013024157152)
// is camera_matrix.at<double>(1,2) or projection_matrix.at<double>(6)

//Function MatchesFilter to filter out bad matches (>2*minDistance)
std::vector< DMatch > MatchesFilter(std::vector< DMatch > raw_matches, cv::Mat descriptors);

int main( int argc, char** argv )
{
  if( argc != 3 )
  { printf("usage: matcher <VideoPath>\n"); return -1; }

  //define constants
  double focal = (camera_matrix.at<double>(0,0) + camera_matrix.at<double>(1,1))/2; //get focal length of the camera from camera_matrix
  Point2d principle_point = Point2d(camera_matrix.at<double>(0,2), camera_matrix.at<double>(1,2)); //get principle point from camera_matrix
  cout << focal << principle_point << endl;

  cout << "Clock:" << clock() << endl; //print current time
  clock_t time_start = clock(); //start of clock

  Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );
  Mat img_2 = imread( argv[2], IMREAD_GRAYSCALE );

  if( !img_1.data || !img_2.data )
    { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;

  cout << "Clock:" << clock() << endl; //print current time

  //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
  int minHessian = 600;
  Ptr<SURF> detector = SURF::create(minHessian);
  // Fast detector and brief descriptor
  // Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(32);
  // Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(10, true);
  // ...
  detector->detect(img_1, keypoints_1); //Find interest points
  cout << "Clock:" << clock() << endl; //print current time
  detector->detect(img_2, keypoints_2); 
  cout << "Clock:" << clock() << endl; //print current time

  detector->compute(img_1, keypoints_1, descriptors_1); //Compute brief descriptors at each keypoint location
  detector->compute(img_2, keypoints_2, descriptors_2);

  //-- Step 2: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );

  cout << "Clock:" << clock() << endl; //print current time

  //-- Step 3: Only keep the good matches
  std::vector< DMatch > filtered_matches = MatchesFilter(matches, descriptors_1);
  cout << "filtered Matches " << filtered_matches.size() << endl;
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               filtered_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  cout << "Clock:" << clock() << endl; //print current time
  //-- Show detected matches
  // imshow( "Matches", img_matches );
  // for( int i = 0; i < (int)matches.size(); i++ )
  // { printf( "-- Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, matches[i].queryIdx, matches[i].trainIdx ); }

  //-- Step 4: Get camera poses
  vector<Point2f> points_1, points_2; //array of feature points
  for( int i = 0; i < filtered_matches.size(); i++ )
  {
   //-- Get the keypoints from the good matches
   points_1.push_back( keypoints_1[ filtered_matches[i].queryIdx ].pt );
   points_2.push_back( keypoints_2[ filtered_matches[i].trainIdx ].pt );
   // cout << "KeyPoint" << i <<": in left image: " << keypoints_1[ filtered_matches[i].queryIdx ].pt;
   // cout << endl <<"      and in right image: " << keypoints_2[ filtered_matches[i].trainIdx ].pt << endl;
  }
  cout << "Clock:" << clock() << endl; //print current time

  // Mat mat_points_1 = Mat(points_1).reshape(1,2);
  // Mat mat_points_2 = Mat(points_2).reshape(1,2);   
  // cout << "Mat(points_1):" << Mat(points_1) << endl;
  // cout << "Mat(points_2):" << Mat(points_2) << endl;
  // cout << "mat_points1:" << mat_points_1 << endl;
  // cout << "mat_points2:" << mat_points_2 << endl;

  Mat rotation_matrix, translation_matrix, mask;
  Mat essential_matrix = findEssentialMat(points_1, points_2, focal, principle_point, RANSAC, 0.999, 1.0, mask);
  recoverPose(essential_matrix, points_1, points_2, rotation_matrix, translation_matrix, focal, principle_point, mask);
  cout << "rotation_matrix:" << endl << rotation_matrix << endl << "translation_matrix:" << endl << translation_matrix << endl;
  cout << "mask:" << endl << mask << endl;

  //-- Step 5: Triangulation
  Mat projection_matrix_1 = (Mat_<double>(3,4) << 1, 0, 0, 0,
                                                  0, 1, 0, 0,
                                                  0, 0, 1, 0);
  Mat relative_projection_matrix_1to2;
  hconcat(rotation_matrix, translation_matrix, relative_projection_matrix_1to2);
  cout << "projection_matrix_2" << relative_projection_matrix_1to2 << endl;
  cout << "Clock:" << clock() << endl; //print current time
  Mat projection_matrix_2 = relative_projection_matrix_1to2;
  // Mat projection_matrix_2 = projection_matrix_1 * relative_projection_matrix_1to2;
  cout << "Clock:" << clock() << endl; //print current time

  Mat points4D;
  triangulatePoints(projection_matrix_1, projection_matrix_2, points_1, points_2, points4D); //output 4xN reconstructed points
  cout << "triangulated points in homogenous coordinates: " << endl << points4D << endl;

  //Time counting to measure performance
  clock_t time_all = clock()-time_start; //end of clock, counting time
  cout<< "time for the whole process: " <<((double)time_all) / CLOCKS_PER_SEC * 1000 << " ms"<< endl; //print out time
  waitKey(0);
  return 0;
}

std::vector< DMatch > MatchesFilter(std::vector< DMatch > raw_matches, cv::Mat descriptors)
{
  //only keep the good matches
  double max_dist = 0; double min_dist = 100;
  std::vector< DMatch > good_matches;
  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors.rows; i++ )
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
  for( int i = 0; i < descriptors.rows; i++ )
  { if( raw_matches[i].distance <= max(2*min_dist, 0.02) )
    { good_matches.push_back( raw_matches[i]); }
  }
  cout << "Number of good matches: " << good_matches.size() << endl;

  return good_matches;
};
