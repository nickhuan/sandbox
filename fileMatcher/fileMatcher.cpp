/*
 * @ SURF_FLANN Matcher 
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

cv::Mat projection_matrix = (Mat_<double>(3,4) << 332.7283325195312, 0, 158.5335691355958, 0, 0, 329.1090698242188, 112.7895539800684, 0, 0, 0, 1, 0); 

//Function MatchesFilter to filter out bad matches (>2*minDistance)
std::vector< DMatch > MatchesFilter(std::vector< DMatch > raw_matches, cv::Mat descriptors);

int main( int argc, char** argv )
{
  if( argc != 3 )
  { printf("usage: matcher <VideoPath>\n"); return -1; }

  Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );
  Mat img_2 = imread( argv[2], IMREAD_GRAYSCALE );

  if( !img_1.data || !img_2.data )
    { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;

  //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
  int minHessian = 1000;
  Ptr<SURF> detector = SURF::create(minHessian);
  // Fast detector and brief descriptor
  // Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(32);
  // Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(10, true);
  // ...
  detector->detect(img_1, keypoints_1); //Find interest points
  detector->detect(img_2, keypoints_2); 

  detector->compute(img_1, keypoints_1, descriptors_1); //Compute brief descriptors at each keypoint location
  detector->compute(img_2, keypoints_2, descriptors_2);

  //-- Step 2: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );

  //-- Step 3: Only keep the good matches
  std::vector< DMatch > filtered_matches = MatchesFilter(matches, descriptors_1);
  cout << "filtered Matches " << filtered_matches.size() << endl;
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               filtered_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  //-- Show detected matches
  imshow( "Matches", img_matches );
  for( int i = 0; i < (int)matches.size(); i++ )
  { printf( "-- Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, matches[i].queryIdx, matches[i].trainIdx ); }

  vector<Point2f> points_1, points_2; //array of feature points
  for( int i = 0; i < filtered_matches.size(); i++ )
  {
   //-- Get the keypoints from the good matches
   points_1.push_back( keypoints_1[ filtered_matches[i].queryIdx ].pt );
   points_2.push_back( keypoints_2[ filtered_matches[i].trainIdx ].pt );
   cout << "KeyPoint" << i <<": in left image: " << keypoints_1[ filtered_matches[i].queryIdx ].pt;
   cout << endl <<"      and in right image: " << keypoints_2[ filtered_matches[i].trainIdx ].pt << endl;
  }

  Mat points4D;
  triangulatePoints(projection_matrix, projection_matrix, points_1, points_2, points4D); //output 4xN reconstructed points
  cout << projection_matrix << endl;
  cout << "triangulated points in homogenous coordinates: " << points4D << endl;
  
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
