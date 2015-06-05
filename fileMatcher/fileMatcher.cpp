/*
 * @ SURF_FLANN Matcher 
 * @ To compile, run:
 * @ g++ fileMatcher.cpp -o fileMatcher `pkg-config --cflags --libs opencv`
 */

#include <stdio.h>
#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

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
  int minHessian = 400;
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
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  //-- Show detected matches
  imshow( "Matches", img_matches );
  for( int i = 0; i < (int)matches.size(); i++ )
  { printf( "-- Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, matches[i].queryIdx, matches[i].trainIdx ); }
  waitKey(0);
  return 0;
}