#include <iostream>
#include <opencv2\opencv.hpp>

#include "HyperlapseGT.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	HyperlapseGT *hgt = HyperlapseGT::create(2, 4);
	hgt->openVideo("D:\\Hyperlapse\\in_video\\TargetVideo2.mp4");
	hgt->setOutputVideoPaTH("D:\\Hyperlapse\\out_video\\OutVideo2.avi");
	hgt->run();
	hgt->remove();
}