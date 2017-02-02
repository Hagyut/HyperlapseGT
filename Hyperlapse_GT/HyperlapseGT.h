#ifndef __HYPERLAPSE_GT_H__
#define __HYPERLAPSE_GT_H__

#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

#define RED_COL Scalar(0, 0, 255)
#define GREEN_COL Scalar(0, 255, 0)
#define BLUE_COL Scalar(255, 0, 0)
#define BLACK_COL Scalar(0, 0, 0)
#define WHITE_COL Scalar(255, 255, 255)

#define RANSAC_THRES 20
#define K_MEANS_ATTEMPS 3

#define CLUSTER_MOVE_COST_CONST 0.0005
#define COST_MAX 200

typedef struct frame_info {
	int num;
	Mat frame;
	vector<KeyPoint> kpts;
	Mat desc;
} FrameInfo;

class HyperlapseGT {

public:	// public methods

	static HyperlapseGT* create(int target_speed, int K) {
		if (target_speed == 2 || target_speed == 4 || target_speed == 8)
			return new HyperlapseGT(target_speed, K);
		else
			return NULL;
	}
	void remove() {
		this->~HyperlapseGT();
	}

	void openVideo(string fpath);			// set input video path
	void setOutputVideoPaTH(string fpath);	// set output video path
	void run();								// create hyperlpase video

private:	// private methods

	void getMinPrevCost(int i, int j, int& cost, int& argmin);
	HyperlapseGT(int speed, int K);
	~HyperlapseGT();
	int getCost(int i, int j);

private:	// private variables

	string i_fpath;
	string o_fpath;
	bool v_ready;
	int v_width;
	int v_height;
	int v_frame_cnt;
	int v_fps;

	int speed_up;	// hyperlapse speed
	int win_a_size;
	int win_b_size;
	int K;			// K means variable

	VideoCapture v_cap;
	VideoWriter v_writer;
	vector<FrameInfo> fi_window_a;
	vector<FrameInfo> fi_window_b;

	ORB orb;
	BFMatcher matcher;

	int** Dv;
	int** Tv;

	vector<int> path;
};

#endif