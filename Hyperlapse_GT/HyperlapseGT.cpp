#include "HyperlapseGT.h"

HyperlapseGT::HyperlapseGT(int speed, int K) {
	speed_up = speed;
	this->K = K;
	win_a_size = speed / 2;
	win_b_size = speed + 1;

	int n_features = 500;
	float scale_factor = 2.5f;
	int n_levels = 5;
	int edge_thres = 31;
	int first_level = 0;
	int WTA_K = 2;
	int score_type = ORB::HARRIS_SCORE;
	int patch_size = 31;

	orb = ORB::ORB(
		n_features,
		scale_factor,
		n_levels,
		edge_thres,
		first_level,
		WTA_K,
		score_type,
		patch_size
		);

	matcher = BFMatcher(NORM_HAMMING);
}
HyperlapseGT::~HyperlapseGT() {
	for (int it = 0; it < win_b_size; it++) {
		delete Dv[it];
		delete Tv[it];
	}
	delete[] Dv;
	delete[] Tv;
}

void HyperlapseGT::openVideo(string fpath) {
	this->i_fpath = fpath;
	v_ready = v_cap.open(i_fpath);
	if (v_ready) {
		v_width = (int)v_cap.get(CV_CAP_PROP_FRAME_WIDTH);
		v_height = (int)v_cap.get(CV_CAP_PROP_FRAME_HEIGHT);
		v_frame_cnt = v_cap.get(CV_CAP_PROP_FRAME_COUNT);
		v_fps = v_cap.get(CV_CAP_PROP_FPS);
	}

	int rows = v_frame_cnt - speed_up / 2 * 3;
	Dv = new int*[rows];
	Tv = new int*[rows];
	for (int it = 0; it < rows; it++) {
		Dv[it] = new int[win_b_size];
		Tv[it] = new int[win_b_size];
	}
}

void HyperlapseGT::setOutputVideoPaTH(string fpath) {
	o_fpath = fpath;
}

void HyperlapseGT::run() {
	Mat in_frame;

	if (!v_ready || v_frame_cnt < 64) {
		cout << "적절한 비디오가 준비되지 않았습니다." << endl;
		return;
	}

	int frame_num = 0;
	while (1) {

		FrameInfo fi;

		if (!v_cap.read(in_frame))
			break;
		in_frame.copyTo(fi.frame);
		orb(fi.frame, cv::Mat(), fi.kpts, fi.desc, false);
		fi.num = frame_num;

		if (frame_num < speed_up / 2) {
			fi_window_a.push_back(fi);
		}
		else if (frame_num >= speed_up / 2 && frame_num < speed_up / 2 * 3 + 1) {
			fi_window_b.push_back(fi);
			if (frame_num == ((speed_up / 2) * 3)) {
				for (int it = 0; it < win_b_size; it++) {
					int i = frame_num - (speed_up / 2 * 3);
					int j = i + (speed_up / 2) + it;
					int win_b_idx = j - i - (speed_up / 2);
					Dv[i][win_b_idx] = getCost(i, j);
				}
			}
		}
		else if (frame_num >= speed_up / 2 * 3 + 1 && frame_num < win_b_size * 2 + 1) {
			fi_window_a.erase(fi_window_a.begin());
			fi_window_a.push_back(fi_window_b.at(0));
			fi_window_b.erase(fi_window_b.begin());
			fi_window_b.push_back(fi);

			for (int it = 0; it < win_b_size; it++) {
				int i = frame_num - (speed_up / 2 * 3);
				int j = i + (speed_up / 2) + it;
				int win_b_idx = j - i - (speed_up / 2);
				Dv[i][win_b_idx] = getCost(i, j);
			}
		}
		else {
			fi_window_a.erase(fi_window_a.begin());
			fi_window_a.push_back(fi_window_b.at(0));
			fi_window_b.erase(fi_window_b.begin());
			fi_window_b.push_back(fi);

			int i = frame_num - (speed_up / 2 * 3);
			int prev_cost, argmin;
			getMinPrevCost(i, 0, prev_cost, argmin);
			for (int it = 0; it < win_b_size; it++) {
				int j = i + (speed_up / 2) + it;
				int win_b_idx = j - i - (speed_up / 2);
				Dv[i][win_b_idx] = getCost(i, j) + prev_cost;
				Tv[i][win_b_idx] = argmin;
			}
		}
		cout << "Working progress.. " << frame_num + 1 << "/" << v_frame_cnt << endl;
		frame_num++;
	}

	ofstream ofs("output.txt");
	int rows = v_frame_cnt - speed_up / 2 * 3;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < win_b_size; j++) {
			ofs << Tv[i][j] << " ";
		}
		ofs << endl;
	}
	ofs.close();

	int min_cost = INT_MAX;
	int s, d;
	for (int it = 0; it < (win_b_size + 1) / 2; it++) {
		int row = rows - 1 - it;
		for (int it2 = 0; it2 < win_b_size; it2++) {
			int tmp_cost = Dv[row][it2];
			if (tmp_cost < min_cost) {
				s = row;
				d = it2;
			}
		}
	}

	path.push_back(s + speed_up / 2 + d);
	while (s > speed_up) {
		path.push_back(s);
		int b = Tv[s][d];
		s = s - b;
		d = b - speed_up / 2;
	}

	v_cap.release();
	v_writer.open(o_fpath, 0, v_fps, Size(v_width, v_height), true);
	v_cap.open(i_fpath);
	Mat in_f, out_f;
	int fn = 0;
	while (1) {
		if (!v_cap.read(in_frame))
			break;
		if (path.empty())
			break;

		v_cap.read(in_f);
		if (fn == path.back()) {
			out_f = in_f;
			v_writer.write(out_f);
			path.pop_back();
		}
		fn++;
	}
	v_writer.release();
}

void HyperlapseGT::getMinPrevCost(int i, int j, int& cost, int& argmin) {

	for (int it = 0; it < win_b_size; it++) {
		int prev_frame = i - (speed_up / 2 + it);
		if (prev_frame < 0)
			break;
		int temp_cost = Dv[prev_frame][it];
		if (it == 0) {
			cost = temp_cost;
			argmin = speed_up / 2 + it;
		}
		else {
			if (temp_cost < cost) {
				cost = temp_cost;
				argmin = speed_up / 2 + it;
			}
		}
	}
}

int HyperlapseGT::getCost(int i, int j) {

	int win_b_idx = j - i - (speed_up / 2);

	FrameInfo fi_i = fi_window_a.at(0);
	FrameInfo fi_j = fi_window_b.at(win_b_idx);

	vector<DMatch> matches;
	matcher.match(fi_i.desc, fi_j.desc, matches);

	vector<Point2f> mpts_i, mpts_j;
	for (int it = 0; it < matches.size(); it++) {
		DMatch& match = matches[it];
		mpts_i.push_back(fi_i.kpts[match.queryIdx].pt);
		mpts_j.push_back(fi_j.kpts[match.trainIdx].pt);
	}

	Mat homography;
	vector<uchar> outlier_mask;
	if (mpts_i.size() > 4)
		homography = findHomography(mpts_i, mpts_j, RANSAC, RANSAC_THRES, outlier_mask);

	Mat labels_i, centers_i;
	int flags = KMEANS_RANDOM_CENTERS;
	TermCriteria tc;

	cv::kmeans(mpts_i, K, labels_i, tc, K_MEANS_ATTEMPS, flags, centers_i);

	vector<Point2f> centers_pi, centers_pj;
	vector<float> cluster_var_i;
	vector<int> cluster_point_cnt;
	for (int it = 0; it < K; it++) {
		centers_pi.push_back(Point2f(centers_i.at<float>(it, 0), centers_i.at<float>(it, 1)));
		centers_pj.push_back(Point2f(0.0f, 0.0f));
		cluster_var_i.push_back(0.0f);
		cluster_point_cnt.push_back(0);
	}

	// Codes to ensure that clustering works


	//
	//Mat fr_i_cp, fr_j_cp;
	//fi_i.frame.copyTo(fr_i_cp);
	//fi_j.frame.copyTo(fr_j_cp);

	//for (int it = 0; it < labels_i.rows; it++) {

	//	if (((char)outlier_mask.at(it)) == 1) {
	//		int idx = labels_i.at<int>(it);
	//		Point2f p = mpts_i[it];
	//		Point2f center = Point(centers_i.at<float>(idx, 0), centers_i.at<float>(idx, 1));

	//		float x_diff = center.x - p.x;
	//		float y_diff = center.y - p.y;
	//		float elm1 = x_diff * x_diff + y_diff * y_diff;
	//		cluster_var_i.at(idx) += elm1;
	//		cluster_point_cnt.at(idx) += 1;

	//		centers_pj.at(idx) += mpts_j[it];

	//		//
	//		if (idx == 0) {
	//			circle(fr_i_cp, mpts_i[it], 4, RED_COL, 3);
	//			circle(fr_j_cp, mpts_j[it], 4, RED_COL, 3);
	//		}
	//		else if (idx == 1) {
	//			circle(fr_i_cp, mpts_i[it], 4, GREEN_COL, 3);
	//			circle(fr_j_cp, mpts_j[it], 4, GREEN_COL, 3);
	//		}
	//		else if (idx == 2) {
	//			circle(fr_i_cp, mpts_i[it], 4, BLUE_COL, 3);
	//			circle(fr_j_cp, mpts_j[it], 4, BLUE_COL, 3);
	//		}
	//		else if (idx == 3) {
	//			circle(fr_i_cp, mpts_i[it], 4, WHITE_COL, 3);
	//			circle(fr_j_cp, mpts_j[it], 4, WHITE_COL, 3);
	//		}
	//		else if (idx == 4) {
	//			circle(fr_i_cp, mpts_i[it], 4, BLACK_COL, 3);
	//			circle(fr_j_cp, mpts_j[it], 4, BLACK_COL, 3);
	//		}
	//	}
	//}

	//imshow("Video i", fr_i_cp);
	//imshow("Video j", fr_j_cp);
	//waitKey(0);

	for (int it = 0; it < K; it++) {
		int point_cnt = cluster_point_cnt.at(it);
		if (point_cnt > 1) {
			cluster_var_i.at(it) /= (point_cnt - 1);
		}

		if (point_cnt != 0) {
			centers_pj.at(it).x /= point_cnt;
			centers_pj.at(it).y /= point_cnt;
		}
		else {
			centers_pj.at(it).x = 0.0f;
			centers_pj.at(it).y = 0.0f;
		}
	}

	int cost_h = 0;

	if (mpts_i.size() > 4) {		// homography가 구해졌을 때
		Point2f frame_center(v_width / 2.0f, v_height / 2.0f);
		Mat_<double> src(3/*rows*/, 1/*cols*/);

		src(0, 0) = frame_center.x;
		src(1, 0) = frame_center.y;
		src(2, 0) = 1.0f;

		Mat_<double> dst = homography * src;

		Point2f translated(dst(0, 0), dst(1, 0));
		Point2f diff = translated - frame_center;

		cost_h = sqrt(diff.x * diff.x + diff.y * diff.y);
		if (cost_h > COST_MAX)
			cost_h = COST_MAX;
	}
	else {
		cost_h = COST_MAX;
	}

	int cost_c = 0;
	for (int it = 0; it < K; it++) {
		Point2f diff = centers_pj.at(it) - centers_pi.at(it);
		float dist = sqrt(diff.x * diff.x + diff.y * diff.y);
		cost_c += dist * sqrt(cluster_var_i.at(it)) * CLUSTER_MOVE_COST_CONST;
	}
	if (cost_c > COST_MAX)
		cost_c = COST_MAX;


	return cost_h + cost_c;
}