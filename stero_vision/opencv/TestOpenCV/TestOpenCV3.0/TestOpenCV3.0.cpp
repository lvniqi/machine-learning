/*
* starter_video.cpp
*
*  Created on: Nov 23, 2010
*      Author: Ethan Rublee
*
*  Modified on: April 17, 2013
*      Author: Kevin Hughes
*
* A starter sample for using OpenCV VideoCapture with capture devices, video files or image sequences
* easy as CV_PI right?
*/

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;

//hide the local functions in an anon namespace
namespace {
	void stereo_Gray2Color(CvMat* gray_mat, CvMat* color_mat)
	{
		if (color_mat)
			cvZero(color_mat);

		int stype = CV_MAT_TYPE(gray_mat->type), dtype = CV_MAT_TYPE(color_mat->type);
		int rows = gray_mat->rows, cols = gray_mat->cols;
		// 判断输入的灰度图和输出的伪彩色图是否大小相同、格式是否符合要求
		if (CV_ARE_SIZES_EQ(gray_mat, color_mat) && stype == CV_8UC1 && dtype == CV_8UC3)
		{
			CvMat* red = cvCreateMat(gray_mat->rows, gray_mat->cols, CV_8U);
			CvMat* green = cvCreateMat(gray_mat->rows, gray_mat->cols, CV_8U);
			CvMat* blue = cvCreateMat(gray_mat->rows, gray_mat->cols, CV_8U);
			CvMat* mask = cvCreateMat(gray_mat->rows, gray_mat->cols, CV_8U);
			// 计算各彩色通道的像素值
			cvSubRS(gray_mat, cvScalar(255), blue);                        // blue(I) = 255 - gray(I)
			cvCopy(gray_mat, red);                                                        // red(I) = gray(I)
			cvCopy(gray_mat, green);                                                // green(I) = gray(I),        if gray(I) < 128
			cvCmpS(green, 128, mask, CV_CMP_GE);                        // green(I) = 255 - gray(I), if gray(I) >= 128
			cvSubRS(green, cvScalar(255), green, mask);
			cvConvertScale(green, green, 2.0, 0.0);
			// 合成伪彩色图
			cvMerge(blue, green, red, NULL, color_mat);
			cvReleaseMat(&red);
			cvReleaseMat(&green);
			cvReleaseMat(&blue);
			cvReleaseMat(&mask);
		}
	}
	void calcu_stereo(Mat left, Mat right) {
		cvtColor(left, left, COLOR_RGB2GRAY);
		cvtColor(right, right, COLOR_RGB2GRAY);
		int numberOfDisparities = 64;
		Mat disp, disp8;
		Ptr<StereoBM> bm = StereoBM::create(numberOfDisparities, 15);
		Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, numberOfDisparities,15);
		bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
		//bm->setPreFilterSize(5);
		bm->setPreFilterCap(63);
		bm->setBlockSize(25);
		bm->setNumDisparities(numberOfDisparities);
		bm->setTextureThreshold(20);
		bm->setUniquenessRatio(15);
		bm->setSpeckleWindowSize(100);
		bm->setSpeckleRange(32);
		bm->setDisp12MaxDiff(2);
		Mat left_p, right_p, dispp;
		//copyMakeBorder(left, left_p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
		//copyMakeBorder(right, right_p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
		sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);
		sgbm->setUniquenessRatio(10);
		sgbm->setSpeckleWindowSize(100);
		sgbm->setSpeckleRange(32);
		sgbm->setDisp12MaxDiff(2);
		
		//sgbm->compute(left_p, right_p, dispp);
		//disp = dispp.colRange(numberOfDisparities, left_p.cols);
		bm->compute(left, right, disp);
		
		disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
		
		Mat disp_color(disp8.rows,disp8.cols,  CV_8UC3);
		CvMat t = disp8; 
		CvMat disp_color_c = disp_color;
		stereo_Gray2Color(&t, &disp_color_c);
		imshow("disparity", disp_color);
		//imshow("disparity", disp8);
	}
	int process(VideoCapture capture, Rect validRoi[2], Mat rmap[2][2],bool is_remap = false) {
		int n = 0;
		char filename[200];
		const std::string window_name = "video";
		std::string window_names[2] = { window_name+" left"+"| q or esc to quit",window_name+" right" + "| q or esc to quit" };
		std::cout << "press space to save a picture. q or esc to quit" << std::endl;
		for (int i = 0; i > 2; i++) {
			namedWindow(window_names[i],WINDOW_AUTOSIZE); //resizable window;
		}
		Mat frame[2];
		Rect cuts[2] = { Rect(52,30,549,398),Rect(52,30,549,398), };
		for (;;) {
			Mat frame_all;
			capture >> frame_all;
			//medianBlur(frame_all, frame_all, 3);
			frame[0] = Mat(frame_all, Rect(0, 0, 640, 480));
			frame[1] = Mat(frame_all, Rect(640, 0, 640, 480));
			for (int i = 0; i < 2; i++) { 
				if (frame[i].empty())
					break;
				if (is_remap) {
					remap(frame[i], frame[i], rmap[i][0], rmap[i][1], INTER_LINEAR);
					//frame[i] = Mat(frame[i], cuts[i]);
				}
				imshow(window_names[i], frame[i]);
			}
			calcu_stereo(frame[0], frame[1]);
			char key = (char)waitKey(1); //delay N millis, usually long enough to display and capture input

			switch (key) {
			case 'q':
			case 'Q':
			case 27: //escape key
				return 0;
			case ' ': //Save an image

				sprintf(filename, "data/%s_%.3d.jpg","left",n);
				imwrite(filename, frame[0]);
				std::cout << "Saved " << filename << std::endl;
				sprintf(filename, "data/%s_%.3d.jpg", "right", n++);
				imwrite(filename, frame[1]);
				std::cout << "Saved " << filename << std::endl;
				break;
			default:
				break;
			}
		}
		return 0;
	}
	void read_extrinsic(std::string intrinsic_filename,std::string extrinsic_filename, Rect validRoi[2], Mat rmap[2][2]) {
		FileStorage fs(intrinsic_filename, FileStorage::READ);
		Size imageSize = Size(640, 480);
		Mat M1, D1, M2, D2;
		fs["M1"] >> M1;
		fs["D1"] >> D1;
		fs["M2"] >> M2;
		fs["D2"] >> D2;

		fs.open(extrinsic_filename, FileStorage::READ);
		Mat R, T, R1, P1, R2, P2;
		Mat Q;
		fs["R"] >> R;
		fs["T"] >> T;

		stereoRectify(M1, D1,
			M2, D2,
			imageSize, R, T, R1, R2, P1, P2, Q,
			CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
		
		Mat map11, map12, map21, map22;
		initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
		initUndistortRectifyMap(M2, D2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	}
}

int capture_main(int argc, char* argv[]) {
	VideoCapture capture(0);
	int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	std::cout << "width:" << width << "height" << height<<std::endl;
	std::cout << "mode:" << capture.get(CV_CAP_PROP_MODE);
	Rect validRoi[2];
	Mat rmap[2][2];
	if (!strcmp(argv[0], "fix")) {
		read_extrinsic("params/intrinsics.yml", "params/extrinsics.yml", validRoi, rmap);
		for (int k = 0; k < 2; k++) {
			std::cout << "validRoi" << validRoi[k].x << "," << validRoi[k].y << "," << validRoi[k].width << "," << validRoi[k].height << std::endl;
		}
		return process(capture, validRoi, rmap, true);
	}
	else {
		return process(capture, validRoi, rmap, false);
	}
	return 0;
}

int stereo_main(int ac, char** av);
int calibration_main(int argc, char* argv[]);
int main(int argc, char* argv[]) {
	FileStorage fs("camera_left.yml", FileStorage::READ);
	Mat cameraMatrix, distCoeffs;
	fs["camera_matrix"] >> cameraMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	if (argc > 2) {
		if (!strcmp(argv[1], "capture")) {
			capture_main(argc-2, &(argv[2]));
		}else if(!strcmp(argv[1], "calibration")) {
			calibration_main(argc-1, &(argv[1]));
		}
	}
	else if (argc > 1) {
		capture_main(argc - 1, &(argv[1]));
	}
	else {
		stereo_main(argc, argv);
	}
	//calibration_main(argc, argv);
}