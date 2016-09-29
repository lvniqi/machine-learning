#include "opencv2/core.hpp"

#include "opencv2/cudastereo.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudawarping.hpp"

#include "internal.hpp"
#include <iostream>

//#define TEST
//#define CHECK_RESULT
#define TEST_COUNT 100

using namespace cv;
using namespace cv::cuda;
using namespace std;

void read_extrinsic(std::string intrinsic_filename,std::string extrinsic_filename, 
		Rect validRoi[2], Mat rmap[2][2]) {
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
	initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_32F, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(M2, D2, R2, P2, imageSize, CV_32F, rmap[1][0], rmap[1][1]);

}

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


void drawText(Mat & frame_all);

uchar test_cost_array[WIDTH*HEIGHT*sizeof(uchar)*D_MAX];
uint  test_census_array[WIDTH*HEIGHT];
const char* keys =
{
    "{t  test   | false | use taet benchmark or not}",
};

int main(int argc, const char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    bool test = parser.get<bool>("test");
    if(test){
        cout<<"start benchmark"<<endl;
    }
    cout << "Built with OpenCV " << CV_VERSION << endl;
    Mat frame_all;
    VideoCapture capture;
    capture.open(0);
    //malloc mem
    size_t memSize = WIDTH*HEIGHT*sizeof(uchar);
    cuda::GpuMat src_l,src_r, dst_l,dst_r;
    cuda::GpuMat d_src, d_dst_t, d_xmap, d_ymap;
    uchar* d_left;
    uchar* d_left_texture;
    uchar* left_texture;
    uchar* d_right;
    uchar* d_left_box;
    uchar* d_right_box;
    
    short* dp_array;
    short* d_dp_array;
    
    uchar* d_dst;
    
	uchar*  d_cost_array;
	uchar*  cost_array;
	uchar*  d_cost_census_array;
	uchar*  cost_census_array;
	
	uchar*  d_cost_sum_array;
	uchar*  cost_sum_array;
	
	uint* d_left_census;
	uint* d_right_census;
	uint* left_census;
	uint* right_census;

    cudaMalloc((void**)&d_left,memSize);
    cudaMalloc((void**)&d_right,memSize);
    cudaMalloc((void**)&d_left_box,memSize);
    cudaMalloc((void**)&d_right_box,memSize);
    cudaMalloc((void**)&d_dst,memSize);
    
    cudaHostAlloc((void**)&dp_array,WIDTH*HEIGHT*sizeof(short)*D_MAX*8,cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **)&d_dp_array, (void *) dp_array, 0);
    
    cudaHostAlloc((void**)&left_texture,memSize,cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **)&d_left_texture, (void *)  left_texture, 0);
    
    cudaHostAlloc((void**)&cost_array,WIDTH*HEIGHT*sizeof(uchar)*D_MAX,cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **)&d_cost_array, (void *)  cost_array, 0);
    cudaHostAlloc((void**)&cost_census_array,WIDTH*HEIGHT*sizeof(uchar)*D_MAX,cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **)&d_cost_census_array, (void *)  cost_census_array, 0);
    cudaHostAlloc((void**)&cost_sum_array,WIDTH*HEIGHT*sizeof(uchar)*D_MAX,cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **)&d_cost_sum_array, (void *)  cost_sum_array, 0);
    
    cudaHostAlloc((void**)&left_census,WIDTH*HEIGHT*sizeof(uint),cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **)&d_left_census, (void *) left_census, 0);
    cudaHostAlloc((void**)&right_census,WIDTH*HEIGHT*sizeof(uint),cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **)&d_right_census, (void *) right_census, 0);
    //load params
    Rect validRoi[2];
    Mat rmap[2][2];
	read_extrinsic("params/intrinsics.yml", "params/extrinsics.yml", validRoi, rmap);
    cv::setUseOptimized(true);
	std::cout << "CV_CPU_SSE2:" << cv::checkHardwareSupport(CV_CPU_SSE2) << std::endl;;
	std::cout << "CV_CPU_NEON:" << cv::checkHardwareSupport(CV_CPU_NEON) << std::endl;;
    if(capture.isOpened())
    {
        cout << "Capture is opened" << endl;
        capture.set(CV_CAP_PROP_FRAME_WIDTH,1280);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT,480);
        int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	      int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
        std::cout << "width:" << width << "height" << height<<std::endl;
        double t,tt;
        #ifdef TEST
	      for(int j=0;j<10;j++)
        #else
        for(;;)
        #endif
        {
            tt = t;
            Mat frame[2];
            Mat frame_box[2];
            t = (double)cvGetTickCount();
            int time_ms = ((t-tt)/(cvGetTickFrequency()*1000));
            printf( "use time: %d ms\r\n",time_ms);
            printf( "frame rate: %d fps\r\n",(1000*TEST_COUNT/time_ms));
            capture >> frame_all;
            if(frame_all.empty())
                break;
            //drawText(frame_all);
            //imshow("Sample", frame_all);
	          //cout<<"frame_all.rows"<<frame_all.rows<<endl;
            Mat image2 = Mat(HEIGHT,WIDTH,CV_8UC1);
            #ifdef TEST
            for(int i=0;i<TEST_COUNT;i++)
            #endif
            {
                frame[0] = Mat(frame_all, Rect(0, 0, 640, 480));
                frame[1] = Mat(frame_all, Rect(640, 0, 640, 480));
				cvtColor(frame[0], frame[0], COLOR_BGR2GRAY);
				cvtColor(frame[1], frame[1], COLOR_BGR2GRAY);
				for (int i = 0; i < 2; i++) { 
					if (frame[i].empty())
						break;
					/*{
					    d_src.upload(frame[i]);
					    d_xmap.upload(rmap[i][0]);
					    d_ymap.upload(rmap[i][1]);
					    cuda::remap(d_src, d_dst_t, d_xmap, d_ymap, INTER_LINEAR);
					    frame[i] = Mat(d_dst_t);
					}*/
					cv::remap(frame[i], frame[i], rmap[i][0], rmap[i][1], INTER_LINEAR);
					frame[i] = Mat(frame[i], Rect(0,87,WIDTH,HEIGHT));
					{
				        blur(frame[i],frame_box[i],Size(9,9));
                    }
				}
                /*{
				    src_l.upload(frame[0]),src_r.upload(frame[1]);
                    Ptr<cuda::Filter> openFilter_l = cuda::createBoxFilter(CV_8UC1, CV_8UC1,Size(15,15));
                    Ptr<cuda::Filter> openFilter_r = cuda::createBoxFilter(CV_8UC1, CV_8UC1,Size(15,15));
                    openFilter_l->apply(src_l, dst_l);
                    openFilter_r->apply(src_r, dst_r);
                    frame_box[0] = Mat(dst_l);
                    frame_box[1] = Mat(dst_r);
                }*/
                cudaMemcpy(d_left,frame[0].data,memSize,cudaMemcpyHostToDevice);
                cudaMemcpy(d_right,frame[1].data,memSize,cudaMemcpyHostToDevice);
                cudaMemcpy(d_left_box,frame_box[0].data,memSize,cudaMemcpyHostToDevice);
                cudaMemcpy(d_right_box,frame_box[1].data,memSize,cudaMemcpyHostToDevice);
                
                diff_caller(frame[0].data,left_texture,d_left_texture,
                    d_left,d_right,d_left_box,d_right_box,
                    dp_array,d_dp_array,
                    d_dst,
                    d_cost_array,d_cost_census_array,
                    cost_sum_array,d_cost_sum_array,
                    d_left_census,d_right_census,WIDTH,HEIGHT);
                cudaMemcpy(image2.data,d_dst,memSize,cudaMemcpyDeviceToHost);
				medianBlur(image2,image2,3);
				#ifdef CHECK_RESULT
				cout<<"CHECK_RESULT"<<endl;
				diff_c(frame[0].data,frame[1].data,test_cost_array,WIDTH,HEIGHT,D_MAX);
				census(frame[0].data,test_census_array,WIDTH,HEIGHT);
				bool error_flag_census = cmp_array<uint>(test_census_array,left_census,WIDTH*HEIGHT);
				bool error_flag = cmp_array<uchar>(test_cost_array,cost_array,WIDTH*HEIGHT*D_MAX);
				/*
				for(int i=0;i<640;i++){
				    for(int j=0;j<480;j++){
				        if(test_census_array[i+j*640] != left_census[i+j*640]){
				            printf("find error: %x != %x ",test_census_array[i+j*640],left_census[i+j*640]);
							printf("(%d,%d)\r\n",i,j);
				        }
				    }
				}
				bool error_flag = false;
				for(int d=0;d<D_MAX;d++){
					for(int row=0;row<480;row++){
						for(int column=D_MAX;column<640;column++){
							int pos = (row*640+column);
							if(test_cost_array[d*FRAME_SIZE+pos] != cost_array[d*FRAME_SIZE+pos]){
								error_flag = true;
								printf("find error: %d != %d ",test_cost_array[d*FRAME_SIZE+pos],cost_array[d*FRAME_SIZE+pos]);
								printf("(%d,%d,%d)\r\n",d,row,column);
							}
						}
					}
				}*/
				if(not error_flag and not error_flag_census){
					printf("CHECK SUCCESS!\r\n");
				}
				#endif
				
			
                //diff_caller(d_left,d_right,d_dst_cuda,640,480);
                //memcpy(image2.data, d_dst, memSize);
            }
            #ifndef TEST

			Mat disp8;
			image2.convertTo(disp8, CV_8U, 255 / (64.0*1.0));
			Mat disp_color(image2.rows,image2.cols,  CV_8UC3);
			CvMat t = disp8; 
			CvMat disp_color_c = disp_color;
			stereo_Gray2Color(&t, &disp_color_c);

            imshow("gpu",disp_color);
            imshow("left",frame[0]);
            #endif
            if(waitKey(1) >= 0)
                break;
        }
    }
    else
    {
        cout << "No capture" << endl;
        frame_all = Mat::zeros(480, 640, CV_8UC1);
        drawText(frame_all);
        imshow("Sample", frame_all);
        waitKey(0);
    }
    cudaFree((void**)&d_left);
    cudaFree((void**)&d_right);
    cudaFree((void**)&d_dst);
    return 0;
}

void drawText(Mat & image)
{
    putText(image, "Hello OpenCV",
            Point(20, 50),
            FONT_HERSHEY_COMPLEX, 1, // font face and scale
            Scalar(255, 255, 255), // white
            1, LINE_AA); // line thickness and type
}
