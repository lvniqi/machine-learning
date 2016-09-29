
#pragma once

#include <opencv2/core/cuda.hpp>
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

#define D_MAX 64
#define WIDTH 640
#define HEIGHT 360
#define HEIGHT_SINGLE 10
#define WINDOW_SIZE_2 7
#define FRAME_SIZE (WIDTH*HEIGHT)
#define COSTS(i,j,d) cost_array[(i)*D_MAX+(j)*WIDTH*D_MAX+(d)]
#define LEFT_IMAGE(i,j) left_image[(i)+(j)*WIDTH]
#define RIGHT_IMAGE(i,j) right_image[(i)+(j)*WIDTH]
#define CELL_DIV(x,BASE) (((x)+(BASE)-1) / (BASE))

using namespace cv;
using namespace cv::cuda;
typedef uchar PixType;
typedef short CostType;
typedef short DispType;
extern "C" void diff_caller(const uchar* h_left,uchar* h_left_texture,uchar* d_left_texture,
                            const uchar* left,const uchar* right, const uchar* left_box,const uchar* right_box, 
                            short* h_dp_array,short* dp_array,
                            uchar* dst,
                            uchar* cost_array,
                            uchar* cost_census_array,
                            uchar* h_cost_sum_array,uchar* cost_sum_array,
                            uint* left_census,
                            uint* right_census,
                            int width, int height);
                        
extern "C" void census_caller(const uchar* src,uint* dst,int width, int height);

void low_texture_detection(const uchar* image,uchar* result,const int window_size);


void diff_c(uchar* left,uchar* right,uchar* cost_array,int width,int height,int disparity);
void census(uchar* src,uint* census_array,int width,int height);

template<typename ARRAY_TYPE>
bool cmp_array(ARRAY_TYPE* a_1,ARRAY_TYPE* a_2,int len){
    bool find_error = false;
    for(int i=0;i<len;i++){
        if(a_1[i] != a_2[i]){
            find_error = true;
            printf("find error: %d != %d ",a_1[i],a_2[i]);
		    printf("pos: (%d)\r\n",i);
        }
    }
    return find_error;
}

//dp1 left to right
void dp_l2r(short* dp,uchar* low_texture,uchar* cost_array,short P1,short P2);

