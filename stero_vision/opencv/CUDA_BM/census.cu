#include "internal.hpp"
using namespace cv;
using namespace cv::cuda;
//the threads number of one block
const int WINDOW_H = 5;
const int WINDOW_W = 7;
const int THREADS_H = 16;
const int THREADS_W = 32;
const int WIDTH_SIZE = THREADS_W + WINDOW_W;
const int HEIGHT_SIZE = THREADS_H + WINDOW_H;
const int WINDOW_H_D2 = WINDOW_H / 2;
const int WINDOW_W_D2 = WINDOW_W / 2;
//窗口 = 5X7 存储空间32bit 中心5点去除，用去30bit 剩余两bit未用
__global__
void census_kernel(const uchar* src, uint* dst, int width, int height){
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int pos = x + y * width;
	__shared__ uchar src_s[WIDTH_SIZE*HEIGHT_SIZE];
	if(x < width && y < height)
	{
		//left top side
		{
		    const int lt_y = y - WINDOW_H_D2;
		    const int lt_x = x - WINDOW_W_D2;
		    if (lt_y >= 0 && lt_x >= 0)
		    {
			    src_s[threadIdx.y*WIDTH_SIZE + threadIdx.x] = src[lt_y*width + lt_x];
		    }
	    }
		//right top side
		{
			const int rt_y = y - WINDOW_H_D2;
			const int rt_x = x - WINDOW_W_D2 + blockDim.x;
			if (threadIdx.x + blockDim.x < WIDTH_SIZE && threadIdx.y < HEIGHT_SIZE) {
				if (rt_y >= 0&& rt_x < width) {
					src_s[threadIdx.y*WIDTH_SIZE + threadIdx.x + blockDim.x] = src[rt_y*width + rt_x];
				}
			}
		}
		//left bottom side
		{
		    const int lb_y = y - WINDOW_H_D2 + blockDim.y;
			const int lb_x = x - WINDOW_W_D2;
			if (threadIdx.x < WIDTH_SIZE && threadIdx.y + blockDim.y < HEIGHT_SIZE) {
				if (lb_y < height && lb_x >= 0) {
					src_s[(threadIdx.y + blockDim.y)*WIDTH_SIZE + threadIdx.x] = src[lb_y*width + lb_x];
				}
			}
		}
	    //right bottom side
	    {
			const int rb_y = y - WINDOW_H_D2 + blockDim.y;
			const int rb_x = x - WINDOW_W_D2 + blockDim.x;
			if (threadIdx.x + blockDim.x < WIDTH_SIZE && threadIdx.y + blockDim.y < HEIGHT_SIZE) {
				if (rb_y < height && rb_x < width) {
					src_s[(threadIdx.y + blockDim.y)*WIDTH_SIZE + threadIdx.x + blockDim.x] = src[rb_y*width + rb_x];
				}
			}
		}
		__syncthreads();
		//计算census
	    if (y >= WINDOW_H_D2 && y < height - WINDOW_H_D2 && x >=WINDOW_W_D2 && x < width - WINDOW_W_D2){
	        const int y_running = threadIdx.y + WINDOW_H_D2;
			const int x_running = threadIdx.x + WINDOW_W_D2;
			const int offset = x_running + y_running * WIDTH_SIZE;
		    //五点平均值
			const int c = ((int)src_s[offset]+src_s[offset-1]+src_s[offset+1]+src_s[offset-WIDTH_SIZE]+src_s[offset+WIDTH_SIZE])/5;
		    uint value=0;
		    //计算中心点上方的census
		    for (int y = -WINDOW_H_D2; y < -1; y++) {
		        const int p_y =  WIDTH_SIZE*(y_running + y);
				for (int x = -WINDOW_W_D2; x <= WINDOW_W_D2; x++) {
					uint result = (c - src_s[p_y + x_running + x]) > 0;
					value <<= 1;
					value += result;
				}
			}
			
			//计算中心点上方的census y = -1
			{
		        const int p_y =  WIDTH_SIZE*(y_running -1);
		        //x < 0
				for (int x = -WINDOW_W_D2; x < 0; x++) {
					uint result = (c - src_s[p_y + x_running + x]) > 0;
					value <<= 1;
					value += result;
				}
				//x > 0
				for (int x = 1; x <=WINDOW_W_D2; x++) {
					uint result = (c - src_s[p_y + x_running + x]) > 0;
					value <<= 1;
					value += result;
				}
			}
			
			//计算中心点census y = 0
			{
		        const int p_y =  WIDTH_SIZE*(y_running);
		        //x < 0
				for (int x = -WINDOW_W_D2; x < -1; x++) {
					uint result = (c - src_s[p_y + x_running + x]) > 0;
					value <<= 1;
					value += result;
				}
				//x > 0
				for (int x = 2; x <=WINDOW_W_D2; x++) {
					uint result = (c - src_s[p_y + x_running + x]) > 0;
					value <<= 1;
					value += result;
				}
			}
			
			//计算中心点下方的census y = 1
			{
		        const int p_y =  WIDTH_SIZE*(y_running + 1);
		        //x < 0
				for (int x = -WINDOW_W_D2; x < 0; x++) {
					uint result = (c - src_s[p_y + x_running + x]) > 0;
					value <<= 1;
					value += result;
				}
				//x > 0
				for (int x = 1; x <=WINDOW_W_D2; x++) {
					uint result = (c - src_s[p_y + x_running + x]) > 0;
					value <<= 1;
					value += result;
				}
			}
			//计算中心点下方的census
		    for (int y = 2; y <= WINDOW_H_D2; y++) {
		        const int p_y =  WIDTH_SIZE*(y_running + y);
				for (int x = -WINDOW_W_D2; x <= WINDOW_W_D2; x++) {
					uint result = (c - src_s[p_y + x_running + x]) > 0;
					value <<= 1;
					value += result;
				}
			}
			dst[pos] = value;
	    }
	}

};

__global__ void diff_census_kernel(const uint* left_census,const uint*  right_census,uchar *cost_array)
{
	__shared__ uint right_s[HEIGHT_SINGLE][WIDTH];
	uint value_left[HEIGHT_SINGLE];
    const int y_base = (blockIdx.y*blockDim.y+threadIdx.y)*HEIGHT_SINGLE;
	const int x = (blockIdx.x*blockDim.x+threadIdx.x)+D_MAX;
	if (y_base < HEIGHT){
		#pragma unroll
		for(int y=0;y<HEIGHT_SINGLE;y++){
			const int pos_left = (y_base+y) * WIDTH + x;
			value_left[y] = left_census[pos_left];
			right_s[y][x] = right_census[pos_left];
			if(x < 2 * D_MAX){
				right_s[y][(x-D_MAX)] = right_census[pos_left-D_MAX];
			}
		}
		__syncthreads();
		#pragma unroll
		for(int disparity=0;disparity<D_MAX;disparity++){
		    for(int y=0;y<HEIGHT_SINGLE;y++){			
			    const int pos_left = (y_base+y) * WIDTH + x;
			    const int pos_right = x-disparity;
			    const uint value_left_t = value_left[y];
			    const uint value_right_t = right_s[y][x-disparity];
			    if(x < WIDTH && y_base+y < HEIGHT) {
				    cost_array[pos_left+FRAME_SIZE*disparity] = (__popc(value_left_t^value_right_t));
				}
			}
		}
	}
}


__global__
void diff_census(){
	__popc(0);
};

extern "C" void census_caller(const uchar* src,uint* dst,int width, int height){
    const dim3   block(THREADS_W, THREADS_H);
    const dim3   grid(CELL_DIV(width,THREADS_W),CELL_DIV(height,THREADS_H));
    //printf("width:%d,height:%d\r\n",width,height);
    census_kernel<<<grid,block>>>(src,dst,width,height);
}
