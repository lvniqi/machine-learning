#include "internal.hpp"
using namespace cv;
using namespace cv::cuda;

const int THREADS_H = 16;
const int THREADS_W = 32;
const int WINDOW_H = 5;
const int WINDOW_W = 7;
const int P1 = 25;
const int P2 = 100;
typedef short cost_type;
__device__ cost_type lr_array[8*WIDTH*HEIGHT*D_MAX];


__global__ void diff_census_kernel(const uint* left_census,const uint*  right_census,uchar *cost_array);



namespace mycuda{
    __device__ short min(short x1,short x2){
        if(x1>=x2){
            return x2;
        }
        return x1;
    }
    __device__ short max(short x1,short x2){
        if(x1<=x2){
            return x2;
        }
        return x1;
    }
}

/* kernels */
//display_range threads per block * grid( nx * ny )
__global__ void diff_kernel(const uchar* left,const uchar* __restrict__ right,uchar *cost_array)
{
	__shared__ uchar right_s[HEIGHT_SINGLE][2*WIDTH+2];
	uchar value_left[HEIGHT_SINGLE];
    const int y_base = (blockIdx.y*blockDim.y+threadIdx.y)*HEIGHT_SINGLE;
	const int x = (blockIdx.x*blockDim.x+threadIdx.x)+D_MAX;
	const int x_s = x<<1;
	if (y_base < HEIGHT){
		#pragma unroll
		for(int y=0;y<HEIGHT_SINGLE;y++){
			const int pos_left = (y_base+y) * WIDTH + x;
			value_left[y] = left[pos_left];
			right_s[y][x_s+1] = right[pos_left];
			if(x == WIDTH-1){
				right_s[y][x_s+2] = right[pos_left];
			}else{
				right_s[y][x_s+2] = (right[pos_left]+right[pos_left+1])/2;
			}
			if(x < 2 * D_MAX){
				right_s[y][((x - D_MAX)<<1)+1] = right[pos_left - D_MAX];
				if(x == D_MAX){
					right_s[y][0] = right[pos_left - D_MAX];
				}
				right_s[y][((x - D_MAX)<<1)+2] = (right[pos_left - D_MAX]+right[pos_left - D_MAX+1])/2;
			}
		}
		__syncthreads();
		int right_pos,pos_l,pos_m,pos_r,right_max,right_min;
		#pragma unroll
		for(int disparity=0;disparity<D_MAX;disparity++){
		    for(int y=0;y<HEIGHT_SINGLE;y++){			
			    const int pos_left = (y_base+y) * WIDTH + x;
			    const int value_left_t = value_left[y];
			    if(x < WIDTH && y_base+y < HEIGHT) {
				    //#pragma unroll
					right_pos = ((x-disparity)<<1);
					pos_l = right_s[y][right_pos-1];
					pos_m = right_s[y][right_pos];
					pos_r = right_s[y][right_pos+1];
	
					right_min = min(pos_m,pos_l);
					right_min = min(right_min,pos_r);
					right_max = max(pos_m,pos_l);
					right_max = max(right_max,pos_r);

				    int diff = max(0,value_left_t-right_max);
				    diff = max(diff,right_min-value_left_t);
				    cost_array[pos_left+FRAME_SIZE*disparity] = diff;
				}
			}
		}
	}
}

//测试 单一路径规划
__global__ void dp_1_kernel(short* dp_array,uchar* cost_all){
    int x = blockIdx.x*blockDim.x+threadIdx.x+D_MAX;
    //__shared__ int disparity_s[6*2*WIDTH*D_MAX];
    short last_line_disparity[2][D_MAX];
    short *last_line_disparity_now = last_line_disparity[0];
    short *last_line_disparity_last = last_line_disparity[1];
    
    
    //上次最佳代价位置
    short last_best_disparity_pos = 0;
    //上次最佳代价值
    short last_best_disparity_value = 0;
    //最佳代价位置
    short now_best_disparity_pos = 0;
    //最佳代价值
    short now_best_disparity_value = 0;
    if(x < WIDTH){
        //y = 0
        {
            //d = 0
            short cost = cost_all[x];
            last_line_disparity_last[0] = cost;
            dp_array[x] = cost;
            last_best_disparity_pos = 0;
            last_best_disparity_value = cost;
            //d > 0
            #pragma unroll
            for(int d=1;d<D_MAX;d++){
                //当前代价
                short cost = cost_all[x+d*FRAME_SIZE];
                last_line_disparity_last[d] = cost;
                dp_array[x+FRAME_SIZE*d] = cost;
                //选取最好的保存
                if(cost < last_best_disparity_value){
                    last_best_disparity_pos = d;
                    last_best_disparity_value = cost;
                }
            }
        }
        //y > 0
        #pragma unroll
        for(int y=1;y<HEIGHT;y++){
            const int pos = y * WIDTH + x;
            //d = 0
            {
                short cost = cost_all[pos];
                //no left disparity
                ///int left_t = P1+last_line_disparity[d-1];
                //mid
                short mid_t = last_line_disparity_last[0];
                //right
                short right_t = last_line_disparity_last[0+1]+P1;
                //last min
                short last_t = last_best_disparity_value+P2;
                //get min of t
                short min_t = mycuda::min(mid_t,mycuda::min(right_t,last_t));
                short d0_value = cost+min_t-last_best_disparity_value;
                dp_array[pos] = d0_value;
                now_best_disparity_pos = 0;
                now_best_disparity_value = d0_value;
                last_line_disparity_now[0] = d0_value;
            }
            #pragma unroll
            for(int d=1;d<D_MAX-1;d++){
                int pos_d = pos+FRAME_SIZE*d;
                short cost = cost_all[pos_d];
                short left_t = last_line_disparity_last[d-1]+P1;
                short mid_t = last_line_disparity_last[d];
                short right_t = last_line_disparity_last[d+1]+P1;
                short last_t = last_best_disparity_value+P2;
                //get min of t
                int min_t = mycuda::min(mid_t,mycuda::min(left_t,mycuda::min(right_t,last_t)));
                short di_value = cost+min_t-last_best_disparity_value;
                dp_array[pos_d] = di_value;
                last_line_disparity_now[d] = di_value;
                if(di_value < now_best_disparity_value){
                    now_best_disparity_pos = d;
                    now_best_disparity_value = di_value;
                }
            }
            //d = D_MAX-1
            {
                int d = D_MAX-1;
                int pos_d = pos+FRAME_SIZE*(D_MAX-1);
                short cost = cost_all[pos_d];
                //no left disparity
                short left_t = last_line_disparity_last[d-1]+P1;
                //mid
                short mid_t = last_line_disparity_last[d];
                //last min
                short last_t = last_best_disparity_value+P2;
                //get min of t
                short min_t = mycuda::min(mid_t,mycuda::min(left_t,last_t));
                short de_value = cost+min_t-last_best_disparity_value;
                dp_array[pos_d] = de_value;
                last_line_disparity_now[d] = de_value;
                if(de_value < last_best_disparity_value){
                    now_best_disparity_pos = d;
                    now_best_disparity_value = de_value;
                }
            }
            //swap
            {
                short *p_t = last_line_disparity_now;
                last_line_disparity_now = last_line_disparity_last;
                last_line_disparity_last = p_t;
                last_best_disparity_pos = now_best_disparity_pos;
                last_best_disparity_value = now_best_disparity_value;
            }
        }
    }
}

//测试 单一路径规划
__global__ void dp_2_kernel(short* dp_array,uchar* cost_all){
    int x = blockIdx.x*blockDim.x+threadIdx.x+D_MAX;
    //__shared__ int disparity_s[6*2*WIDTH*D_MAX];
    short last_line_disparity[2][D_MAX];
    short *last_line_disparity_now = last_line_disparity[0];
    short *last_line_disparity_last = last_line_disparity[1];
    
    
    //上次最佳代价位置
    short last_best_disparity_pos = 0;
    //上次最佳代价值
    short last_best_disparity_value = 0;
    //最佳代价位置
    short now_best_disparity_pos = 0;
    //最佳代价值
    short now_best_disparity_value = 0;
    if(x < WIDTH){
        //y = 0
        {
            //d = 0
            int pos = x+(D_MAX-1)*WIDTH;
            short cost = cost_all[pos];
            last_line_disparity_last[0] = cost;
            dp_array[pos] = cost;
            last_best_disparity_pos = 0;
            last_best_disparity_value = cost;
            //d > 0
            #pragma unroll
            for(int d=1;d<D_MAX;d++){
                //当前代价
                short cost = cost_all[pos];
                last_line_disparity_last[d] = cost;
                dp_array[pos+FRAME_SIZE*d] = cost;
                //选取最好的保存
                if(cost < last_best_disparity_value){
                    last_best_disparity_pos = d;
                    last_best_disparity_value = cost;
                }
            }
        }
        //y > 0
        #pragma unroll
        for(int y=HEIGHT-2;y>=0;y--){
            const int pos = y * WIDTH + x;
            //d = 0
            {
                short cost = cost_all[pos];
                //no left disparity
                ///int left_t = P1+last_line_disparity[d-1];
                //mid
                short mid_t = last_line_disparity_last[0];
                //right
                short right_t = last_line_disparity_last[0+1]+P1;
                //last min
                short last_t = last_best_disparity_value+P2;
                //get min of t
                short min_t = mycuda::min(mid_t,mycuda::min(right_t,last_t));
                short d0_value = cost+min_t-last_best_disparity_value;
                dp_array[pos] = d0_value;
                now_best_disparity_pos = 0;
                now_best_disparity_value = d0_value;
                last_line_disparity_now[0] = d0_value;
            }
            #pragma unroll
            for(int d=1;d<D_MAX-1;d++){
                int pos_d = pos+FRAME_SIZE*d;
                short cost = cost_all[pos_d];
                short left_t = last_line_disparity_last[d-1]+P1;
                short mid_t = last_line_disparity_last[d];
                short right_t = last_line_disparity_last[d+1]+P1;
                short last_t = last_best_disparity_value+P2;
                //get min of t
                int min_t = mycuda::min(mid_t,mycuda::min(left_t,mycuda::min(right_t,last_t)));
                short di_value = cost+min_t-last_best_disparity_value;
                dp_array[pos_d] = di_value;
                last_line_disparity_now[d] = di_value;
                if(di_value < now_best_disparity_value){
                    now_best_disparity_pos = d;
                    now_best_disparity_value = di_value;
                }
            }
            //d = D_MAX-1
            {
                int d = D_MAX-1;
                int pos_d = pos+FRAME_SIZE*(D_MAX-1);
                short cost = cost_all[pos_d];
                //no left disparity
                short left_t = last_line_disparity_last[d-1]+P1;
                //mid
                short mid_t = last_line_disparity_last[d];
                //last min
                short last_t = last_best_disparity_value+P2;
                //get min of t
                short min_t = mycuda::min(mid_t,mycuda::min(left_t,last_t));
                short de_value = cost+min_t-last_best_disparity_value;
                dp_array[pos_d] = de_value;
                last_line_disparity_now[d] = de_value;
                if(de_value < last_best_disparity_value){
                    now_best_disparity_pos = d;
                    now_best_disparity_value = de_value;
                }
            }
            //swap
            {
                short *p_t = last_line_disparity_now;
                last_line_disparity_now = last_line_disparity_last;
                last_line_disparity_last = p_t;
                last_best_disparity_pos = now_best_disparity_pos;
                last_best_disparity_value = now_best_disparity_value;
            }
        }
    }
}
//测试 单一路径规划
__global__ void dp_3_kernel(short* dp_array,uchar* cost_all){
    int x = blockIdx.x*blockDim.x+threadIdx.x+D_MAX;
    //__shared__ int disparity_s[6*2*WIDTH*D_MAX];
    short last_line_disparity[2][D_MAX];
    short *last_line_disparity_now = last_line_disparity[0];
    short *last_line_disparity_last = last_line_disparity[1];
    
    
    //上次最佳代价位置
    short last_best_disparity_pos = 0;
    //上次最佳代价值
    short last_best_disparity_value = 0;
    //最佳代价位置
    short now_best_disparity_pos = 0;
    //最佳代价值
    short now_best_disparity_value = 0;
    if(x < WIDTH){
        //y = 0
        {
            //d = 0
            int pos = x+(D_MAX-1)*WIDTH;
            short cost = cost_all[pos];
            last_line_disparity_last[0] = cost;
            dp_array[pos] = cost;
            last_best_disparity_pos = 0;
            last_best_disparity_value = cost;
            //d > 0
            #pragma unroll
            for(int d=1;d<D_MAX;d++){
                //当前代价
                short cost = cost_all[pos];
                last_line_disparity_last[d] = cost;
                dp_array[pos+FRAME_SIZE*d] = cost;
                //选取最好的保存
                if(cost < last_best_disparity_value){
                    last_best_disparity_pos = d;
                    last_best_disparity_value = cost;
                }
            }
        }
        //y > 0
        #pragma unroll
        for(int y=HEIGHT-2;y>=0;y--){
            x--;
            if(x<0){
                return;
            }
            const int pos = y * WIDTH + x;
            //d = 0
            {
                short cost = cost_all[pos];
                //no left disparity
                ///int left_t = P1+last_line_disparity[d-1];
                //mid
                short mid_t = last_line_disparity_last[0];
                //right
                short right_t = last_line_disparity_last[0+1]+P1;
                //last min
                short last_t = last_best_disparity_value+P2;
                //get min of t
                short min_t = mycuda::min(mid_t,mycuda::min(right_t,last_t));
                short d0_value = cost+min_t-last_best_disparity_value;
                dp_array[pos] = d0_value;
                now_best_disparity_pos = 0;
                now_best_disparity_value = d0_value;
                last_line_disparity_now[0] = d0_value;
            }
            #pragma unroll
            for(int d=1;d<D_MAX-1;d++){
                int pos_d = pos+FRAME_SIZE*d;
                short cost = cost_all[pos_d];
                short left_t = last_line_disparity_last[d-1]+P1;
                short mid_t = last_line_disparity_last[d];
                short right_t = last_line_disparity_last[d+1]+P1;
                short last_t = last_best_disparity_value+P2;
                //get min of t
                int min_t = mycuda::min(mid_t,mycuda::min(left_t,mycuda::min(right_t,last_t)));
                short di_value = cost+min_t-last_best_disparity_value;
                dp_array[pos_d] = di_value;
                last_line_disparity_now[d] = di_value;
                if(di_value < now_best_disparity_value){
                    now_best_disparity_pos = d;
                    now_best_disparity_value = di_value;
                }
            }
            //d = D_MAX-1
            {
                int d = D_MAX-1;
                int pos_d = pos+FRAME_SIZE*(D_MAX-1);
                short cost = cost_all[pos_d];
                //no left disparity
                short left_t = last_line_disparity_last[d-1]+P1;
                //mid
                short mid_t = last_line_disparity_last[d];
                //last min
                short last_t = last_best_disparity_value+P2;
                //get min of t
                short min_t = mycuda::min(mid_t,mycuda::min(left_t,last_t));
                short de_value = cost+min_t-last_best_disparity_value;
                dp_array[pos_d] = de_value;
                last_line_disparity_now[d] = de_value;
                if(de_value < last_best_disparity_value){
                    now_best_disparity_pos = d;
                    now_best_disparity_value = de_value;
                }
            }
            //swap
            {
                short *p_t = last_line_disparity_now;
                last_line_disparity_now = last_line_disparity_last;
                last_line_disparity_last = p_t;
                last_best_disparity_pos = now_best_disparity_pos;
                last_best_disparity_value = now_best_disparity_value;
            }
        }
    }
}
__global__ void dp_4_kernel(short* dp_array,uchar* cost_all){
    int x = blockIdx.x*blockDim.x+threadIdx.x+D_MAX;
    //__shared__ int disparity_s[6*2*WIDTH*D_MAX];
    short last_line_disparity[2][D_MAX];
    short *last_line_disparity_now = last_line_disparity[0];
    short *last_line_disparity_last = last_line_disparity[1];
    
    
    //上次最佳代价位置
    short last_best_disparity_pos = 0;
    //上次最佳代价值
    short last_best_disparity_value = 0;
    //最佳代价位置
    short now_best_disparity_pos = 0;
    //最佳代价值
    short now_best_disparity_value = 0;
    if(x < WIDTH){
        //y = 0
        {
            //d = 0
            short cost = cost_all[x];
            last_line_disparity_last[0] = cost;
            dp_array[x] = cost;
            last_best_disparity_pos = 0;
            last_best_disparity_value = cost;
            //d > 0
            #pragma unroll
            for(int d=1;d<D_MAX;d++){
                //当前代价
                short cost = cost_all[x+d*FRAME_SIZE];
                last_line_disparity_last[d] = cost;
                dp_array[x+FRAME_SIZE*d] = cost;
                //选取最好的保存
                if(cost < last_best_disparity_value){
                    last_best_disparity_pos = d;
                    last_best_disparity_value = cost;
                }
            }
        }
        //y > 0
        #pragma unroll
        for(int y=1;y<HEIGHT;y++){
            x++;
            if(x>=WIDTH){
                return;
            }
            const int pos = y * WIDTH + x;
            //d = 0
            {
                short cost = cost_all[pos];
                //no left disparity
                ///int left_t = P1+last_line_disparity[d-1];
                //mid
                short mid_t = last_line_disparity_last[0];
                //right
                short right_t = last_line_disparity_last[0+1]+P1;
                //last min
                short last_t = last_best_disparity_value+P2;
                //get min of t
                short min_t = mycuda::min(mid_t,mycuda::min(right_t,last_t));
                short d0_value = cost+min_t-last_best_disparity_value;
                dp_array[pos] = d0_value;
                now_best_disparity_pos = 0;
                now_best_disparity_value = d0_value;
                last_line_disparity_now[0] = d0_value;
            }
            #pragma unroll
            for(int d=1;d<D_MAX-1;d++){
                int pos_d = pos+FRAME_SIZE*d;
                short cost = cost_all[pos_d];
                short left_t = last_line_disparity_last[d-1]+P1;
                short mid_t = last_line_disparity_last[d];
                short right_t = last_line_disparity_last[d+1]+P1;
                short last_t = last_best_disparity_value+P2;
                //get min of t
                int min_t = mycuda::min(mid_t,mycuda::min(left_t,mycuda::min(right_t,last_t)));
                short di_value = cost+min_t-last_best_disparity_value;
                dp_array[pos_d] = di_value;
                last_line_disparity_now[d] = di_value;
                if(di_value < now_best_disparity_value){
                    now_best_disparity_pos = d;
                    now_best_disparity_value = di_value;
                }
            }
            //d = D_MAX-1
            {
                int d = D_MAX-1;
                int pos_d = pos+FRAME_SIZE*(D_MAX-1);
                short cost = cost_all[pos_d];
                //no left disparity
                short left_t = last_line_disparity_last[d-1]+P1;
                //mid
                short mid_t = last_line_disparity_last[d];
                //last min
                short last_t = last_best_disparity_value+P2;
                //get min of t
                short min_t = mycuda::min(mid_t,mycuda::min(left_t,last_t));
                short de_value = cost+min_t-last_best_disparity_value;
                dp_array[pos_d] = de_value;
                last_line_disparity_now[d] = de_value;
                if(de_value < last_best_disparity_value){
                    now_best_disparity_pos = d;
                    now_best_disparity_value = de_value;
                }
            }
            //swap
            {
                short *p_t = last_line_disparity_now;
                last_line_disparity_now = last_line_disparity_last;
                last_line_disparity_last = p_t;
                last_best_disparity_pos = now_best_disparity_pos;
                last_best_disparity_value = now_best_disparity_value;
            }
        }
    }
}
__global__ void dp_5_kernel(short* dp_array,uchar* cost_all){
    int x = blockIdx.x*blockDim.x+threadIdx.x+D_MAX;
    //__shared__ int disparity_s[6*2*WIDTH*D_MAX];
    short last_line_disparity[2][D_MAX];
    short *last_line_disparity_now = last_line_disparity[0];
    short *last_line_disparity_last = last_line_disparity[1];
    
    
    //上次最佳代价位置
    short last_best_disparity_pos = 0;
    //上次最佳代价值
    short last_best_disparity_value = 0;
    //最佳代价位置
    short now_best_disparity_pos = 0;
    //最佳代价值
    short now_best_disparity_value = 0;
    if(x < WIDTH){
        //y = 0
        {
            //d = 0
            short cost = cost_all[x];
            last_line_disparity_last[0] = cost;
            dp_array[x] = cost;
            last_best_disparity_pos = 0;
            last_best_disparity_value = cost;
            //d > 0
            #pragma unroll
            for(int d=1;d<D_MAX;d++){
                //当前代价
                short cost = cost_all[x+d*FRAME_SIZE];
                last_line_disparity_last[d] = cost;
                dp_array[x+FRAME_SIZE*d] = cost;
                //选取最好的保存
                if(cost < last_best_disparity_value){
                    last_best_disparity_pos = d;
                    last_best_disparity_value = cost;
                }
            }
        }
        //y > 0
        #pragma unroll
        for(int y=1;y<HEIGHT;y++){
            x--;
            if(x<0){
                return;
            }
            const int pos = y * WIDTH + x;
            //d = 0
            {
                short cost = cost_all[pos];
                //no left disparity
                ///int left_t = P1+last_line_disparity[d-1];
                //mid
                short mid_t = last_line_disparity_last[0];
                //right
                short right_t = last_line_disparity_last[0+1]+P1;
                //last min
                short last_t = last_best_disparity_value+P2;
                //get min of t
                short min_t = mycuda::min(mid_t,mycuda::min(right_t,last_t));
                short d0_value = cost+min_t-last_best_disparity_value;
                dp_array[pos] = d0_value;
                now_best_disparity_pos = 0;
                now_best_disparity_value = d0_value;
                last_line_disparity_now[0] = d0_value;
            }
            #pragma unroll
            for(int d=1;d<D_MAX-1;d++){
                int pos_d = pos+FRAME_SIZE*d;
                short cost = cost_all[pos_d];
                short left_t = last_line_disparity_last[d-1]+P1;
                short mid_t = last_line_disparity_last[d];
                short right_t = last_line_disparity_last[d+1]+P1;
                short last_t = last_best_disparity_value+P2;
                //get min of t
                int min_t = mycuda::min(mid_t,mycuda::min(left_t,mycuda::min(right_t,last_t)));
                short di_value = cost+min_t-last_best_disparity_value;
                dp_array[pos_d] = di_value;
                last_line_disparity_now[d] = di_value;
                if(di_value < now_best_disparity_value){
                    now_best_disparity_pos = d;
                    now_best_disparity_value = di_value;
                }
            }
            //d = D_MAX-1
            {
                int d = D_MAX-1;
                int pos_d = pos+FRAME_SIZE*(D_MAX-1);
                short cost = cost_all[pos_d];
                //no left disparity
                short left_t = last_line_disparity_last[d-1]+P1;
                //mid
                short mid_t = last_line_disparity_last[d];
                //last min
                short last_t = last_best_disparity_value+P2;
                //get min of t
                short min_t = mycuda::min(mid_t,mycuda::min(left_t,last_t));
                short de_value = cost+min_t-last_best_disparity_value;
                dp_array[pos_d] = de_value;
                last_line_disparity_now[d] = de_value;
                if(de_value < last_best_disparity_value){
                    now_best_disparity_pos = d;
                    now_best_disparity_value = de_value;
                }
            }
            //swap
            {
                short *p_t = last_line_disparity_now;
                last_line_disparity_now = last_line_disparity_last;
                last_line_disparity_last = p_t;
                last_best_disparity_pos = now_best_disparity_pos;
                last_best_disparity_value = now_best_disparity_value;
            }
        }
    }
}
__global__ void dp_6_kernel(short* dp_array,uchar* cost_all){
    int x = blockIdx.x*blockDim.x+threadIdx.x+D_MAX;
    //__shared__ int disparity_s[6*2*WIDTH*D_MAX];
    short last_line_disparity[2][D_MAX];
    short *last_line_disparity_now = last_line_disparity[0];
    short *last_line_disparity_last = last_line_disparity[1];
    
    
    //上次最佳代价位置
    short last_best_disparity_pos = 0;
    //上次最佳代价值
    short last_best_disparity_value = 0;
    //最佳代价位置
    short now_best_disparity_pos = 0;
    //最佳代价值
    short now_best_disparity_value = 0;
    if(x < WIDTH){
        //y = 0
        {
            //d = 0
            int pos = x+(D_MAX-1)*WIDTH;
            short cost = cost_all[pos];
            last_line_disparity_last[0] = cost;
            dp_array[pos] = cost;
            last_best_disparity_pos = 0;
            last_best_disparity_value = cost;
            //d > 0
            #pragma unroll
            for(int d=1;d<D_MAX;d++){
                //当前代价
                short cost = cost_all[pos];
                last_line_disparity_last[d] = cost;
                dp_array[pos+FRAME_SIZE*d] = cost;
                //选取最好的保存
                if(cost < last_best_disparity_value){
                    last_best_disparity_pos = d;
                    last_best_disparity_value = cost;
                }
            }
        }
        //y > 0
        #pragma unroll
        for(int y=HEIGHT-2;y>=0;y--){
            x++;
            if(x>=WIDTH){
                return;
            }
            const int pos = y * WIDTH + x;
            //d = 0
            {
                short cost = cost_all[pos];
                //no left disparity
                ///int left_t = P1+last_line_disparity[d-1];
                //mid
                short mid_t = last_line_disparity_last[0];
                //right
                short right_t = last_line_disparity_last[0+1]+P1;
                //last min
                short last_t = last_best_disparity_value+P2;
                //get min of t
                short min_t = mycuda::min(mid_t,mycuda::min(right_t,last_t));
                short d0_value = cost+min_t-last_best_disparity_value;
                dp_array[pos] = d0_value;
                now_best_disparity_pos = 0;
                now_best_disparity_value = d0_value;
                last_line_disparity_now[0] = d0_value;
            }
            #pragma unroll
            for(int d=1;d<D_MAX-1;d++){
                int pos_d = pos+FRAME_SIZE*d;
                short cost = cost_all[pos_d];
                short left_t = last_line_disparity_last[d-1]+P1;
                short mid_t = last_line_disparity_last[d];
                short right_t = last_line_disparity_last[d+1]+P1;
                short last_t = last_best_disparity_value+P2;
                //get min of t
                int min_t = mycuda::min(mid_t,mycuda::min(left_t,mycuda::min(right_t,last_t)));
                short di_value = cost+min_t-last_best_disparity_value;
                dp_array[pos_d] = di_value;
                last_line_disparity_now[d] = di_value;
                if(di_value < now_best_disparity_value){
                    now_best_disparity_pos = d;
                    now_best_disparity_value = di_value;
                }
            }
            //d = D_MAX-1
            {
                int d = D_MAX-1;
                int pos_d = pos+FRAME_SIZE*(D_MAX-1);
                short cost = cost_all[pos_d];
                //no left disparity
                short left_t = last_line_disparity_last[d-1]+P1;
                //mid
                short mid_t = last_line_disparity_last[d];
                //last min
                short last_t = last_best_disparity_value+P2;
                //get min of t
                short min_t = mycuda::min(mid_t,mycuda::min(left_t,last_t));
                short de_value = cost+min_t-last_best_disparity_value;
                dp_array[pos_d] = de_value;
                last_line_disparity_now[d] = de_value;
                if(de_value < last_best_disparity_value){
                    now_best_disparity_pos = d;
                    now_best_disparity_value = de_value;
                }
            }
            //swap
            {
                short *p_t = last_line_disparity_now;
                last_line_disparity_now = last_line_disparity_last;
                last_line_disparity_last = p_t;
                last_best_disparity_pos = now_best_disparity_pos;
                last_best_disparity_value = now_best_disparity_value;
            }
        }
    }
}
__global__ void cost_sum_kernel(int width, int height,const uchar* d_left_texture,uchar* cost_array,uchar* cost_census_array,
                                uchar* cost_all)
{
	
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    int x = blockIdx.x*blockDim.x+threadIdx.x+D_MAX;
    int pos = y * width + x;
	if(x < width && y < height) {
	    float texture = d_left_texture[pos];
	    float P1 = texture >= 45?1:texture/45.0;
	    float P2 = 1-P1;
		for(int d=0;d<D_MAX;d++){
		    int pos_d = pos+d*FRAME_SIZE;
		    float cost_census = cost_census_array[pos_d]<<3;
		    float cost_block = cost_array[pos_d];
		    float sum_cost = P1*cost_census+P2*cost_block;
		    cost_all[pos_d] = sum_cost;
		}
	}
}

__global__ void wta_kernel(uchar* dst, int width, int height,short* dp_array,uchar* left_texture)
{
	
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    int x = blockIdx.x*blockDim.x+threadIdx.x+D_MAX;
    int pos = y * width + x;
	if(x < width && y < height) {
		int real_disparity = 0,min_cost = 
		    dp_array[pos]+dp_array[pos+D_MAX*FRAME_SIZE]+
		    dp_array[pos+2*D_MAX*FRAME_SIZE]+dp_array[pos+3*D_MAX*FRAME_SIZE]+
		    dp_array[pos+4*D_MAX*FRAME_SIZE]+dp_array[pos+5*D_MAX*FRAME_SIZE];
	    if(left_texture[pos] >0){
		    for(int d=0;d<D_MAX;d++){
		        int pos_now = pos+width*height*d;
		        int vlaue = 
		            dp_array[pos_now]+dp_array[pos_now+D_MAX*FRAME_SIZE]+
		            dp_array[pos_now+2*D_MAX*FRAME_SIZE]+dp_array[pos_now+3*D_MAX*FRAME_SIZE]+
		            dp_array[pos_now+4*D_MAX*FRAME_SIZE]+dp_array[pos_now+5*D_MAX*FRAME_SIZE];
		        //int vlaue = cost_array[pos+width*height*d];
			    if(vlaue < min_cost){
				    min_cost = vlaue;
				    real_disparity = d;
			    }
		    }
		}
		dst[pos] = real_disparity;
	}
}

__global__ void transpose(uchar *odata,uchar *idata)  
{  
    __shared__ float block[16][16];  
    const int BLOCK_DIM = 16;
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;  
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;  
    if((xIndex < WIDTH) && (yIndex < HEIGHT))  
    {  
        unsigned int index_in = yIndex * WIDTH + xIndex;  
        block[threadIdx.y][threadIdx.x] = idata[index_in];  
    }  
  
    __syncthreads();  
  
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;  
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;  
    if((xIndex < HEIGHT) && (yIndex < WIDTH))  
    {  
        unsigned int index_out = yIndex * HEIGHT + xIndex;  
        odata[index_out] = block[threadIdx.x][threadIdx.y];  
    }  
}

extern "C" void diff_caller(const uchar* h_left,uchar* h_left_texture,uchar* d_left_texture,
                            const uchar* left,const uchar* right, const uchar* left_box,const uchar* right_box, 
                            short* h_dp_array,short* dp_array,
                            uchar* dst,
                            uchar* cost_array,
                            uchar* cost_census_array,
                            uchar* h_cost_sum_array,uchar* cost_sum_array,
                            uint* left_census,
                            uint* right_census,
                            int width, int height)
{
    census_caller(left,left_census,width,height);
    //cudaThreadSynchronize();
    census_caller(right,right_census,width,height);
    //cudaThreadSynchronize();
    {
		dim3 block(width-D_MAX,1);
		dim3 grid(1,CELL_DIV(height,block.y*HEIGHT_SINGLE));
		diff_kernel<<<grid,block>>>(left_box,right_box,cost_array);
		//cudaThreadSynchronize();
	}
    low_texture_detection(h_left,h_left_texture,7);
    cudaThreadSynchronize();
    {
		dim3 block(width-D_MAX,1);
		dim3 grid(1,CELL_DIV(height,(HEIGHT_SINGLE*block.y)));
		diff_census_kernel<<<grid,block>>>(left_census,right_census,cost_census_array);
		cudaThreadSynchronize();
	}
	
	/*{
		dim3 block(D_MAX,16);
		dim3 grid(1,CELL_DIV(height,block.y));
		sum_column_kernel<<<grid,block>>>(cost_array);
		cudaThreadSynchronize();
	}
	{
		dim3 block(1,WIDTH);
		dim3 grid(D_MAX,1);
		sum_row_kernel<<<grid,block>>>(cost_array);
		cudaThreadSynchronize();
	}*/
	{
		dim3 block(32,16);
		dim3 grid(CELL_DIV(width-D_MAX,block.x),CELL_DIV(height,block.y));
	    cost_sum_kernel<<<grid,block>>>(width,height,d_left_texture,cost_array,cost_census_array,cost_sum_array);
	    cudaThreadSynchronize();
    }
    //dp_l2r(h_dp_array,h_left_texture,h_cost_sum_array,25,100);
	{
	    dim3 block(WIDTH,1);
		dim3 grid(CELL_DIV(width-D_MAX,block.x),1);
		dp_1_kernel<<<grid,block>>>(dp_array,cost_sum_array);
        dp_2_kernel<<<grid,block>>>(dp_array+D_MAX*FRAME_SIZE,cost_sum_array);
        dp_3_kernel<<<grid,block>>>(dp_array+2*D_MAX*FRAME_SIZE,cost_sum_array);
        dp_4_kernel<<<grid,block>>>(dp_array+3*D_MAX*FRAME_SIZE,cost_sum_array);
        dp_5_kernel<<<grid,block>>>(dp_array+4*D_MAX*FRAME_SIZE,cost_sum_array);
        dp_6_kernel<<<grid,block>>>(dp_array+5*D_MAX*FRAME_SIZE,cost_sum_array);
		cudaThreadSynchronize();
	}
	{
		dim3 block(32,16);
		dim3 grid(CELL_DIV(width-D_MAX,block.x),CELL_DIV(height,block.y));
		wta_kernel<<<grid,block>>>(dst,width,height,dp_array,d_left_texture);

		cudaThreadSynchronize();
	}
}
