#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "internal.hpp"
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;

/*static void calcPixelCostBT( const Mat& img1, const Mat& img2,
							CostType* cost,
                            PixType* buffer, const PixType* tab,
                            int tabOfs, int )
{
    int x, c, width = img1.cols, cn = img1.channels();
    int minX1 = std::max(maxD, 0), maxX1 = width + std::min(minD, 0);
    int minX2 = std::max(minX1 - maxD, 0), maxX2 = std::min(maxX1 - minD, width);
    int D = maxD - minD, width1 = maxX1 - minX1, width2 = maxX2 - minX2;
    const PixType *row1 = img1.ptr<PixType>(y), *row2 = img2.ptr<PixType>(y);
    PixType *prow1 = buffer + width2*2, *prow2 = prow1 + width*cn*2;

    tab += tabOfs;

    for( c = 0; c < cn*2; c++ )
    {
        prow1[width*c] = prow1[width*c + width-1] =
        prow2[width*c] = prow2[width*c + width-1] = tab[0];
    }

    int n1 = y > 0 ? -(int)img1.step : 0, s1 = y < img1.rows-1 ? (int)img1.step : 0;
    int n2 = y > 0 ? -(int)img2.step : 0, s2 = y < img2.rows-1 ? (int)img2.step : 0;


    for( x = 1; x < width-1; x++ )
    {
        prow1[x] = tab[(row1[x+1] - row1[x-1])*2 + row1[x+n1+1] - row1[x+n1-1] + row1[x+s1+1] - row1[x+s1-1]];
        prow2[width-1-x] = tab[(row2[x+1] - row2[x-1])*2 + row2[x+n2+1] - row2[x+n2-1] + row2[x+s2+1] - row2[x+s2-1]];

        prow1[x+width] = row1[x];
        prow2[width-1-x+width] = row2[x];
    }

    memset( cost, 0, width1*D*sizeof(cost[0]) );

    buffer -= minX2;
    cost -= minX1*D + minD; // simplify the cost indices inside the loop

    for( c = 0; c < cn*2; c++, prow1 += width, prow2 += width )
    {
        int diff_scale = c < cn ? 0 : 2;

        // precompute
        //   v0 = min(row2[x-1/2], row2[x], row2[x+1/2]) and
        //   v0 = max(row2[x-1/2], row2[x], row2[x+1/2]) and
        for( x = minX2; x < maxX2; x++ )
        {
            int v = prow2[x];
            int vl = x > 0 ? (v + prow2[x-1])/2 : v;
            int vr = x < width-1 ? (v + prow2[x+1])/2 : v;
            int v0 = std::min(vl, vr); v0 = std::min(v0, v);
            int v0 = std::max(vl, vr); v0 = std::max(v0, v);
            buffer[x] = (PixType)v0;
            buffer[x + width2] = (PixType)v0;
        }

        for( x = minX1; x < maxX1; x++ )
        {
            int u = prow1[x];
            int ul = x > 0 ? (u + prow1[x-1])/2 : u;
            int ur = x < width-1 ? (u + prow1[x+1])/2 : u;
            int u0 = std::min(ul, ur); u0 = std::min(u0, u);
            int u1 = std::max(ul, ur); u1 = std::max(u1, u);

            v_uint8x16 _u  = v_setall_u8((uchar)u), _u0 = v_setall_u8((uchar)u0);
            v_uint8x16 _u1 = v_setall_u8((uchar)u1);

            for( int d = minD; d < maxD; d += 16 )
            {
                v_uint8x16 _v  = v_load(prow2  + width-x-1 + d);
                v_uint8x16 _v0 = v_load(buffer + width-x-1 + d);
                v_uint8x16 _v0 = v_load(buffer + width-x-1 + d + width2);
                v_uint8x16 c0 = v_max(_u - _v0, _v0 - _u);
                v_uint8x16 c1 = v_max(_v - _u1, _u0 - _v);
                v_uint8x16 diff = v_min(c0, c1);

                v_int16x8 _c0 = v_load_aligned(cost + x*D + d);
                v_int16x8 _c1 = v_load_aligned(cost + x*D + d + 8);

                v_uint16x8 diff1,diff2;
                v_expand(diff,diff1,diff2);
                v_store_aligned(cost + x*D + d,     _c0 + v_reinterpret_as_s16(diff1 >> diff_scale));
                v_store_aligned(cost + x*D + d + 8, _c1 + v_reinterpret_as_s16(diff2 >> diff_scale));
            }
        }
    }
}*/

void diff_c(uchar* left,uchar* right,uchar* cost_array,int width,int height,int disparity){
    for(int d=0;d<disparity;d++){
	    for(int row=0;row<height;row++){
		    for(int column=D_MAX;column<width;column++){
			    int pos = (row*width+column);
			    int value_left_t = left[pos];
			    int pos_m = right[pos-d];
			    int pos_l = (right[pos-d-1]+pos_m)/2;
			    int pos_r = (right[pos-d+1]+pos_m)/2;
			    int right_max,right_min;
			    if(column -d ==0){
				    pos_l = right[pos-d];
			    }else if(column -d == width-1){
				    pos_r = right[pos-d];
			    }
			    right_min = min(pos_m,pos_l);
			    right_min = min(right_min,pos_r);
			    right_max = max(pos_m,pos_l);
			    right_max = max(right_max,pos_r);
			    int diff = max(0,value_left_t-right_max);
        		diff = max(diff,right_min-value_left_t);
			    cost_array[d*height*width+pos] = diff;
		    }		
	    }
    }
}

void census(uchar* src,uint* census_array,int width,int height){
    const int WINDOW_H = 5;
    const int WINDOW_W = 7;
    const int WINDOW_H_D2 = WINDOW_H / 2;
    const int WINDOW_W_D2 = WINDOW_W / 2;
    for(int row=WINDOW_H_D2 ;row<height-WINDOW_H_D2 ;row++){
        for(int column=WINDOW_W_D2 ;column<width-WINDOW_W_D2 ;column++){
            const int base_pos = row*width+column;
            const int c = ((int)src[base_pos]+
            src[base_pos-1]+src[base_pos+1]+
            src[base_pos-width]+src[base_pos+width])/5;
            uint value = 0;
            //计算中心点上方的census
		    for (int y = -WINDOW_H_D2; y < -1; y++) {
		        const int p_y =  width*(row + y);
				for (int x = -WINDOW_W_D2; x <= WINDOW_W_D2; x++) {
					uint result = (c - src[p_y + column + x]) > 0;
					value <<= 1;
					value += result;
				}
			}
			
			//计算中心点上方的census y = -1
			{
		        const int p_y =  width*(row -1);
		        //x < 0
				for (int x = -WINDOW_W_D2; x < 0; x++) {
					uint result = (c - src[p_y + column + x]) > 0;
					value <<= 1;
					value += result;
				}
				//x > 0
				for (int x = 1; x <=WINDOW_W_D2; x++) {
					uint result = (c - src[p_y + column + x]) > 0;
					value <<= 1;
					value += result;
				}
			}
			
			//计算中心点census y = 0
			{
		        const int p_y =  width*(row);
		        //x < -2
				for (int x = -WINDOW_W_D2; x < -1; x++) {
					uint result = (c - src[p_y + column + x]) > 0;
					value <<= 1;
					value += result;
				}
				//x > 1
				for (int x = 2; x <=WINDOW_W_D2; x++) {
					uint result = (c - src[p_y + column + x]) > 0;
					value <<= 1;
					value += result;
				}
			}
			
			//计算中心点下方的census y = 1
			{
		        const int p_y =  width*(row + 1);
		        //x < 0
				for (int x = -WINDOW_W_D2; x < 0; x++) {
					uint result = (c - src[p_y + column + x]) > 0;
					value <<= 1;
					value += result;
				}
				//x > 0
				for (int x = 1; x <=WINDOW_W_D2; x++) {
					uint result = (c - src[p_y + column + x]) > 0;
					value <<= 1;
					value += result;
				}
			}
			//计算中心点下方的census
		    for (int y = 2; y <= WINDOW_H_D2; y++) {
		        const int p_y =  width*(row + y);
				for (int x = -WINDOW_W_D2; x <= WINDOW_W_D2; x++) {
					uint result = (c - src[p_y + column + x]) > 0;
					value <<= 1;
					value += result;
				}
			}
			census_array[base_pos] = value;
			
        }
        
    }
}
//构建积分图
void integral(int* integral_result, uchar* image) {
	//列积分
	int rowSum[WIDTH]; // sum of each column

	// 计算第一行积分
	for (int column = 0; column<WIDTH; column++) {
		rowSum[column] = image[column];
		integral_result[column] = image[column];
		if (column>0) {
			integral_result[column] += integral_result[column - 1];
		}
	}
	for (int row = 1; row<HEIGHT; row++) {
		int offset = row*WIDTH;
		// 每列首行 = 上一行 + 此数据
		rowSum[0] += image[offset];
		integral_result[offset] = rowSum[0];
		for (int column = 1; column < WIDTH; column++) {
			int pos = offset + column;
			// 其余行 
			rowSum[column] += image[pos];
			integral_result[pos] = rowSum[column] + integral_result[pos - 1];
		}
	}
}
//低纹理检测
void low_texture_detection(const uchar* image,uchar* result,const int window_size) {
    short difference_integral[WIDTH*HEIGHT];
	//得到差分结果 以 行 为一维空间搜索
	for (int row = 0; row < HEIGHT; row++) {
		for (int column = 0; column < WIDTH - 1; column++) {
			int pos = row*WIDTH + column;
			int pos_next = pos +1;
			//进行积分
			if (column > 0) {
			    difference_integral[pos] = abs((short)image[pos_next] - (short)image[pos]);
				difference_integral[pos] += difference_integral[pos - 1];
			}
		}
	}
	for (int row = 0; row < HEIGHT; row++) {
		for (int column = 0; column < WIDTH - window_size; column++) {
			int pos_start = row*WIDTH + column;
			int pos = pos_start + window_size / 2;
			int pos_end = pos_start + window_size-1;
			int sum = 0;
			sum = difference_integral[pos_end];
			if (column > 0) {
				sum -= difference_integral[pos_start - 1];
			}
			result[pos] = sum>255?255:sum;
		}
	}
}
//dp算法 单个路径 测试
//dp1 left to right
void dp_l2r(short* dp,uchar* low_texture,uchar* cost_array,short P1,short P2){
    short min_cost_last;
    short min_cost_now;
    for(int row = 0;row<HEIGHT;row++){
        int pos_r = row*WIDTH;
        //column = 0
        {
            min_cost_last = 32767;
            for(int d=0;d<D_MAX;d++){
                int pos = pos_r+d*FRAME_SIZE;
                int cost = cost_array[pos];
                dp[pos] = cost;
                if(cost<min_cost_last){
                    min_cost_last = cost;
                }
            }
        }
        //column >0
        for(int column = 1;column<WIDTH;column++){
            int pos_c = pos_r+column;
            min_cost_now = 32767;
            for(int d=0;d<D_MAX;d++){
                int pos = pos_c+d*FRAME_SIZE;
                short cost = cost_array[pos];
                short v0=32767,v1=32767,v2=32767,vl=32767;
                if(d>0){
                    v0 = dp[pos-1-FRAME_SIZE]+P1;
                }
                if(d<D_MAX-1){
                    v2 = dp[pos+1-FRAME_SIZE]+P1;
                }
                v1 = dp[pos-FRAME_SIZE];
                vl = min_cost_last+P2;
                //the cost of this direction
                short cost_t =std::min(v0,std::min(v1,std::min(v2,vl)))-min_cost_last;
                dp[pos] = cost_t;
                if(cost_t<min_cost_now){
                    min_cost_now = cost_t;
                }
            }
            std::swap(min_cost_last,min_cost_now);
        }
    }
}
