#include "stdafx.h"
#include "cpp_speed_up.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
//计算census
void __stdcall get_census(BOOLEAN result[], const INT16 image[], const INT32 strides[], const INT32 shapes[], const INT32 window_size) {
	//row length
	int row_length = shapes[0];
	//column length
	int column_length = shapes[1];
	//row and column size
	int S0 = strides[0] / sizeof(INT16);
	int S1 = strides[1] / sizeof(INT16);
	//先计算中间的像素
	for (int row = 0; row < row_length; row++) {
		for (int column = 0; column < column_length; column++) {
			//中点位置
			int pos = row*S0 + column*S1;
			//待测值
			int mid_value = image[pos];
			//开始位置
			int start_pos = (row - window_size / 2)*S0 + (column - window_size / 2)*S1;
			//开始行
			int start_row = 0;
			//结束行
			int stop_row = window_size;
			//开始列
			int start_column = 0;
			//结束列
			int stop_column = window_size;
			/*//上侧边界检查
			if (row < window_size / 2){
				start_row = window_size / 2 - row;
			}
			//下侧边界检查
			else if(row >= row_length - window_size / 2) {
				stop_row = (window_size/2+1) + row_length - row;
			}
			//左侧边界检查
			if (column < window_size / 2) {
				start_column = window_size / 2 - column;
			}
			//右侧边界检查
			else if (column >= column_length - window_size / 2) {
				stop_column = (window_size / 2 + 1) + column_length - column;
			}*/

			for (int i = start_row; i < stop_row; i++) {
				for (int j = start_column; j < stop_column; j++) {

					result[((row*column_length + column)*window_size*window_size) + i*window_size + j] =
						image[start_pos + i*S0 + j*S1] > mid_value ? true : false;
				}
			}
		}
	}
}

//计算两个bool数组hamming距离
INT16 __stdcall get_hamming_distance(const BOOLEAN census1[], const BOOLEAN census2[],const int start, const int end) {
	//不同计数
	INT16 count = 0;
	for (int i = start; i < end; i++) {
		if (census1[i] != census2[i]) {
			count++;
		}
	}
	return count;
}
//计算单点代价
void __stdcall compute_cost_census_d(INT16 result[],BOOLEAN left[],BOOLEAN right[],const INT16 strides[],const INT16 shapes[]) {
	//row length
	int row_length = shapes[0];
	//column length
	int column_length = shapes[1];
	//census length
	int census_length = shapes[2];
	//row and column size
	int S0 = strides[0] / sizeof(BOOLEAN);
	int S1 = strides[1] / sizeof(BOOLEAN);
	int S2 = strides[2] / sizeof(BOOLEAN);
	for (int row = 0; row < row_length; row++) {
		for (int column = 0; column < column_length; column++) {
			int pos = row*S0 + column*S1;
			result[row*column_length + column] = 
				get_hamming_distance(&(left[pos]), &(right[pos]), 0,census_length);
		}
	}
}