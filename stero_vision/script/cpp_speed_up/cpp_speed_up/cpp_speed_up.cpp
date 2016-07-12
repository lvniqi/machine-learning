// cpp_speed_up.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "cpp_speed_up.h"
#include "stdio.h"
int __stdcall test1(int a, int b) {
	return a + b;
}

double __stdcall test2(double a[], int strides[], int shapes[]) {
	double sum = 0;
	int i, j, M, N, S0, S1;
	M = shapes[0]; N = shapes[1];
	S0 = strides[0] / sizeof(double);
	S1 = strides[1] / sizeof(double);

	for (i = 0; i<M; i++) {
		for (j = 0; j<N; j++) {
			sum += a[i*S0 + j*S1];
		}
	}
	return sum;
}

void __stdcall compute_cost_d(INT16 result[], INT16 left[], INT16 right[], INT16 strides[], INT16 shapes[]) {
	//row length
	int row_length = shapes[0];
	//column length
	int column_length = shapes[1];
	//row and column size
	int S0 = strides[0] / sizeof(INT16);
	int S1 = strides[1] / sizeof(INT16);
	for (int row = 0; row < row_length; row ++) {
		for (int column = 0; column < column_length; column ++) {
			int pos = row*S0 + column*S1;
			if (left[pos] > right[pos]) {
				result[pos] = left[pos] - right[pos];
			}
			else {
				result[pos] = right[pos] - left[pos];
			}
		}
	}
}

void __stdcall aggregate_cost(INT32 result[], INT16 diff[],INT32 diff_strides[], INT32 result_strides[], INT16 shapes[], INT16 window_size) {
	//deep length
	int d_max = shapes[0];
	//row length
	int row_length = shapes[1];
	//column length
	int column_length = shapes[2];
	//deep size
	int S_deep = diff_strides[0] / sizeof(INT16);
	//row size
	int S_row = diff_strides[1] / sizeof(INT16);
	//column size
	int S_column = diff_strides[2] / sizeof(INT16);


	//result row size
	int S_row_r = result_strides[0] / sizeof(INT32);
	//result column size
	int S_column_r = result_strides[1] / sizeof(INT32);
	//result deep size
	int S_deep_r = result_strides[2] / sizeof(INT32);
	//printf("row_length:%d column_length:%d d_max:%d\r\n", row_length,column_length,d_max);
	for (int d = 0; d < d_max; d++) {
		//printf("d:%d\r\n", d);
		INT16* diff_this_deep = &(diff[S_deep*d]);
		for (int row = 0; row < row_length; row++) {
			//先确定异常边界
			int top=0, bottom = row_length;
			//如果正常 重新确定边界
			if (row - window_size / 2 > 0) {
				top = row - window_size / 2;
			}
			if (row + window_size / 2 + 1 <= row_length) {
				bottom = row + window_size / 2 + 1;
			}
			for (int column = 0; column < column_length; column++) {
				//先确定异常边界
				int left = 0,right = column_length;
				//如果正常 重新确定边界
				if (column - window_size / 2 > 0) {
					left = column - window_size / 2;
				}
				if (column + window_size / 2 + 1 <= column_length) {
					right = column + window_size / 2 + 1;
				}
				//求和 SAD
				INT32 sad = 0;
				for (int i = top; i < bottom; i++) {
					for (int j = left; j < right; j++) {
						sad += diff_this_deep[i*S_row + j*S_column];
					}
				}
				result[row*S_row_r + column*S_column_r + d*S_deep_r] = sad;
			}
		}
	}
}