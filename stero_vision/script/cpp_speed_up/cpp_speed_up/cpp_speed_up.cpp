// cpp_speed_up.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "cpp_speed_up.h"

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

void __stdcall compute_cost_d(INT16 result[], INT16 left[], INT16 right[], int strides[], int shapes[]) {
	//行数
	int row_length = shapes[0];
	//列数
	int column_length = shapes[1];
	//行列宽度
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
