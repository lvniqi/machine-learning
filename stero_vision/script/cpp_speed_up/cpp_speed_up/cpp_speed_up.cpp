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