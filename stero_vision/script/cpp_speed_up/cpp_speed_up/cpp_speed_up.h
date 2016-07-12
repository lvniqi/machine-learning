#pragma once

extern "C" _declspec(dllexport) int __stdcall test1(int a, int b);
extern "C" _declspec(dllexport) double __stdcall test2(double a[], int strides[], int shapes[]);
extern "C" _declspec(dllexport) void __stdcall compute_cost_d(INT16 result[], INT16 left[], INT16 right[], INT16 strides[], INT16 shapes[]);
extern "C" _declspec(dllexport) void __stdcall aggregate_cost(INT32 result[], INT16 diff[], INT32 diff_strides[], INT32 result_strides[], INT16 shapes[], INT16 window_size);