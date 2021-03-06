#pragma once
//次像素 16
#define SUB_PIXEL_LEVEL 16
//构建积分图
extern "C" _declspec(dllexport) void __stdcall integral(INT32 integral_result[], INT16 image[], int row_length, int column_length);
//计算单点代价
extern "C" _declspec(dllexport) void __stdcall compute_cost_d(INT16 result[], INT16 left[], INT16 right[], INT16 strides[], INT16 shapes[]);
//计算单点代价 BT版本
extern "C" _declspec(dllexport) void __stdcall compute_cost_bt_d(INT16 result[], INT16 left[], INT16 right[], INT16 strides[], INT16 shapes[]);
//代价聚合
extern "C" _declspec(dllexport) void __stdcall aggregate_cost(INT32 result[], INT16 diff[], const INT32 diff_strides[], const INT32 result_strides[], const INT16 shapes[], const INT16 window_size);
//动态规划
extern "C" _declspec(dllexport) void __stdcall DP_search_forward(INT16 result[], float cost[], const INT16 sad_row[], const INT32 column_length, const  INT32 d_max, const float p);
//动态规划 简化版
extern "C" _declspec(dllexport) void __stdcall DP_search_forward2(INT16 result[], float cost[], const  INT16 sad_row[], const INT32 column_length, const INT32 d_max, const float p);
//视差计算
extern "C" _declspec(dllexport) void __stdcall get_result(INT16 result[], const INT32 sad_diff[], const INT32 strides[], const INT32 shapes[]);
//亚像素求精
extern "C" _declspec(dllexport) int __stdcall subpixel_calculator(int d, int f_d, int f_d_l, int f_d_r);
//左右视差检查
extern "C" _declspec(dllexport) void __stdcall left_right_check(INT16 result[], INT16 left[], INT16 right[], const INT32 strides[], const INT32 shapes[]);
//视差后处理
extern "C" _declspec(dllexport) void __stdcall post_processing(INT16 result[], const INT16 left_right_result[], const INT32 strides[], const INT32 shapes[], const INT32 window_size, const INT32 d_max);

/*-------------census-------------*/
//计算census
extern "C" _declspec(dllexport) void __stdcall get_census(BOOLEAN result[], const INT16 image[], const INT32 strides[], const INT32 shapes[], const INT32 window_size);
//计算两个bool数组hamming距离
extern "C" _declspec(dllexport) INT16 __stdcall get_hamming_distance(const BOOLEAN census1[], const BOOLEAN census2[], const int len);
//计算单点代价 census版本
extern "C" _declspec(dllexport) void __stdcall compute_cost_census_d(INT16 result[], BOOLEAN left[], BOOLEAN right[], const INT16 strides[], const INT16 shapes[]);


/*-------------低纹理区-------------*/
//为低纹理区设计的右侧匹配算法
extern "C" _declspec(dllexport) void __stdcall get_low_texture_cost_r(INT16 result[], INT16 low_texture[], const INT32 shapes[]);
//为低纹理区设计的左侧匹配算法
extern "C" _declspec(dllexport) void __stdcall get_low_texture_cost_l(INT16 result[], INT16 low_texture[], const INT32 shapes[]);

/*-------------低纹理区域检测-------------*/
extern "C" _declspec(dllexport) void __stdcall low_texture_detection(INT16 row_result[], INT16 column_result[], const INT16 image[], const INT32 strides[], const INT32 shapes[], const INT32 window_size, const INT32 texture_range);
//代价聚合 使用低纹理区域检测获得的窗口
extern "C" _declspec(dllexport) void __stdcall aggregate_cost_window(INT32 result[], INT16 diff[], const INT32 diff_strides[], const INT32 result_strides[], const INT16 shapes[], const INT16 row_window[],const INT16 column_window[],const INT16 window_size_min,const INT16 window_size_max);

/*-------------semi-grobal-------------*/
//动态规划 简化版 X
extern "C" _declspec(dllexport) void __stdcall SGM_search(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max, const float p);