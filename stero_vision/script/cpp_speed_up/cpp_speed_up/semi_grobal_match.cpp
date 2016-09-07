#include "stdafx.h"
#include "cpp_speed_up.h"
#include <math.h>

void __stdcall DP_init_top(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max) {
	//初始化 遍历第一行所有数据
	for (int column = 0; column < column_length; column++) {
		//第一个数据 没有约束项
		for (int d = 0; d < d_max; d++) {
			//代价等于 数据项
			cost[column*d_max + d] = sad[column*d_max + d];
		}
	}
}
void __stdcall DP_init_bottom(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max) {
	//初始化 遍历最后行所有数据
	for (int column = 0; column < column_length - 1; column++) {
		//第一个数据 没有约束项
		for (int d = 0; d < d_max; d++) {
			//代价等于 数据项
			cost[((row_length - 1)*column_length + column)*d_max + d] = sad[((row_length - 1)*column_length + column)*d_max + d];
		}
	}
}
void __stdcall DP_init_left(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max) {
	//初始化 遍历第一列所有数据
	for (int row = 0; row < row_length; row++) {
		//第一个数据 没有约束项
		for (int d = 0; d < d_max; d++) {
			//代价等于 数据项
			cost[row*column_length*d_max + d] = sad[row*column_length*d_max + d];
		}
	}
}
void __stdcall DP_init_right(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max) {
	for (int row = 0; row < row_length; row++) {
		//第一个数据 没有约束项
		for (int d = 0; d < d_max; d++) {
			//代价等于 数据项
			cost[(row*column_length + column_length - 1)*d_max + d] = sad[(row*column_length + column_length - 1)*d_max + d];
		}
	}
}


//动态规划 简化版 基础
void __stdcall DP_search_base(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max, const float p) {
	;
}

//动态规划 简化版 left to right
void __stdcall DP_search_forward(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max, const float p) {
	int p_i = 10 * p;//10用于减少float to int 损失
	for (int row = 0; row < row_length; row++) {
		int row_pos = row*column_length*d_max;
		//第一个数据 没有约束项
		for (int d = 0; d < d_max; d++) {
			//代价等于 数据项
			int cost_t = sad[row_pos + 0 * d_max + d];
			cost[row_pos + 0 * d_max + d] = cost_t;
		}
		//遍历一行所有数据
		for (int column = 1; column < column_length; column++) {
			//遍历所有视差
			for (int d = 0; d < d_max; d++) {
				//差异值
				int d_diff = 2;
				//实际内存位置
				int pos = row_pos + column * d_max + d;
				//上次的最佳视差
				int min_last = 0;
				//总体代价结果
				INT32 cost_result = 0x70000000;
				//上一个点的视差
				for (int last_disparity = (d - d_diff < 0 ? 0 : d - d_diff); last_disparity < (d + d_diff + 1 < d_max ? d + d_diff + 1 : d_max); last_disparity++) {
					//约束项
					int disparity_diff = abs(d - last_disparity);
					int cost_disparity = p_i*disparity_diff;
					// 代价等于 数据项 + 约束系数*约束项
					int cost_now = 10 * sad[pos] + cost_disparity;//10用于减少float to int 损失
																		//代价合计
					int cost_sum = cost_now + cost[row_pos + column * d_max - d_max + last_disparity];
					//取得最小值
					if (cost_sum < cost_result) {
						cost_result = cost_sum;
					}
				}
				cost[pos] = cost_result;
			}
		}
	}
}
//动态规划 简化版 right to left
void __stdcall DP_search_reverse(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max, const float p) {
	int p_i = 10 * p;//10用于减少float to int 损失
	for (int row = 0; row < row_length; row++) {
		int row_pos = row*column_length*d_max;
		//第一个数据 没有约束项
		for (int d = 0; d < d_max; d++) {
			//代价等于 数据项
			int cost_t = sad[row_pos + (column_length - 1) * d_max + d];
			cost[row_pos + (column_length - 1) * d_max + d] = cost_t;
		}
		//遍历一行所有数据
		for (int column = column_length - 2; column >= 0; column--) {
			//遍历所有视差
			for (int d = 0; d < d_max; d++) {
				//差异值
				int d_diff = 2;
				//实际内存位置
				int pos = row_pos + column * d_max + d;
				//上次的最佳视差
				int min_last = 0;
				//总体代价结果
				INT32 cost_result = 0x70000000;
				//上一个点的视差
				for (int last_disparity = (d - d_diff < 0 ? 0 : d - d_diff); last_disparity < (d + d_diff + 1 < d_max ? d + d_diff + 1 : d_max); last_disparity++) {
					//约束项
					int disparity_diff = abs(d - last_disparity);
					int cost_disparity = p_i *disparity_diff;
					// 代价等于 数据项 + 约束系数*约束项
					int cost_now = 10 * sad[pos] + cost_disparity;//10用于减少float to int 损失
																  //代价合计
					int cost_sum = cost_now + cost[row_pos + column * d_max + d_max + last_disparity];
					//取得最小值
					if (cost_sum < cost_result) {
						cost_result = cost_sum;
					}
				}
				cost[pos] = cost_result;
			}
		}
	}
}

//动态规划 简化版 top to bottom
void __stdcall DP_search_down(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max, const float p) {
	int p_i = 10 * p;//10用于减少float to int 损失
	DP_init_top(cost, sad, row_length, column_length, d_max);
	for (int row = 1; row < row_length; row++) {
		int row_pos = row*column_length*d_max;
		//遍历一行所有数据
		for (int column = 0; column < column_length; column++) {
			//遍历所有视差
			for (int d = 0; d < d_max; d++) {
				//差异值
				int d_diff = 2;
				//实际内存位置
				int pos = row_pos + column * d_max + d;
				//上次的最佳视差
				int min_last = 0;
				//总体代价结果
				INT32 cost_result = 0x70000000;
				//上一个点的视差
				for (int last_disparity = (d - d_diff < 0 ? 0 : d - d_diff); last_disparity < (d + d_diff + 1 < d_max ? d + d_diff + 1 : d_max); last_disparity++) {
					//约束项
					int disparity_diff = abs(d - last_disparity);
					int cost_disparity = p_i *disparity_diff;
					// 代价等于 数据项 + 约束系数*约束项
					int cost_now = 10 * sad[pos] + cost_disparity;//10用于减少float to int 损失
					//代价合计
					int cost_sum = cost_now + cost[row_pos - column_length*d_max + column * d_max + last_disparity];
					//取得最小值
					if (cost_sum < cost_result) {
						cost_result = cost_sum;
					}
				}
				cost[pos] = cost_result;
			}
		}
	}
}
//动态规划 简化版 bottom to top
void __stdcall DP_search_up(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max, const float p) {
	int p_i = 10 * p;//10用于减少float to int 损失
	//初始化 遍历最后列所有数据
	DP_init_bottom(cost, sad, row_length, column_length, d_max);
	for (int row = row_length - 2; row >= 0; row--) {
		int row_pos = row*column_length*d_max;
		//遍历一行所有数据
		for (int column = 0; column < column_length; column++) {
			//遍历所有视差
			for (int d = 0; d < d_max; d++) {
				//差异值
				int d_diff = 2;
				//实际内存位置
				int pos = row_pos + column * d_max + d;
				//上次的最佳视差
				int min_last = 0;
				//总体代价结果
				INT32 cost_result = 0x70000000;
				//上一个点的视差
				for (int last_disparity = (d - d_diff < 0 ? 0 : d - d_diff); last_disparity < (d + d_diff + 1 < d_max ? d + d_diff + 1 : d_max); last_disparity++) {
					//约束项
					int disparity_diff = abs(d - last_disparity);
					int cost_disparity = p_i *disparity_diff;
					// 代价等于 数据项 + 约束系数*约束项
					int cost_now = 10 * sad[pos] + cost_disparity;//10用于减少float to int 损失
																  //代价合计
					int cost_sum = cost_now + cost[row_pos + column_length*d_max + column * d_max + last_disparity];
					//取得最小值
					if (cost_sum < cost_result) {
						cost_result = cost_sum;
					}
				}
				cost[pos] = cost_result;
			}
		}
	}
}
//动态规划 简化版 45°
void __stdcall DP_search_45(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max, const float p) {
	int p_i = 10 * p;//10用于减少float to int 损失
	//初始化 遍历最后行所有数据
	DP_init_bottom(cost, sad, row_length, column_length, d_max);
	//初始化 遍历第一列所有数据
	DP_init_left(cost, sad, row_length, column_length, d_max);
	for (int row = row_length - 2; row >= 0; row--) {
		int row_pos = row*column_length*d_max;
		//遍历一行所有数据
		for (int column = 1; column < column_length; column++) {
			//遍历所有视差
			for (int d = 0; d < d_max; d++) {
				//差异值
				int d_diff = 2;
				//实际内存位置
				int pos = row_pos + column * d_max + d;
				//上次的最佳视差
				int min_last = 0;
				//总体代价结果
				INT32 cost_result = 0x70000000;
				//上一个点的视差
				for (int last_disparity = (d - d_diff < 0 ? 0 : d - d_diff); last_disparity < (d + d_diff + 1 < d_max ? d + d_diff + 1 : d_max); last_disparity++) {
					//约束项
					int disparity_diff = abs(d - last_disparity);
					int cost_disparity = p_i *disparity_diff;
					// 代价等于 数据项 + 约束系数*约束项
					int cost_now = 10 * sad[pos] + cost_disparity;//10用于减少float to int 损失
					//上一个代价的 位置
					int last_pos = row_pos + column_length*d_max + (column-1) * d_max + last_disparity;
					//代价合计
					int cost_sum = cost_now + cost[last_pos];
					//取得最小值
					if (cost_sum < cost_result) {
						cost_result = cost_sum;
					}
				}
				cost[pos] = cost_result;
			}
		}
	}
}

//动态规划 简化版 135°
void __stdcall DP_search_135(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max, const float p) {
	int p_i = 10 * p;//10用于减少float to int 损失
	//初始化 遍历最后行所有数据
	DP_init_bottom(cost, sad, row_length, column_length, d_max);
	//初始化 遍历最后列所有数据
	DP_init_right(cost, sad, row_length, column_length, d_max);
	for (int row = row_length - 2; row >= 0; row--) {
		int row_pos = row*column_length*d_max;
		//遍历一行所有数据
		for (int column = column_length-2; column >= 0; column--) {
			//遍历所有视差
			for (int d = 0; d < d_max; d++) {
				//差异值
				int d_diff = 2;
				//实际内存位置
				int pos = row_pos + column * d_max + d;
				//上次的最佳视差
				int min_last = 0;
				//总体代价结果
				INT32 cost_result = 0x70000000;
				//上一个点的视差
				for (int last_disparity = (d - d_diff < 0 ? 0 : d - d_diff); last_disparity < (d + d_diff + 1 < d_max ? d + d_diff + 1 : d_max); last_disparity++) {
					//约束项
					int disparity_diff = abs(d - last_disparity);
					int cost_disparity = p_i *disparity_diff;
					// 代价等于 数据项 + 约束系数*约束项
					int cost_now = 10 * sad[pos] + cost_disparity;//10用于减少float to int 损失
																  //上一个代价的 位置
					int last_pos = row_pos + column_length*d_max + (column+1) * d_max + last_disparity;
					//代价合计
					int cost_sum = cost_now + cost[last_pos];
					//取得最小值
					if (cost_sum < cost_result) {
						cost_result = cost_sum;
					}
				}
				cost[pos] = cost_result;
			}
		}
	}
}

//动态规划 简化版 225°
void __stdcall DP_search_225(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max, const float p) {
	int p_i = 10 * p;//10用于减少float to int 损失
	//初始化 遍历第一行所有数据
	DP_init_top(cost, sad, row_length, column_length, d_max);
	//初始化 遍历最后列所有数据
	DP_init_right(cost, sad, row_length, column_length, d_max);
	for (int row = 1; row < row_length; row++) {
		int row_pos = row*column_length*d_max;
		//遍历一行所有数据
		for (int column = column_length - 2; column >= 0; column--) {
			//遍历所有视差
			for (int d = 0; d < d_max; d++) {
				//差异值
				int d_diff = 2;
				//实际内存位置
				int pos = row_pos + column * d_max + d;
				//上次的最佳视差
				int min_last = 0;
				//总体代价结果
				INT32 cost_result = 0x70000000;
				//上一个点的视差
				for (int last_disparity = (d - d_diff < 0 ? 0 : d - d_diff); last_disparity < (d + d_diff + 1 < d_max ? d + d_diff + 1 : d_max); last_disparity++) {
					//约束项
					int disparity_diff = abs(d - last_disparity);
					int cost_disparity = p_i *disparity_diff;
					// 代价等于 数据项 + 约束系数*约束项
					int cost_now = 10 * sad[pos] + cost_disparity;//10用于减少float to int 损失
					//上一个代价的 位置
					int last_pos = row_pos - column_length*d_max + (column+1) * d_max + last_disparity;
					//代价合计
					int cost_sum = cost_now + cost[last_pos];
					//取得最小值
					if (cost_sum < cost_result) {
						cost_result = cost_sum;
					}
				}
				cost[pos] = cost_result;
			}
		}
	}
}

//动态规划 简化版 315°
void __stdcall DP_search_315(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max, const float p) {
	int p_i = 10 * p;//10用于减少float to int 损失
	//初始化 遍历第一行所有数据
	DP_init_top(cost, sad, row_length, column_length, d_max);
	//初始化 遍历第一列所有数据
	DP_init_left(cost, sad, row_length, column_length, d_max);
	for (int row = 1; row < row_length; row++) {
		int row_pos = row*column_length*d_max;
		//遍历一行所有数据
		for (int column = 1; column < column_length; column++) {
			//遍历所有视差
			for (int d = 0; d < d_max; d++) {
				//差异值
				int d_diff = 2;
				//实际内存位置
				int pos = row_pos + column * d_max + d;
				//上次的最佳视差
				int min_last = 0;
				//总体代价结果
				INT32 cost_result = 0x70000000;
				//上一个点的视差
				for (int last_disparity = (d - d_diff < 0 ? 0 : d - d_diff); last_disparity < (d + d_diff + 1 < d_max ? d + d_diff + 1 : d_max); last_disparity++) {
					//约束项
					int disparity_diff = abs(d - last_disparity);
					int cost_disparity = p_i *disparity_diff;
					// 代价等于 数据项 + 约束系数*约束项
					int cost_now = 10 * sad[pos] + cost_disparity;//10用于减少float to int 损失
																  //上一个代价的 位置
					int last_pos = row_pos - column_length*d_max + (column - 1) * d_max + last_disparity;
					//代价合计
					int cost_sum = cost_now + cost[last_pos];
					//取得最小值
					if (cost_sum < cost_result) {
						cost_result = cost_sum;
					}
				}
				cost[pos] = cost_result;
			}
		}
	}
}


//动态规划 简化版 X
void __stdcall SGM_search(INT32 cost[], const  INT32 sad[], const INT32 row_length, const INT32 column_length, const INT32 d_max, const float p) {
	INT32* cost_forward = new INT32[row_length*column_length*d_max];
	INT32* cost_reverse = new INT32[row_length*column_length*d_max];
	INT32* cost_up = new INT32[row_length*column_length*d_max];
	INT32* cost_down = new INT32[row_length*column_length*d_max];
	INT32* cost_45 = new INT32[row_length*column_length*d_max];
	INT32* cost_135 = new INT32[row_length*column_length*d_max];
	INT32* cost_225 = new INT32[row_length*column_length*d_max]; 
	INT32* cost_315 = new INT32[row_length*column_length*d_max];
	DP_search_forward(cost_forward, sad, row_length, column_length, d_max, p);
	DP_search_reverse(cost_reverse, sad, row_length, column_length, d_max, p);
	DP_search_up(cost_up, sad, row_length, column_length, d_max, p);
	DP_search_down(cost_down, sad, row_length, column_length, d_max, p);
	DP_search_45(cost_45, sad, row_length, column_length, d_max, p/1.4);
	DP_search_135(cost_135, sad, row_length, column_length, d_max, p / 1.4);
	DP_search_225(cost_225, sad, row_length, column_length, d_max, p/1.4);
	DP_search_315(cost_315, sad, row_length, column_length, d_max, p / 1.4);

	for (int i = 0; i < row_length; i++) {
		for (int j = 0; j < column_length; j++) {
			for (int k = 0; k < d_max; k++) {
				int pos = (i*column_length + j)*d_max + k;
				cost[pos] = (
					cost_forward[pos] + cost_reverse[pos] + 
					cost_up[pos] + cost_down[pos] +
					cost_45[pos] + cost_225[pos] +
					cost_135[pos] + cost_315[pos]
					) / 8;
				//cost[pos] = (cost_forward[pos] + cost_reverse[pos]) / 2;
			}
		}
	}
	delete(cost_forward);
	delete(cost_reverse);
	delete(cost_up);
	delete(cost_down);
	delete(cost_45);
	delete(cost_135);
	delete(cost_225);
	delete(cost_315);
}