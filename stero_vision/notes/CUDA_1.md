CUDA编程(5)
=========
本以为CUDA没啥麻烦的，结果被坑了。想实现SGM，结果在代价聚合的地方就慢成狗。看来还是滚去啃OPENCV的CUDA源码试试。
## 程序入口
```CPP

void stereoBM_CUDA(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& disp, int maxdisp, int winsz, const PtrStepSz<unsigned int>& minSSD_buf, cudaStream_t& stream)
        {
            int winsz2 = winsz >> 1;

            if (winsz2 == 0 || winsz2 >= calles_num)
                CV_Error(cv::Error::StsBadArg, "Unsupported window size");

            cudaSafeCall( cudaMemset2D(disp.data, disp.step, 0, disp.cols, disp.rows) );
            cudaSafeCall( cudaMemset2D(minSSD_buf.data, minSSD_buf.step, 0xFF, minSSD_buf.cols * minSSD_buf.elemSize(), disp.rows) );

            cudaSafeCall( cudaMemcpyToSymbol( cwidth, &left.cols, sizeof(left.cols) ) );
            cudaSafeCall( cudaMemcpyToSymbol( cheight, &left.rows, sizeof(left.rows) ) );
            cudaSafeCall( cudaMemcpyToSymbol( cminSSDImage, &minSSD_buf.data, sizeof(minSSD_buf.data) ) );

            size_t minssd_step = minSSD_buf.step/minSSD_buf.elemSize();
            cudaSafeCall( cudaMemcpyToSymbol( cminSSD_step,  &minssd_step, sizeof(minssd_step) ) );

            callers[winsz2](left, right, disp, maxdisp, stream);
        }

```
* 首先将窗口大小winsz/2 得到 半个窗口大小
* 将深度数据和SSD数据清空
* 拷贝宽度、高度、最小SSD图
* 运行**callers[winsz2](left, right, disp, maxdisp, stream)** 函数

至于**callers** 是啥？
```CPP
typedef void (*kernel_caller_t)(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& disp, int maxdisp, cudaStream_t & stream);

const static kernel_caller_t callers[] =
        {
            0,
            kernel_caller< 1>, kernel_caller< 2>, kernel_caller< 3>, kernel_caller< 4>, kernel_caller< 5>,
            kernel_caller< 6>, kernel_caller< 7>, kernel_caller< 8>, kernel_caller< 9>, kernel_caller<10>,
            kernel_caller<11>, kernel_caller<12>, kernel_caller<13>, kernel_caller<14>, kernel_caller<15>,
            kernel_caller<16>, kernel_caller<17>, kernel_caller<18>, kernel_caller<19>, kernel_caller<20>,
            kernel_caller<21>, kernel_caller<22>, kernel_caller<23>, kernel_caller<24>, kernel_caller<25>

            //0,0,0, 0,0,0, 0,0,kernel_caller<9>
        };
```
可见caller是一个**函数指针数组**

## CUDA入口
```CPP
template<int RADIUS> void kernel_caller(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& disp, int maxdisp, cudaStream_t & stream)
        {
            dim3 grid(1,1,1);
            dim3 threads(BLOCK_W, 1, 1);

            grid.x = divUp(left.cols - maxdisp - 2 * RADIUS, BLOCK_W);
            grid.y = divUp(left.rows - 2 * RADIUS, ROWSperTHREAD);

            //See above:  #define COL_SSD_SIZE (BLOCK_W + 2 * RADIUS)
            size_t smem_size = (BLOCK_W + N_DISPARITIES * (BLOCK_W + 2 * RADIUS)) * sizeof(unsigned int);

            stereoKernel<RADIUS><<<grid, threads, smem_size, stream>>>(left.data, right.data, left.step, disp, maxdisp);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        };
```
这就是为CUDA分配**线程**和**线程块**的函数。
貌似是一个模板函数，然而没有泛型，所以也不算，猜测是为了节省资源而为之。
* 单个线程处理的列数为**ROWSperTHREAD**(21)
* 线程块处理的行数为**BLOCK_W**(128)
* 线程块宽度为(行数-最大深度-窗口大小)/(单个线程处理的列数)
* 线程块高度为(列数-窗口大小)/(线程块列数)
然后分配内存为整个一行窗口缓冲(单块行数 + 深度* (单块行数+ 窗口大小))
**(限制！！TK1 中，最大为49152 bytes)**

## 核函数
```CPP
template<int RADIUS>
__global__ void stereoKernel(unsigned char *left, unsigned char *right, size_t img_step, PtrStepb disp, int maxdisp)
{
	extern __shared__ unsigned int col_ssd_cache[];
	volatile unsigned int *col_ssd = col_ssd_cache + BLOCK_W + threadIdx.x;
	volatile unsigned int *col_ssd_extra = threadIdx.x < (2 * RADIUS) ? col_ssd + BLOCK_W : 0;  //#define N_DIRTY_PIXELS (2 * RADIUS)

	//#define X (blockIdx.x * BLOCK_W + threadIdx.x + STEREO_MAXD)
	int X = (blockIdx.x * BLOCK_W + threadIdx.x + maxdisp + RADIUS);
    //#define Y (__mul24(blockIdx.y, ROWSperTHREAD) + RADIUS)
	#define Y (blockIdx.y * ROWSperTHREAD + RADIUS)
	//int Y = blockIdx.y * ROWSperTHREAD + RADIUS;

	unsigned int* minSSDImage = cminSSDImage + X + Y * cminSSD_step;
	unsigned char* disparImage = disp.data + X + Y * disp.step;
 /*   if (X < cwidth)
	{
		unsigned int *minSSDImage_end = minSSDImage + min(ROWSperTHREAD, cheight - Y) * minssd_step;
		for(uint *ptr = minSSDImage; ptr != minSSDImage_end; ptr += minssd_step )
			*ptr = 0xFFFFFFFF;
	}*/
	int end_row = ::min(ROWSperTHREAD, cheight - Y - RADIUS);
	int y_tex;
	int x_tex = X - RADIUS;

	if (x_tex >= cwidth)
		return;

	for(int d = STEREO_MIND; d < maxdisp; d += STEREO_DISP_STEP)
	{
		y_tex = Y - RADIUS;

		InitColSSD<RADIUS>(x_tex, y_tex, img_step, left, right, d, col_ssd);

		if (col_ssd_extra > 0)
			if (x_tex + BLOCK_W < cwidth)
				InitColSSD<RADIUS>(x_tex + BLOCK_W, y_tex, img_step, left, right, d, col_ssd_extra);

		__syncthreads(); //before MinSSD function

		if (X < cwidth - RADIUS && Y < cheight - RADIUS)
		{
			uint2 minSSD = MinSSD<RADIUS>(col_ssd_cache + threadIdx.x, col_ssd);
			if (minSSD.x < minSSDImage[0])
			{
				disparImage[0] = (unsigned char)(d + minSSD.y);
				minSSDImage[0] = minSSD.x;
			}
		}

		for(int row = 1; row < end_row; row++)
		{
			int idx1 = y_tex * img_step + x_tex;
			int idx2 = (y_tex + (2 * RADIUS + 1)) * img_step + x_tex;

			__syncthreads();

			StepDown<RADIUS>(idx1, idx2, left, right, d, col_ssd);

			if (col_ssd_extra)
				if (x_tex + BLOCK_W < cwidth)
					StepDown<RADIUS>(idx1, idx2, left + BLOCK_W, right + BLOCK_W, d, col_ssd_extra);

			y_tex += 1;

			__syncthreads(); //before MinSSD function

			if (X < cwidth - RADIUS && row < cheight - RADIUS - Y)
			{
				int idx = row * cminSSD_step;
				uint2 minSSD = MinSSD<RADIUS>(col_ssd_cache + threadIdx.x, col_ssd);
				if (minSSD.x < minSSDImage[idx])
				{
					disparImage[disp.step * row] = (unsigned char)(d + minSSD.y);
					minSSDImage[idx] = minSSD.x;
				}
			}
		} // for row loop
	} // for d loop
}
```
此函数为主函数 要好好观察下:
* **col_ssd_cache** 为外部动态分配的shared memory
* volatile unsigned int *col_ssd = col_ssd_cache + BLOCK_W + threadIdx.x;
* X = 行号+ 最大深度+窗口大小/2
* Y = 列号 * 单个线程处理的列数+ 窗口大小/2)
* ssd图像位置 