//测试 单一路径规划
__global__ void dp_kernel(uchar* dst,const uchar* cost_array,const uchar* cost_census_array){
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
    //y = 0
    {
        //d = 0
        short cost = ((cost_census_array[x]<<2)+(cost_array[x]>>1));
        last_line_disparity_last[0] = cost;
        sum_array[x] = cost;
        last_best_disparity_pos = 0;
        last_best_disparity_value = cost;
        //d > 0
        for(int d=1;d<D_MAX;d++){
            //当前代价
            short cost = ((cost_census_array[x+d*FRAME_SIZE]<<2)+(cost_array[x+d*FRAME_SIZE]>>1));
            last_line_disparity_last[d] = cost;
            sum_array[x+FRAME_SIZE*d] = cost;
            //选取最好的保存
            if(cost < last_best_disparity_value){
                last_best_disparity_pos = d;
                last_best_disparity_value = cost;
            }
        }
    }
    //y > 0
    for(int y=1;y<HEIGHT;y++){
        const int pos = y * WIDTH + x;
        //d = 0
        {
            short cost = ((cost_census_array[pos]<<2)+(cost_array[pos]>>1));
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
            sum_array[pos] = d0_value;
            now_best_disparity_pos = 0;
            now_best_disparity_value = d0_value;
            last_line_disparity_now[0] = d0_value;
        }
        for(int d=1;d<D_MAX-1;d++){
            int pos_d = pos+FRAME_SIZE*d;
            short cost = ((cost_census_array[pos_d]<<2)+(cost_array[pos_d]>>1));
            short left_t = last_line_disparity_last[d-1]+P1;
            short mid_t = last_line_disparity_last[d];
            short right_t = last_line_disparity_last[d+1]+P1;
            short last_t = last_best_disparity_value+P2;
            //get min of t
            int min_t = mycuda::min(mid_t,mycuda::min(left_t,mycuda::min(right_t,last_t)));
            short di_value = cost+min_t-last_best_disparity_value;
            sum_array[pos_d] = di_value;
            last_line_disparity_now[d] = di_value;
            if(di_value < now_best_disparity_value){
                now_best_disparity_pos = d;
                now_best_disparity_value = di_value;
            }
            //dst[pos] = last_best_disparity_pos;
        }
        //d = D_MAX-1
        {
            int d = D_MAX-1;
            int pos_d = pos+FRAME_SIZE*(D_MAX-1);
            short cost = ((cost_census_array[pos_d]<<2)+(cost_array[pos_d]>>1));
            //no left disparity
            short left_t = last_line_disparity_last[d-1]+P1;
            //mid
            short mid_t = last_line_disparity_last[d];
            //last min
            short last_t = last_best_disparity_value+P2;
            //get min of t
            short min_t = mycuda::min(mid_t,mycuda::min(left_t,last_t));
            short de_value = cost+min_t-last_best_disparity_value;
            sum_array[pos_d] = de_value;
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
        //dst[pos] = last_best_disparity_pos;
    }
    
    //y = HEIGHT-1
    {
        const int pos = (HEIGHT-1)*WIDTH+x;
        //d = 0
        short cost = ((cost_census_array[pos]<<2)+(cost_array[pos]>>1));
        last_line_disparity_last[0] = cost;
        sum_array[pos] = cost;
        last_best_disparity_pos = 0;
        last_best_disparity_value = cost + sum_array[pos];
        //d > 0
        for(int d=1;d<D_MAX;d++){
            //当前代价
            short cost = ((cost_census_array[pos+d*FRAME_SIZE]<<2)+(cost_array[pos+d*FRAME_SIZE]>>1));
            //calcute after up and down 
            short sum_value = cost + sum_array[pos+FRAME_SIZE*d];
            last_line_disparity_last[d] = cost;
            sum_array[pos+FRAME_SIZE*d] = cost;
            //选取最好的保存
            if(sum_value < last_best_disparity_value){
                last_best_disparity_pos = d;
                last_best_disparity_value = sum_value;
            }
        }
        dst[pos] = last_best_disparity_pos;
    }
    //y > 0
    for(int y=HEIGHT-2;y>=0;y--){
        const int pos = y * WIDTH + x;
        //d = 0
        {
            short cost = ((cost_census_array[pos]<<2)+(cost_array[pos]>>1));
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
            //calcute after up and down 
            short sum_d0_value = d0_value + sum_array[pos];
            sum_array[pos] = d0_value;
            now_best_disparity_pos = 0;
            now_best_disparity_value = sum_d0_value;
            last_line_disparity_now[0] = d0_value;
        }
        for(int d=1;d<D_MAX-1;d++){
            int pos_d = pos+FRAME_SIZE*d;
            short cost = ((cost_census_array[pos_d]<<2)+(cost_array[pos_d]>>1));
            short left_t = last_line_disparity_last[d-1]+P1;
            short mid_t = last_line_disparity_last[d];
            short right_t = last_line_disparity_last[d+1]+P1;
            short last_t = last_best_disparity_value+P2;
            //get min of t
            int min_t = mycuda::min(mid_t,mycuda::min(left_t,mycuda::min(right_t,last_t)));
            short di_value = cost+min_t-last_best_disparity_value;
            //calcute after up and down 
            short sum_di_value = di_value + sum_array[pos_d];
            sum_array[pos_d] = di_value;
            last_line_disparity_now[d] = di_value;
            if(sum_di_value < now_best_disparity_value){
                now_best_disparity_pos = d;
                now_best_disparity_value = sum_di_value;
            }
        }
        //d = D_MAX-1
        {
            int d = D_MAX-1;
            int pos_d = pos+FRAME_SIZE*(D_MAX-1);
            short cost = ((cost_census_array[pos_d]<<2)+(cost_array[pos_d]>>1));
            //no left disparity
            short left_t = last_line_disparity_last[d-1]+P1;
            //mid
            short mid_t = last_line_disparity_last[d];
            //last min
            short last_t = last_best_disparity_value+P2;
            //get min of t
            short min_t = mycuda::min(mid_t,mycuda::min(left_t,last_t));
            short de_value = cost+min_t-last_best_disparity_value;
            //calcute after up and down 
            short sum_de_value = de_value + sum_array[pos_d];
            sum_array[pos_d] = de_value;
            last_line_disparity_now[d] = de_value;
            if(sum_de_value < last_best_disparity_value){
                now_best_disparity_pos = d;
                now_best_disparity_value = sum_de_value;
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
        dst[pos] = last_best_disparity_pos;
    }
}

//测试 单一路径规划
__global__ void dp_kernel(uchar* dst,const uchar* cost_array,const uchar* cost_census_array){
    int d = blockIdx.x*blockDim.x+threadIdx.x;
    int x = blockIdx.y*blockDim.y+threadIdx.y+D_MAX;
    __shared__ int disparity_s[2][D_MAX];
    int *last_line_disparity_now = disparity_s[0];
    int *last_line_disparity_last = disparity_s[1];
    int left_t,mid_t,right_t;
    int pos = x+d*FRAME_SIZE;
    //y = 0
    {
        //d = 0
        short cost = ((cost_census_array[pos]<<2)+(cost_array[pos]>>1));
        last_line_disparity_last[d] = cost;
        sum_array[pos] = cost;
    }
    __syncthreads();
    //y > 0
    for(int y=1;y<HEIGHT;y++){
        pos = x+d*FRAME_SIZE+y*WIDTH;
        left_t = 0x0ffffff;
        right_t = 0x0ffffff;
        int cost = ((cost_census_array[pos]<<2)+(cost_array[pos]>>1));
        mid_t = last_line_disparity_last[d];
        if(d > 0){
            left_t = last_line_disparity_last[d-1]+P1;
        }
        if(d < D_MAX-1){
            right_t = last_line_disparity_last[d+1]+P1;
        }
        
        //get min of t
        int min_t = min(mid_t,min(left_t,right_t));
        int di_value = cost+min_t;
        sum_array[pos] = di_value;
        last_line_disparity_last[d] = di_value;
        //swap
        {
            int *p_t = last_line_disparity_now;
            last_line_disparity_now = last_line_disparity_last;
            last_line_disparity_last = p_t;
        }
    }
}


//测试 单一路径规划
__global__ void dp_kernel(uchar* dst,const uchar* cost_array,const uchar* cost_census_array){
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
    //y = 0
    {
        //d = 0
        short cost = ((cost_census_array[x]<<2)+(cost_array[x]>>1));
        last_line_disparity_last[0] = cost;
        sum_array[x] = cost;
        last_best_disparity_pos = 0;
        last_best_disparity_value = cost;
        //d > 0
        for(int d=1;d<D_MAX;d++){
            //当前代价
            short cost = ((cost_census_array[x+d*FRAME_SIZE]<<2)+(cost_array[x+d*FRAME_SIZE]>>1));
            last_line_disparity_last[d] = cost;
            sum_array[x+FRAME_SIZE*d] = cost;
            //选取最好的保存
            if(cost < last_best_disparity_value){
                last_best_disparity_pos = d;
                last_best_disparity_value = cost;
            }
        }
    }
    //y > 0
    for(int y=1;y<HEIGHT;y++){
        const int pos = y * WIDTH + x;
        //d = 0
        {
            short cost = ((cost_census_array[pos]<<2)+(cost_array[pos]>>1));
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
            sum_array[pos] = d0_value;
            now_best_disparity_pos = 0;
            now_best_disparity_value = d0_value;
            last_line_disparity_now[0] = d0_value;
        }
        for(int d=1;d<D_MAX-1;d++){
            int pos_d = pos+FRAME_SIZE*d;
            short cost = ((cost_census_array[pos_d]<<2)+(cost_array[pos_d]>>1));
            short left_t = last_line_disparity_last[d-1]+P1;
            short mid_t = last_line_disparity_last[d];
            short right_t = last_line_disparity_last[d+1]+P1;
            short last_t = last_best_disparity_value+P2;
            //get min of t
            int min_t = mycuda::min(mid_t,mycuda::min(left_t,mycuda::min(right_t,last_t)));
            short di_value = cost+min_t-last_best_disparity_value;
            sum_array[pos_d] = di_value;
            last_line_disparity_now[d] = di_value;
            if(di_value < now_best_disparity_value){
                now_best_disparity_pos = d;
                now_best_disparity_value = di_value;
            }
            dst[pos] = last_best_disparity_pos;
        }
        //d = D_MAX-1
        {
            int d = D_MAX-1;
            int pos_d = pos+FRAME_SIZE*(D_MAX-1);
            short cost = ((cost_census_array[pos_d]<<2)+(cost_array[pos_d]>>1));
            //no left disparity
            short left_t = last_line_disparity_last[d-1]+P1;
            //mid
            short mid_t = last_line_disparity_last[d];
            //last min
            short last_t = last_best_disparity_value+P2;
            //get min of t
            short min_t = mycuda::min(mid_t,mycuda::min(left_t,last_t));
            short de_value = cost+min_t-last_best_disparity_value;
            sum_array[pos_d] = de_value;
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
        dst[pos] = last_best_disparity_pos;
    }
}
