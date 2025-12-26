#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

#define cudaCheck(err) _cudaCheck(err,__FILE__,__LINE__)

void _cudaCheck(cudaError_t error,const char *file,int line){
    if(error != cudaSuccess){
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n",
            file,line,cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
}


__global__ void matrix_elementwise_add(float* a_device,float* b_device,float* c_device,int M,int N){
    int col=blockDim.x * blockIdx.x + threadIdx.x;
    int row=boolDim.y*blockIdx.y + threadIdx.y;

    if(col>=M || row>=N )return;

    c_device[col*M+row]=a_device[col*M+row]+b_device[col*M+row];
}


int main()
{   
    constexpr int M=3,N=4;

    //alloc host memory
    float* a_h=(float*)malloc(N*sizeof(float));
    float* b_h=(float*)malloc(N*sizeof(float));
    float* c_h=(float*)malloc(N*sizeof(float));

    //initialize data
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            int idx=i*M+j;
            a_h[idx]=j;
            b_h[idx]=100-1-j;
        }
    }

    //alloc device memory
    float* a_d=nullptr;
    float* b_d=nullptr;
    float* c_d=nullptr;

    cudaCheck(cudaMalloc((void**)(&a_d),M*N*sizeof(float)))
    cudaCheck(cudaMalloc((void**)(&b_d),M*N*sizeof(float)));
    cudaCheck(cudaMalloc((void**)(&c_d),M*N*sizeof(float)));
    cudaCheck(cudaMemcpy(a_d,a_h,M*N*sizeof(float),cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(b_d,b_h,M*N*sizeof(float),cudaMemcpyHostToDevice));

     // 使用2D网格配置
    dim3 block_size(16, 16);  // 256个线程
    dim3 grid_size((M + block_size.x - 1) / block_size.x,
                   (N + block_size.y - 1) / block_size.y);

    matrix_elementwise_add<<<grid_size,block_size>>>(a_d,b_d,c_d,M,N);

    cudaCheck(cudaMemcpy(c_h,c_d,M*N*sizeof(float),cudaMemcpyDeviceToHost));

    // 释放内存
    free(a_h);
    free(b_h);
    free(c_h);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return 0;
}
