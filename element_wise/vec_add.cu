#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

//convert to vector with 4 float
#define FLOAT4(a) *(float4*)(&(a))
//Round down
#define CEIL(a,b) ((a+b-1)/(b))
// #define FLOAT4_CXX(a) (reinterpret_cast<float4*>(&(a))[0])

#define cudaCheck(err) _cudaCheck(err,__FILE__,__LINE__)

void _cudaCheck(cudaError_t error,const char *file,int line){
    if(error != cudaSuccess){
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n",
            file,line,cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
}

/*
trade off:
compute cost increase and memory cost decrease
*/
__global__ void elementwise_add_float4(float* a,float* b,float* c,int N){
    int idx=(blockDim.x * blockIdx.x + threadIdx.x)*4;
    if(idx >=N)return;

    float4 tmp_a=FLOAT4(a[idx]);
    float4 tmp_b=FLOAT4(b[idx]);
    float4 tmp_c;

    tmp_c.x=tmp_a.x+tmp_b.x;
    tmp_c.y=tmp_a.y+tmp_b.y;
    tmp_c.z=tmp_a.z+tmp_b.z;
    tmp_c.w=tmp_a.w+tmp_b.w;

    FLOAT4(c[idx])=tmp_c
}

int main(){

    constexpr int N=7;

    //alloc host memory
    float* a_h=(float*)malloc(N*sizeof(float));
    float* b_h=(float*)malloc(N*sizeof(float));
    float* c_h=(float*)malloc(N*sizeof(float));

    //initialize data
    for(int i=0;i<N;i++){
        a_h[i]=i;
        b_h[i]=N-1-i;
    }

    //alloc device memory
    float* a_d=nullptr;
    float* b_d=nullptr;
    float* c_d=nullptr;

    cudaCheck(cudaMalloc((void**)(&a_d),N*sizeof(float)));
    cudaCheck(cudaMalloc((void**)(&b_d),N*sizeof(float)));
    cudaCheck(cudaMalloc((void**)(&c_d),N*sizeof(float)));
    cudaCheck(cudaMemcpy(a_d,a_h,N*sizeof(float),cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(b_d,b_h,N*sizeof(float),cudaMemcpyHostToDevice));

    int block_size=1024;
    int grid_size=CEIL(CEIL(N,4),1024);

    elementwise_add_float4<<<grid_size,block_size>>>(a_d,b_d,c_d,N);

    cudaCheck( cudaMemcpy(c_h,c_d,N*sizeof(float),cudaMemcpyDeviceToHost) );
    printf("a_h:\n");
    for(int i=0;i<N;i++)printf("%f ",a_h[i]);
    printf("\n");
    printf("b_h:\n");
    for(int i=0;i<N;i++)printf("%f ",b_h[i]);
    printf("\n");
    printf("c_h:\n");
    for(int i=0;i<N;i++)printf("%f ",c_h[i]);
    printf("\n");

    // 释放内存
    free(a_h);
    free(b_h);
    free(c_h);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return 0;
}

