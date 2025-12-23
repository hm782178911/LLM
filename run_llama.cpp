#include<arm_neon.h>
#include<cstdint>
#include<cmath>

#define __fp16 float16_t

/*
    定义一个int8量化的数据结构，16个int8共享一个fp16的缩放因子
*/
struct Qint8x16{
    float16_t scale;
    int8_t elements[16];
}

/*
    K矩阵 [M,K]   Q矩阵[K,N]
    为了便于缓存优化
    src0 [M,K]保留原来的K矩阵，src1[N,K]采用Q矩阵的转置形式，这一步可以提前做处理，让权重文件直接就是转置的形式 
*/

Qint8x16 int8x16_quantize(float16_t* vec)
{   
    float16_t max_v=vec[0],min_v=vec[0];
    for(int i=0;i<16;i++){
        max_v=max(max_v,vec[i]);
        min_v=min(min_v,vec[i]);
    }
    float16_t scale=(max_v-min_v)/255;

    Qint8x16 res;
    res.scale=scale;
    for(int i=0;i<16;i++){
        res.elements[i]=(vec[i]-min_v)/scale;
    }

    return res;
}

Qint8x16 Qint8_ADD(const Qint8x16 *src0,const Qint8x16 *src1)
{
    float16_t res[16]={0};

    for(int i=0;i<16;i++){
        res[i]=src0->scale * src0->elements[i]
                + src1->scale * src1->elements[i];
    }

    return int8x16_quantize(&res);
}


void Qint8_Gemm(const Qint8x16 *src0,const Qint8x16 *src1,const Qint8x16 *dst,int M,int N,int K)
{
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            int32x4_t sum0=vdup_n_s32(0);
            int32x4_t sum1=vdup_n_s32(0);
            for(int k=0;k<K;k+=16){
                int8x16_t vec1=vld1q_s8(src0+i*K+k);
                int8x16_t vec2=vld1q_s8(src1+j*K+k);

                int32x4_t partial = vdotq_s32(vdupq_n_s32(0), vec1, vec2);
            }
        }
    }
}


double sigmoid(double x){
    return 1.0/(1.0+std::exp(-x));
}

float sigmoidf(float x) {
    return 1.0f / (1.0f + std::expf(-x));
}