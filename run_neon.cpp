// gemm_w8a32_neon.c
#include <arm_neon.h>
#include <stddef.h>
#include <string.h>

// ==================== 数据结构 ====================
// 注意：scale存储为fp16编码，但计算时转为fp32
typedef struct {
    uint16_t scale_fp16;   // fp16编码存储，节省空间
    int8_t weights[16];    // 16个int8权重
} WeightGroup;

// ==================== fp16转换函数 ====================
static inline float fp16_to_fp32(uint16_t fp16) {
    // 快速fp16转fp32（ARM可能有内置指令）
    #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        __fp16 temp;
        memcpy(&temp, &fp16, sizeof(fp16));
        return (float)temp;
    #else
        // 软件实现
        uint32_t sign = (fp16 >> 15) & 0x1;
        uint32_t exp = (fp16 >> 10) & 0x1F;
        uint32_t frac = fp16 & 0x3FF;
        
        if (exp == 0) {
            // 非规格化数
            return (sign ? -1.0f : 1.0f) * (frac / 1024.0f) * 0.00006103515625f; // 2^-14
        } else if (exp == 31) {
            // 无穷大或NaN
            return (frac == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
        } else {
            // 规格化数
            float result = 1.0f + (frac / 1024.0f);
            result *= powf(2.0f, (int)exp - 15);
            return sign ? -result : result;
        }
    #endif
}

// ==================== 核心GEMM：W8A32 ====================
void gemm_w8a32_neon(
    const WeightGroup* weights,   // [M, K/16] 权重组
    const float* input,           // [K] fp32激活值
    float* output,                // [M] fp32输出
    int M,                        // 输出维度
    int K                         // 输入维度，必须是16的倍数
) {
    const int groups_per_row = K / 16;
    
    // 并行处理输出
    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        const WeightGroup* w_row = &weights[m * groups_per_row];
        const float* in_ptr = input;
        
        // 4个fp32累加器
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        
        // 在K维度上遍历分组
        for (int g = 0; g < groups_per_row; g += 2) {
            // ===== 预取 =====
            __builtin_prefetch(&w_row[g + 4], 0, 0);
            __builtin_prefetch(in_ptr + 64, 0, 0);
            
            // ===== 第1组 (0-15) =====
            {
                // 1. 获取并转换scale: fp16 → fp32
                float scale0 = fp16_to_fp32(w_row[g].scale_fp16);
                float32x4_t scale_vec0 = vdupq_n_f32(scale0);
                
                // 2. 加载16个int8权重
                int8x16_t w8x16_0 = vld1q_s8(w_row[g].weights);
                
                // 3. 加载16个fp32激活值
                float32x4_t in0 = vld1q_f32(in_ptr);
                float32x4_t in1 = vld1q_f32(in_ptr + 4);
                float32x4_t in2 = vld1q_f32(in_ptr + 8);
                float32x4_t in3 = vld1q_f32(in_ptr + 12);
                
                // 4. 权重转换：int8 → int16 → fp32
                // 低8位权重
                int16x8_t w_low_0 = vmovl_s8(vget_low_s8(w8x16_0));
                float32x4_t w00 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_low_0)));
                float32x4_t w01 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_low_0)));
                
                // 高8位权重
                int16x8_t w_high_0 = vmovl_s8(vget_high_s8(w8x16_0));
                float32x4_t w02 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_high_0)));
                float32x4_t w03 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_high_0)));
                
                // 5. 应用scale（fp32 × fp32）
                w00 = vmulq_f32(w00, scale_vec0);
                w01 = vmulq_f32(w01, scale_vec0);
                w02 = vmulq_f32(w02, scale_vec0);
                w03 = vmulq_f32(w03, scale_vec0);
                
                // 6. 乘积累加（fp32 × fp32）
                acc0 = vfmaq_f32(acc0, w00, in0);
                acc1 = vfmaq_f32(acc1, w01, in1);
                acc2 = vfmaq_f32(acc2, w02, in2);
                acc3 = vfmaq_f32(acc3, w03, in3);
            }
            
            // ===== 第2组 (16-31) =====
            {
                float scale1 = fp16_to_fp32(w_row[g + 1].scale_fp16);
                float32x4_t scale_vec1 = vdupq_n_f32(scale1);
                
                int8x16_t w8x16_1 = vld1q_s8(w_row[g + 1].weights);
                
                float32x4_t in4 = vld1q_f32(in_ptr + 16);
                float32x4_t in5 = vld1q_f32(in_ptr + 20);
                float32x4_t in6 = vld1q_f32(in_ptr + 24);
                float32x4_t in7 = vld1q_f32(in_ptr + 28);
                
                int16x8_t w_low_1 = vmovl_s8(vget_low_s8(w8x16_1));
                float32x4_t w10 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_low_1)));
                float32x4_t w11 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_low_1)));
                
                int16x8_t w_high_1 = vmovl_s8(vget_high_s8(w8x16_1));
                float32x4_t w12 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_high_1)));
                float32x4_t w13 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_high_1)));
                
                w10 = vmulq_f32(w10, scale_vec1);
                w11 = vmulq_f32(w11, scale_vec1);
                w12 = vmulq_f32(w12, scale_vec1);
                w13 = vmulq_f32(w13, scale_vec1);
                
                acc0 = vfmaq_f32(acc0, w10, in4);
                acc1 = vfmaq_f32(acc1, w11, in5);
                acc2 = vfmaq_f32(acc2, w12, in6);
                acc3 = vfmaq_f32(acc3, w13, in7);
            }
            
            // 移动输入指针：2组 × 16 = 32个激活值
            in_ptr += 32;
        }
        
        // ===== 合并累加器 =====
        float32x4_t sum = vaddq_f32(acc0, acc1);
        sum = vaddq_f32(sum, acc2);
        sum = vaddq_f32(sum, acc3);
        
        // ===== 水平求和 =====
        // 方法1：vaddvq（如果有）
        #if defined(__ARM_FEATURE_FMA) && defined(__aarch64__)
        float total = vaddvq_f32(sum);
        #else
        // 方法2：手动求和
        float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
        float total = vget_lane_f32(vpadd_f32(sum2, sum2), 0);
        #endif
        
        output[m] = total;
    }
}


