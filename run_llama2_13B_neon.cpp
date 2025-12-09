// llama_gemm_w8a32_5120.c
#define D_MODEL 5120
#define GROUPS_PER_ROW 320  // 5120 / 16

void llama_gemm_w8a32_5120(
    const WeightGroup* weights,   // [M, 320]
    const float* hidden_state,    // [5120] fp32隐藏状态
    float* output,                // [M] fp32输出
    int M                         // 输出维度
) {
    const int groups_per_row = GROUPS_PER_ROW;
    
    for (int m = 0; m < M; m++) {
        const WeightGroup* w_row = &weights[m * groups_per_row];
        const float* in_ptr = hidden_state;
        
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        
        // 循环展开：一次处理8组（128个权重）
        for (int g = 0; g < groups_per_row; g += 8) {
            // 预取
            __builtin_prefetch(&w_row[g + 16], 0, 0);
            __builtin_prefetch(in_ptr + 128, 0, 0);
            
            // 处理8个连续的权重组
            for (int i = 0; i < 8; i++) {
                // 转换scale: fp16 → fp32
                float scale = fp16_to_fp32(w_row[g + i].scale_fp16);
                float32x4_t scale_vec = vdupq_n_f32(scale);
                
                // 加载权重
                int8x16_t w8x16 = vld1q_s8(w_row[g + i].weights);
                
                // 加载激活值
                float32x4_t in0 = vld1q_f32(in_ptr + i * 16);
                float32x4_t in1 = vld1q_f32(in_ptr + i * 16 + 4);
                float32x4_t in2 = vld1q_f32(in_ptr + i * 16 + 8);
                float32x4_t in3 = vld1q_f32(in_ptr + i * 16 + 12);
                
                // 权重转换和应用scale
                int16x8_t w_low = vmovl_s8(vget_low_s8(w8x16));
                float32x4_t w0 = vmulq_f32(
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_low))), scale_vec);
                float32x4_t w1 = vmulq_f32(
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_low))), scale_vec);
                
                int16x8_t w_high = vmovl_s8(vget_high_s8(w8x16));
                float32x4_t w2 = vmulq_f32(
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_high))), scale_vec);
                float32x4_t w3 = vmulq_f32(
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_high))), scale_vec);
                
                // 乘积累加
                acc0 = vfmaq_f32(acc0, w0, in0);
                acc1 = vfmaq_f32(acc1, w1, in1);
                acc2 = vfmaq_f32(acc2, w2, in2);
                acc3 = vfmaq_f32(acc3, w3, in3);
            }
            
            in_ptr += 128;  // 8组 × 16 = 128
        }
        
        // 合并和求和
        float32x4_t sum = vaddq_f32(acc0, acc1);
        sum = vaddq_f32(sum, acc2);
        sum = vaddq_f32(sum, acc3);
        
        // 水平求和
        float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
        float total = vget_lane_f32(vpadd_f32(sum2, sum2), 0);
        
        output[m] = total;
    }
}


