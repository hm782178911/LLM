#include <arm_neon.h>
#include <cmath>
#include <cstring>

//使用NEON优化的完整SwiGLU实现
class SwiGLUNEON {
private:
    int d_model;  // 如4096
    int d_ff;     // 如11008
    
    // 权重矩阵（确保内存对齐）
    float* W_gate;  // [d_model, d_ff]
    float* W_up;    // [d_model, d_ff]
    float* W_down;  // [d_ff, d_model]
    
public:
    SwiGLUNEON(int d_model, int d_ff) : d_model(d_model), d_ff(d_ff) {
        // 对齐内存分配（NEON要求128位/16字节对齐）
        W_gate = (float*)aligned_alloc(16, d_model * d_ff * sizeof(float));
        W_up = (float*)aligned_alloc(16, d_model * d_ff * sizeof(float));
        W_down = (float*)aligned_alloc(16, d_ff * d_model * sizeof(float));
        
        initializeWeightsNEON();
    }
    
    ~SwiGLUNEON() {
        free(W_gate);
        free(W_up);
        free(W_down);
    }
    
    // NEON优化的前向传播
    void forward_neon(const float* input, float* output, int batch_size = 1) {
        // 使用对齐的临时缓冲区
        float* gate = (float*)aligned_alloc(16, batch_size * d_ff * sizeof(float));
        float* up = (float*)aligned_alloc(16, batch_size * d_ff * sizeof(float));
        float* hidden = (float*)aligned_alloc(16, batch_size * d_ff * sizeof(float));
        
        // 1. NEON优化的矩阵乘法
        matmul_neon(input, W_gate, gate, batch_size, d_ff, d_model);
        matmul_neon(input, W_up, up, batch_size, d_ff, d_model);
        
        // 2. NEON优化的SiLU激活 + 逐元素相乘
        silu_mul_neon(gate, up, hidden, batch_size * d_ff);
        
        // 3. 下投影
        matmul_neon(hidden, W_down, output, batch_size, d_model, d_ff);
        
        free(gate);
        free(up);
        free(hidden);
    }
    
private:
    // ============ NEON优化的矩阵乘法 ============
    void matmul_neon(const float* A, const float* B, float* C,
                     int M, int N, int K) {
        // A: [M, K], B: [K, N], C: [M, N]
        
        // 清零输出矩阵（使用NEON）
        memset_neon(C, 0, M * N * sizeof(float));
        
        // 循环展开 + NEON
        for (int i = 0; i < M; i++) {
            const float* A_row = &A[i * K];
            
            for (int k = 0; k < K; k++) {
                float a_val = A_row[k];
                
                // 广播标量a_val到NEON寄存器
                float32x4_t a_vec = vdupq_n_f32(a_val);
                
                const float* B_col = &B[k * N];
                float* C_row = &C[i * N];
                
                // 一次处理16个元素（4个NEON寄存器）
                int j = 0;
                for (; j <= N - 16; j += 16) {
                    // 加载B的16个元素
                    float32x4_t b0 = vld1q_f32(B_col + j);
                    float32x4_t b1 = vld1q_f32(B_col + j + 4);
                    float32x4_t b2 = vld1q_f32(B_col + j + 8);
                    float32x4_t b3 = vld1q_f32(B_col + j + 12);
                    
                    // 加载C的当前值
                    float32x4_t c0 = vld1q_f32(C_row + j);
                    float32x4_t c1 = vld1q_f32(C_row + j + 4);
                    float32x4_t c2 = vld1q_f32(C_row + j + 8);
                    float32x4_t c3 = vld1q_f32(C_row + j + 12);
                    
                    // 乘加操作: C = C + a * B
                    c0 = vmlaq_f32(c0, a_vec, b0);
                    c1 = vmlaq_f32(c1, a_vec, b1);
                    c2 = vmlaq_f32(c2, a_vec, b2);
                    c3 = vmlaq_f32(c3, a_vec, b3);
                    
                    // 存回C
                    vst1q_f32(C_row + j, c0);
                    vst1q_f32(C_row + j + 4, c1);
                    vst1q_f32(C_row + j + 8, c2);
                    vst1q_f32(C_row + j + 12, c3);
                }
                
                // 处理剩余元素
                for (; j < N; j++) {
                    C_row[j] += a_val * B_col[j];
                }
            }
        }
    }
    
    // ============ NEON优化的SiLU激活+乘法融合 ============
    void silu_mul_neon(float* gate, float* up, float* hidden, int size) {
        const float32x4_t zero = vdupq_n_f32(0.0f);
        const float32x4_t one = vdupq_n_f32(1.0f);
        const float32x4_t neg_one = vdupq_n_f32(-1.0f);
        
        int i = 0;
        for (; i <= size - 16; i += 16) {
            // 一次加载16个元素
            float32x4_t g0 = vld1q_f32(gate + i);
            float32x4_t g1 = vld1q_f32(gate + i + 4);
            float32x4_t g2 = vld1q_f32(gate + i + 8);
            float32x4_t g3 = vld1q_f32(gate + i + 12);
            
            float32x4_t u0 = vld1q_f32(up + i);
            float32x4_t u1 = vld1q_f32(up + i + 4);
            float32x4_t u2 = vld1q_f32(up + i + 8);
            float32x4_t u3 = vld1q_f32(up + i + 12);
            
            // 计算sigmoid(gate)的NEON近似
            float32x4_t s0 = fast_sigmoid_neon(g0);
            float32x4_t s1 = fast_sigmoid_neon(g1);
            float32x4_t s2 = fast_sigmoid_neon(g2);
            float32x4_t s3 = fast_sigmoid_neon(g3);
            
            // SiLU(gate) = gate * sigmoid(gate)
            float32x4_t silu0 = vmulq_f32(g0, s0);
            float32x4_t silu1 = vmulq_f32(g1, s1);
            float32x4_t silu2 = vmulq_f32(g2, s2);
            float32x4_t silu3 = vmulq_f32(g3, s3);
            
            // hidden = SiLU(gate) * up
            float32x4_t h0 = vmulq_f32(silu0, u0);
            float32x4_t h1 = vmulq_f32(silu1, u1);
            float32x4_t h2 = vmulq_f32(silu2, u2);
            float32x4_t h3 = vmulq_f32(silu3, u3);
            
            vst1q_f32(hidden + i, h0);
            vst1q_f32(hidden + i + 4, h1);
            vst1q_f32(hidden + i + 8, h2);
            vst1q_f32(hidden + i + 12, h3);
        }
        
        // 处理剩余元素
        for (; i < size; i++) {
            float g = gate[i];
            // 快速SiLU近似
            float s = fast_sigmoid_scalar(g);
            hidden[i] = (g * s) * up[i];
        }
    }
    
    // ============ 快速sigmoid的NEON实现 ============
    float32x4_t fast_sigmoid_neon(float32x4_t x) {
        // 近似公式: sigmoid(x) ≈ 0.5 + 0.5 * tanh(0.5 * x)
        const float32x4_t half = vdupq_n_f32(0.5f);
        const float32x4_t one = vdupq_n_f32(1.0f);
        
        // 计算0.5 * x
        float32x4_t half_x = vmulq_f32(half, x);
        
        // 使用NEON的tanh近似（或者使用更快的分段线性近似）
        // 这里简化，实际可用多项式近似
        float32x4_t tanh_half_x = fast_tanh_neon(half_x);
        
        // 0.5 + 0.5 * tanh(0.5 * x)
        float32x4_t result = vaddq_f32(half, vmulq_f32(half, tanh_half_x));
        
        return result;
    }
    
    // 快速tanh的NEON近似
    float32x4_t fast_tanh_neon(float32x4_t x) {
        // tanh(x) ≈ x * (27 + x^2) / (27 + 9 * x^2)
        const float32x4_t twenty_seven = vdupq_n_f32(27.0f);
        const float32x4_t nine = vdupq_n_f32(9.0f);
        
        float32x4_t x2 = vmulq_f32(x, x);
        float32x4_t numerator = vmulq_f32(x, vaddq_f32(twenty_seven, x2));
        float32x4_t denominator = vaddq_f32(twenty_seven, vmulq_f32(nine, x2));
        
        return vdivq_f32(numerator, denominator);
    }
    
    // 标量版本的快速sigmoid（处理剩余元素）
    float fast_sigmoid_scalar(float x) {
        // 更快的分段线性近似
        if (x < -6.0f) return 0.0f;
        if (x > 6.0f) return 1.0f;
        
        // 线性插值
        float y = 0.5f + 0.5f * (x / (1.0f + fabsf(x)));
        return y;
    }
    
    // ============ NEON优化的内存设置 ============
    void memset_neon(float* ptr, float value, size_t size) {
        float32x4_t val_vec = vdupq_n_f32(value);
        size_t num_vectors = size / (4 * sizeof(float));
        
        for (size_t i = 0; i < num_vectors; i++) {
            vst1q_f32(ptr + i * 4, val_vec);
        }
        
        // 处理剩余字节
        size_t remaining = size % (4 * sizeof(float));
        for (size_t i = size - remaining; i < size; i++) {
            ptr[i] = value;
        }
    }
    
    // ============ NEON优化的权重初始化 ============
    void initializeWeightsNEON() {
        // 使用NEON加速的Xavier初始化
        float scale = sqrtf(2.0f / (d_model + d_ff));
        float32x4_t scale_vec = vdupq_n_f32(scale * 2.0f);
        float32x4_t neg_scale_vec = vdupq_n_f32(-scale);
        
        // 初始化W_gate和W_up
        for (int i = 0; i < d_model * d_ff; i += 4) {
            // 生成随机数（简化，实际应用需要更好的随机数生成）
            float32x4_t rand_vec = {
                (float)rand() / RAND_MAX,
                (float)rand() / RAND_MAX,
                (float)rand() / RAND_MAX,
                (float)rand() / RAND_MAX
            };
            
            // 缩放到[-scale, scale]
            float32x4_t weight = vmlaq_f32(neg_scale_vec, scale_vec, rand_vec);
            
            vst1q_f32(W_gate + i, weight);
            vst1q_f32(W_up + i, weight);
        }
        
        // 初始化W_down
        for (int i = 0; i < d_ff * d_model; i += 4) {
            float32x4_t rand_vec = {
                (float)rand() / RAND_MAX,
                (float)rand() / RAND_MAX,
                (float)rand() / RAND_MAX,
                (float)rand() / RAND_MAX
            };
            
            float32x4_t weight = vmlaq_f32(neg_scale_vec, scale_vec, rand_vec);
            vst1q_f32(W_down + i, weight);
        }
    }
};


//进一步优化：内存布局和缓存友好
class SwiGLUNEONOptimized {
private:
    int d_model, d_ff;
    
    // 使用转置的权重布局，便于NEON访问
    float* W_gate_T;  // [d_ff, d_model] 转置存储
    float* W_up_T;    // [d_ff, d_model]
    float* W_down;    // [d_ff, d_model] 原始布局
    
public:
    // NEON优化的转置矩阵乘法（更好的内存访问模式）
    void matmul_transpose_neon(const float* A, const float* B_T, float* C,
                               int M, int N, int K) {
        // A: [M, K], B_T: [N, K]（转置的B）, C: [M, N]
        
        // 外循环i，内循环j使用NEON
        for (int i = 0; i < M; i++) {
            const float* A_row = &A[i * K];
            float* C_row = &C[i * N];
            
            // 对每个输出列j
            for (int j = 0; j < N; j++) {
                const float* B_col = &B_T[j * K];  // 连续访问！
                
                // 使用NEON计算点积
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                int k = 0;
                
                // 一次处理16个元素
                for (; k <= K - 16; k += 16) {
                    float32x4_t a0 = vld1q_f32(A_row + k);
                    float32x4_t a1 = vld1q_f32(A_row + k + 4);
                    float32x4_t a2 = vld1q_f32(A_row + k + 8);
                    float32x4_t a3 = vld1q_f32(A_row + k + 12);
                    
                    float32x4_t b0 = vld1q_f32(B_col + k);
                    float32x4_t b1 = vld1q_f32(B_col + k + 4);
                    float32x4_t b2 = vld1q_f32(B_col + k + 8);
                    float32x4_t b3 = vld1q_f32(B_col + k + 12);
                    
                    sum_vec = vmlaq_f32(sum_vec, a0, b0);
                    sum_vec = vmlaq_f32(sum_vec, a1, b1);
                    sum_vec = vmlaq_f32(sum_vec, a2, b2);
                    sum_vec = vmlaq_f32(sum_vec, a3, b3);
                }
                
                // 水平求和
                float sum = horizontal_sum_neon(sum_vec);
                
                // 处理剩余元素
                for (; k < K; k++) {
                    sum += A_row[k] * B_col[k];
                }
                
                C_row[j] = sum;
            }
        }
    }
    
    float horizontal_sum_neon(float32x4_t vec) {
        // 水平求和：a[0]+a[1]+a[2]+a[3]
        float32x2_t sum2 = vadd_f32(vget_low_f32(vec), vget_high_f32(vec));
        float32x2_t sum = vpadd_f32(sum2, sum2);
        return vget_lane_f32(sum, 0);
    }
};

