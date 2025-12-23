#include<cmath>
#include<vector>
#include<cstring>
#include<immintrin.h>  // SIMD指令集

class SwiGLUBasic {
private:
    int d_model;     // 输入输出维度，如4096
    int d_ff;        // 中间维度，如11008
    
    // 权重矩阵（列优先存储便于SIMD）
    float* W_gate;   // [d_ff, d_model] 实际存储为列优先
    float* W_up;     // [d_ff, d_model]
    float* W_down;   // [d_model, d_ff]
    
public:
    SwiGLUBasic(int d_model, int d_ff) : d_model(d_model), d_ff(d_ff) {
        // 分配内存（实际应考虑内存对齐）
        W_gate = new float[d_model * d_ff];
        W_up = new float[d_model * d_ff];
        W_down = new float[d_ff * d_model];
        
        // 初始化权重（简单示例）
        initializeWeights();
    }
    
    ~SwiGLUBasic() {
        delete[] W_gate;
        delete[] W_up;
        delete[] W_down;
    }
    
    void initializeWeights() {
        // Xavier/Glorot初始化
        float scale = sqrtf(2.0f / (d_model + d_ff));
        for (int i = 0; i < d_model * d_ff; i++) {
            W_gate[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
            W_up[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
        }
        for (int i = 0; i < d_ff * d_model; i++) {
            W_down[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
        }
    }
    
    // 基础实现：y = down_proj(silu(gate_proj(x)) * up_proj(x))
    void forward(const float* input, float* output, int batch_size = 1) {
        // 临时缓冲区
        float* gate = new float[batch_size * d_ff];
        float* up = new float[batch_size * d_ff];
        float* hidden = new float[batch_size * d_ff];
        
        // 1. 计算gate_proj和up_proj
        matmul(input, W_gate, gate, batch_size, d_ff, d_model);
        matmul(input, W_up, up, batch_size, d_ff, d_model);
        
        // 2. 应用SiLU激活：silu(x) = x * sigmoid(x)
        for (int i = 0; i < batch_size * d_ff; i++) {
            gate[i] = silu(gate[i]);  // silu(x) = x * sigmoid(x)
        }
        
        // 3. 逐元素相乘：hidden = gate * up
        for (int i = 0; i < batch_size * d_ff; i++) {
            hidden[i] = gate[i] * up[i];
        }
        
        // 4. 下投影：output = hidden * W_down
        matmul(hidden, W_down, output, batch_size, d_model, d_ff);
        
        delete[] gate;
        delete[] up;
        delete[] hidden;
    }
    
private:
    // 简单的矩阵乘法（可优化）
    void matmul(const float* A, const float* B, float* C, 
                int M, int N, int K) {
        // A: [M, K], B: [K, N], C: [M, N]
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
    
    // 数值稳定的sigmoid
    float sigmoid(float x) {
        if (x >= 0) {
            return 1.0f / (1.0f + expf(-x));
        } else {
            float ex = expf(x);
            return ex / (1.0f + ex);
        }
    }
    
    // SiLU: x * sigmoid(x)
    float silu(float x) {
        return x * sigmoid(x);
    }
};


//分块计算版本，适合大矩阵
class SwiGLUBlocked {
private:
    int d_model, d_ff;
    float *W_gate, *W_up, *W_down;
    
    static const int BLOCK_SIZE = 256;  // 分块大小
    
public:
    void forward_blocked(const float* input, float* output, int batch_size = 1) {
        float* gate = new float[batch_size * d_ff];
        float* up = new float[batch_size * d_ff];
        float* hidden = new float[batch_size * d_ff];
        
        // 分块矩阵乘法
        matmul_blocked(input, W_gate, gate, batch_size, d_ff, d_model);
        matmul_blocked(input, W_up, up, batch_size, d_ff, d_model);
        
        // 应用SiLU并相乘
        for (int i = 0; i < batch_size * d_ff; i++) {
            float x = gate[i];
            // 数值稳定的SiLU
            if (x >= 0) {
                float z = expf(-x);
                gate[i] = x / (1.0f + z);
            } else {
                float z = expf(x);
                gate[i] = x * z / (1.0f + z);
            }
            hidden[i] = gate[i] * up[i];
        }
        
        matmul_blocked(hidden, W_down, output, batch_size, d_model, d_ff);
        
        delete[] gate;
        delete[] up;
        delete[] hidden;
    }
    
private:
    void matmul_blocked(const float* A, const float* B, float* C,
                        int M, int N, int K) {
        // 清零输出矩阵
        memset(C, 0, M * N * sizeof(float));
        
        // 分块计算
        for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
            int i1 = std::min(i0 + BLOCK_SIZE, M);
            
            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                int k1 = std::min(k0 + BLOCK_SIZE, K);
                
                for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                    int j1 = std::min(j0 + BLOCK_SIZE, N);
                    
                    // 计算当前分块
                    for (int i = i0; i < i1; i++) {
                        for (int k = k0; k < k1; k++) {
                            float a = A[i * K + k];
                            for (int j = j0; j < j1; j++) {
                                C[i * N + j] += a * B[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    }
};