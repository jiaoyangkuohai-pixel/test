
def comput_memory():
    B = 671
    P = B*10**9
    gpu_memory = 16 * P / (1024**3)





# Implementing the memory computation formulas in Python based on the image provided

def memory_no_recompute(s, b, h, L, a, t):
    return s * b * h * L * (10 + 24 / t + 5 * (a * s) / (h * t))

def memory_selective_recompute(s, b, h, L, t):
    return s * b * h * L * (10 + 24 / t)

def memory_full_recompute(s, b, h, L):
    return 2 * s * b * h * L

def memory_example():
    # Example usage of these functions:
    s = 4096  # Sequence length
    b = 1    # Batch size per GPU
    h = 8192 # Hidden size dimension
    L = 80    # Number of layers in the transformer model
    a = 64    # Number of attention heads
    t = 1     # Degree of tensor parallelism

    # Calculate memory requirements
    memory_no_recompute_val = memory_no_recompute(s, b, h, L, a, t) /(1024**3)
    memory_selective_recompute_val = memory_selective_recompute(s, b, h, L, t)/(1024**3)
    memory_full_recompute_val = memory_full_recompute(s, b, h, L)/(1024**3)

    return (memory_no_recompute_val, memory_selective_recompute_val, memory_full_recompute_val)

def compute_gpu_counts():
    B = 671
    P = B*10**9  # 参数量
    gpu_memory = 16 * P / (1024**3)

    ratio = 0.3 # 显卡的利用率
    epoch = 2 # 假设需要1个epoch
    D = 5 *10**9  # 5P约10G-20G数据  
    C = 8 * P * D # openai算力测试， 采用重计算的话6->8
    days = 24 # 假设需要5天训练完成
    a800 = 312 * 10 **12 * ratio # A800算力
    a100_a800 = 1.05 # 耗时 通信性能有33%的降低，导致增加5%的耗时

    count = C*epoch/(days/a100_a800)/86400/a800 # 需要的卡数
    return count

if __name__ == "__main__":
    print(f"{'='*10}重计算demo{'='*10}")
    print(memory_example())
    print(f"{'='*10}计算显卡数量{'='*10}")
    gpu_count = compute_gpu_counts()
    print(f"显卡数量计算结果: \n{gpu_count}")
