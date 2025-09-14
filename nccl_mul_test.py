import torch
import torch.distributed as dist
import time
import os
from tqdm import tqdm
def measure_all_reduce_speed(world_size=1, num_iters=100, tensor_size=1024):
    """
    测试 all_reduce 操作的通信速度
    Args:
        world_size (int): 进程总数
        num_iters (int): 测试迭代次数
        tensor_size (int): 单位张量的大小（单位：float32 个数）
    """
    try:
        # 初始化分布式环境
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        os.environ['NCCL_ALGO'] = 'Ring'
        dist.init_process_group(backend='nccl', rank=0, world_size=world_size)
        
        device = torch.device("cuda:0")  # 设置使用 GPU
        tensor = torch.ones(tensor_size, device=device)  # 创建张量

        # 同步开始时间
        dist.barrier()
        start_time = time.time()

        # 运行多次 all_reduce 操作
        for _ in range(num_iters):
            dist.all_reduce(tensor)

        # 同步结束时间
        dist.barrier()
        end_time = time.time()

        # 计算平均耗时
        total_time = end_time - start_time
        avg_time = total_time / num_iters

        # 数据传输量 (单位: MB)
        tensor_size_bytes = tensor_size * 4  # float32 是 4 字节
        data_transfer_mb = tensor_size_bytes * 2 / (1024 ** 2)  # 进程之间双向通信

        # 吞吐量 (单位: GB/s)
        bandwidth = data_transfer_mb / avg_time / 1024

        # 输出结果
        print(f"Tensor size: {tensor_size} float32 ({tensor_size_bytes / (1024 ** 2):.2f} MB)")
        print(f"Average all_reduce time: {avg_time:.6f} seconds")
        print(f"Bandwidth: {bandwidth:.2f} GB/s")
    finally:
        # 清理
        dist.destroy_process_group()


if __name__ == "__main__":
    if torch.cuda.is_available():
        measure_all_reduce_speed(world_size=1, num_iters=100, tensor_size=1024*1024 * 1024)
    else:
        print("CUDA GPU 不可用，请检查环境。")
