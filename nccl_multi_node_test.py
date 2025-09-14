#!/usr/bin/env python3
"""
真正的多机多卡NCCL通信测试
支持在多个节点上同时运行多个GPU进程
"""

import torch
import torch.distributed as dist
import time
import argparse
import json
from typing import Dict, List
import numpy as np
import os
import socket


class NCCLMultiNodeTester:
    """多机多卡NCCL集群通信测试器"""
    
    def __init__(self):
        # torchrun会自动设置这些环境变量
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        self.node_rank = int(os.environ.get('NODE_RANK', 0))
        self.device = None
        self.results = {}
        
    def init_distributed(self):
        """初始化分布式环境"""
        # 设置OpenMP线程数 (避免多进程线程竞争)
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        # 设置NCCL环境变量
        os.environ['NCCL_ALGO'] = 'Ring'
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_IB_DISABLE'] = '0'  # 启用InfiniBand
        os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # 指定网络接口
        
        # 初始化进程组 - torchrun已经设置了环境变量
        dist.init_process_group(backend='nccl')
        
        # 设置设备 - 支持CUDA_VISIBLE_DEVICES
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        
        # 获取节点信息
        hostname = socket.gethostname()
        
        # 获取CUDA设备信息
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')
        actual_device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(actual_device_id)
        
        print(f"进程 {self.rank} 初始化完成:")
        print(f"  - 节点: {hostname} (NODE_RANK: {self.node_rank})")
        print(f"  - 本地RANK: {self.local_rank}")
        print(f"  - CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        print(f"  - 实际设备ID: {actual_device_id}")
        print(f"  - 设备名称: {device_name}")
        print(f"  - PyTorch设备: {self.device}")
        print(f"  - 总进程数: {self.world_size}")
        print(f"  - 本地进程数: {self.local_world_size}")
        
    def cleanup(self):
        """清理分布式环境"""
        if dist.is_initialized():
            dist.destroy_process_group()
            
    def test_allreduce_bandwidth(self, tensor_sizes: List[int], num_iters: int = 10) -> Dict:
        """测试AllReduce操作的带宽"""
        if self.rank == 0:
            print(f"\n开始AllReduce带宽测试...")
        
        results = {}
        
        for size in tensor_sizes:
            # 创建张量
            tensor = torch.ones(size, dtype=torch.float32, device=self.device)
            
            # 预热
            for _ in range(3):
                dist.all_reduce(tensor)
            
            # 同步所有进程
            dist.barrier()
            
            # 开始计时
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 执行多次AllReduce操作
            for _ in range(num_iters):
                dist.all_reduce(tensor)
            
            # 结束计时
            torch.cuda.synchronize()
            end_time = time.time()
            
            # 计算带宽
            total_time = end_time - start_time
            avg_time = total_time / num_iters
            
            # 数据传输量计算
            data_size_bytes = size * 4  # float32 = 4 bytes
            total_data_transfer = data_size_bytes * (self.world_size - 1)
            bandwidth_gbps = (total_data_transfer / avg_time) / (1024**3)
            
            results[size] = {
                'avg_time': avg_time,
                'bandwidth_gbps': bandwidth_gbps,
                'data_size_mb': data_size_bytes / (1024**2)
            }
            
            if self.rank == 0:
                print(f"张量大小 {size} ({data_size_bytes/(1024**2):.2f}MB), "
                      f"平均时间 {avg_time:.6f}s, 带宽 {bandwidth_gbps:.2f} GB/s")
        
        return results
    
    def test_allgather_bandwidth(self, tensor_sizes: List[int], num_iters: int = 10) -> Dict:
        """测试AllGather操作的带宽"""
        if self.rank == 0:
            print(f"\n开始AllGather带宽测试...")
        
        results = {}
        
        for size in tensor_sizes:
            # 创建输入张量
            input_tensor = torch.ones(size, dtype=torch.float32, device=self.device) * self.rank
            # 创建输出张量
            output_tensor = torch.zeros(size * self.world_size, dtype=torch.float32, device=self.device)
            
            # 预热
            for _ in range(3):
                dist.all_gather_into_tensor(output_tensor, input_tensor)
            
            # 同步所有进程
            dist.barrier()
            
            # 开始计时
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 执行多次AllGather操作
            for _ in range(num_iters):
                dist.all_gather_into_tensor(output_tensor, input_tensor)
            
            # 结束计时
            torch.cuda.synchronize()
            end_time = time.time()
            
            # 计算带宽
            total_time = end_time - start_time
            avg_time = total_time / num_iters
            
            data_size_bytes = size * 4
            total_data_transfer = data_size_bytes * (self.world_size - 1)
            bandwidth_gbps = (total_data_transfer / avg_time) / (1024**3)
            
            results[size] = {
                'avg_time': avg_time,
                'bandwidth_gbps': bandwidth_gbps,
                'data_size_mb': data_size_bytes / (1024**2)
            }
            
            if self.rank == 0:
                print(f"AllGather 张量大小 {size} ({data_size_bytes/(1024**2):.2f}MB), "
                      f"平均时间 {avg_time:.6f}s, 带宽 {bandwidth_gbps:.2f} GB/s")
        
        return results
    
    def test_reduce_scatter_bandwidth(self, tensor_sizes: List[int], num_iters: int = 10) -> Dict:
        """测试ReduceScatter操作的带宽"""
        if self.rank == 0:
            print(f"\n开始ReduceScatter带宽测试...")
        
        results = {}
        
        for size in tensor_sizes:
            # 创建输入张量
            input_tensor = torch.ones(size * self.world_size, dtype=torch.float32, device=self.device) * self.rank
            # 创建输出张量
            output_tensor = torch.zeros(size, dtype=torch.float32, device=self.device)
            
            # 预热
            for _ in range(3):
                dist.reduce_scatter_tensor(output_tensor, input_tensor)
            
            # 同步所有进程
            dist.barrier()
            
            # 开始计时
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 执行多次ReduceScatter操作
            for _ in range(num_iters):
                dist.reduce_scatter_tensor(output_tensor, input_tensor)
            
            # 结束计时
            torch.cuda.synchronize()
            end_time = time.time()
            
            # 计算带宽
            total_time = end_time - start_time
            avg_time = total_time / num_iters
            
            data_size_bytes = size * 4
            total_data_transfer = data_size_bytes * (self.world_size - 1)
            bandwidth_gbps = (total_data_transfer / avg_time) / (1024**3)
            
            results[size] = {
                'avg_time': avg_time,
                'bandwidth_gbps': bandwidth_gbps,
                'data_size_mb': data_size_bytes / (1024**2)
            }
            
            if self.rank == 0:
                print(f"ReduceScatter 张量大小 {size} ({data_size_bytes/(1024**2):.2f}MB), "
                      f"平均时间 {avg_time:.6f}s, 带宽 {bandwidth_gbps:.2f} GB/s")
        
        return results
    
    def test_latency(self, tensor_size: int = 1024, num_iters: int = 100) -> Dict:
        """测试通信延迟"""
        if self.rank == 0:
            print(f"\n开始延迟测试...")
        
        tensor = torch.ones(tensor_size, dtype=torch.float32, device=self.device)
        
        # 预热
        for _ in range(10):
            dist.all_reduce(tensor)
        
        # 同步所有进程
        dist.barrier()
        
        times = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start_time = time.time()
            
            dist.all_reduce(tensor)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # 计算统计信息
        times = np.array(times)
        latency_stats = {
            'mean_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'min_latency_ms': np.min(times) * 1000,
            'max_latency_ms': np.max(times) * 1000,
            'p50_latency_ms': np.percentile(times, 50) * 1000,
            'p95_latency_ms': np.percentile(times, 95) * 1000,
            'p99_latency_ms': np.percentile(times, 99) * 1000
        }
        
        if self.rank == 0:
            print(f"延迟统计:")
            print(f"  平均: {latency_stats['mean_latency_ms']:.3f}ms")
            print(f"  P95: {latency_stats['p95_latency_ms']:.3f}ms")
            print(f"  P99: {latency_stats['p99_latency_ms']:.3f}ms")
        
        return latency_stats
    
    def test_cross_node_communication(self, tensor_size: int = 1024*1024) -> Dict:
        """测试跨节点通信性能"""
        if self.rank == 0:
            print(f"\n开始跨节点通信测试...")
        
        # 创建张量
        tensor = torch.ones(tensor_size, dtype=torch.float32, device=self.device) * self.rank
        
        # 测试跨节点AllReduce
        dist.barrier()
        torch.cuda.synchronize()
        start_time = time.time()
        
        dist.all_reduce(tensor)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        cross_node_time = end_time - start_time
        data_size_mb = tensor_size * 4 / (1024**2)
        cross_node_bandwidth = (data_size_mb * (self.world_size - 1)) / cross_node_time / 1024  # GB/s
        
        if self.rank == 0:
            print(f"跨节点通信性能:")
            print(f"  数据大小: {data_size_mb:.2f}MB")
            print(f"  通信时间: {cross_node_time:.6f}s")
            print(f"  带宽: {cross_node_bandwidth:.2f} GB/s")
        
        return {
            'cross_node_time': cross_node_time,
            'cross_node_bandwidth_gbps': cross_node_bandwidth,
            'data_size_mb': data_size_mb
        }
    
    def run_comprehensive_test(self, tensor_sizes: List[int], num_iters: int = 10) -> Dict:
        """运行综合测试"""
        if self.rank == 0:
            print(f"\n开始多机多卡综合通信测试...")
            print(f"总进程数: {self.world_size}")
            print(f"节点数: {self.node_rank + 1}")
            print(f"每节点进程数: {self.local_world_size}")
        
        # 初始化分布式环境
        self.init_distributed()
        
        try:
            # 等待所有进程初始化完成
            dist.barrier()
            
            # 运行各种测试
            results = {
                'rank': self.rank,
                'local_rank': self.local_rank,
                'node_rank': self.node_rank,
                'world_size': self.world_size,
                'local_world_size': self.local_world_size,
                'device': str(self.device),
                'hostname': socket.gethostname(),
                'allreduce_bandwidth': self.test_allreduce_bandwidth(tensor_sizes, num_iters),
                'allgather_bandwidth': self.test_allgather_bandwidth(tensor_sizes, num_iters),
                'reduce_scatter_bandwidth': self.test_reduce_scatter_bandwidth(tensor_sizes, num_iters),
                'latency': self.test_latency(tensor_sizes[0] if tensor_sizes else 1024, num_iters * 10),
                'cross_node_communication': self.test_cross_node_communication()
            }
            
            # 同步所有进程
            dist.barrier()
            
            # 只在rank 0保存结果
            if self.rank == 0:
                with open('nccl_multi_node_results.json', 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n测试结果已保存到 nccl_multi_node_results.json")
            
            return results
            
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(description='多机多卡NCCL通信测试')
    parser.add_argument('--tensor-sizes', type=int, nargs='+', 
                       default=[1024, 10240, 102400, 1024000, 10240000],
                       help='测试张量大小列表')
    parser.add_argument('--num-iters', type=int, default=10,
                       help='每个测试的迭代次数')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("错误: CUDA不可用，请检查GPU环境")
        return
    
    # 创建测试器并运行测试
    tester = NCCLMultiNodeTester()
    tester.run_comprehensive_test(args.tensor_sizes, args.num_iters)


if __name__ == "__main__":
    main()
