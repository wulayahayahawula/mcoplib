#!/usr/bin/env python3
"""
简化版本的 fused_add_rms_norm nvbench 性能测试

该脚本提供了一个简单的接口来测试 fused_add_rms_norm 算子性能，
适合快速验证和调试。

使用方法:
    python fused_add_rms_norm_simple.py
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cuda.bench as bench
import torch
import mcoplib._C


def as_torch_cuda_stream(cs: bench.CudaStream, dev: int | None) -> torch.cuda.ExternalStream:
    """将 nvbench 的 CudaStream 转换为 PyTorch 的 ExternalStream"""
    return torch.cuda.ExternalStream(
        stream_ptr=cs.addressof(), device=torch.cuda.device(dev)
    )


def simple_fused_add_rms_norm_benchmark(state: bench.State) -> None:
    """
    简化的 fused_add_rms_norm benchmark
    使用固定的小规模数据进行测试
    """
    # 设置最小样本数
    state.set_min_samples(100)

    # 获取设备和流
    dev_id = state.get_device()
    tc_s = as_torch_cuda_stream(state.get_stream(), dev_id)

    # 固定测试参数
    num_tokens = 1024
    hidden_size = 512
    dtype = torch.float16
    epsilon = 1e-6

    # 添加摘要信息
    state.add_summary("NumTokens", str(num_tokens))
    state.add_summary("HiddenSize", str(hidden_size))
    state.add_summary("DType", "float16")

    # 计算统计信息
    total_elements = num_tokens * hidden_size
    state.add_element_count(total_elements)

    element_size = 2  # float16
    read_bytes = total_elements * element_size * 2 + hidden_size * element_size
    write_bytes = total_elements * element_size * 2
    state.add_global_memory_reads(read_bytes)
    state.add_global_memory_writes(write_bytes)

    # 分配测试数据
    with torch.cuda.stream(tc_s):
        hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device=f'cuda:{dev_id}')
        residual = torch.randn((num_tokens, hidden_size), dtype=dtype, device=f'cuda:{dev_id}')
        weight = torch.randn((hidden_size,), dtype=dtype, device=f'cuda:{dev_id}')

    # 定义 launcher
    def launcher(launch: bench.Launch) -> None:
        launch_tc_s = as_torch_cuda_stream(launch.get_stream(), dev_id)
        with torch.cuda.stream(launch_tc_s):
            torch.ops._C.fused_add_rms_norm(
                hidden_states,
                residual,
                weight,
                epsilon
            )

    # 执行 benchmark
    state.exec(launcher, sync=True)


if __name__ == "__main__":
    # 注册 benchmark
    bench.register(simple_fused_add_rms_norm_benchmark)

    # 运行
    bench.run_all_benchmarks(sys.argv)
