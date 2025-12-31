#!/usr/bin/env python3
"""
使用 nvbench 测试 fused_add_rms_norm 算子性能

该脚本使用 NVBench Python API 来测试 mcoplib 中 fused_add_rms_norm 算子的性能。
支持多种数据形状和数据类型的性能测试。

使用方法:
    python fused_add_rms_norm_nvbench.py --help
    python fused_add_rms_norm_nvbench.py --devices 0
    python fused_add_rms_norm_nvbench.py --list

示例:
    # 在特定设备上运行所有测试
    python fused_add_rms_norm_nvbench.py --devices 0

    # 列出可用的 benchmark
    python fused_add_rms_norm_nvbench.py --list

    # 以 JSON 格式输出结果
    python fused_add_rms_norm_nvbench.py --devices 0 --json
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cuda.bench as bench
import torch

# 导入 mcoplib 算子
try:
    import mcoplib._C
except ImportError as e:
    print(f"警告: 无法导入 mcoplib._C: {e}")
    print("请确保 mcoplib 已正确安装")


def as_torch_cuda_stream(cs: bench.CudaStream, dev: int | None) -> torch.cuda.ExternalStream:
    """
    将 nvbench 的 CudaStream 转换为 PyTorch 的 ExternalStream

    Args:
        cs: nvbench 的 CudaStream 对象
        dev: 设备 ID

    Returns:
        PyTorch 的 ExternalStream 对象
    """
    return torch.cuda.ExternalStream(
        stream_ptr=cs.addressof(), device=torch.cuda.device(dev)
    )


def fused_add_rms_norm_benchmark(state: bench.State) -> None:
    """
    使用 nvbench 测试 fused_add_rms_norm 算子性能

    该函数测试 torch.ops._C.fused_add_rms_norm 的性能，
    支持不同的数据形状和类型。

    Args:
        state: nvbench 的 State 对象，用于管理 benchmark 状态
    """
    # 设置节流阈值（可选）
    state.set_throttle_threshold(0.25)

    # 获取设备和流
    dev_id = state.get_device()
    tc_s = as_torch_cuda_stream(state.get_stream(), dev_id)

    # 从 state 获取参数，如果没有则使用默认值
    num_tokens = state.get_int64_or_default("NumTokens", 1024)
    hidden_size = state.get_int64_or_default("HiddenSize", 512)

    # 数据类型：0=float16, 1=float32, 2=bfloat16
    dtype_id = state.get_int64_or_default("DType", 0)

    dtype_map = {
        0: (torch.float16, "float16"),
        1: (torch.float32, "float32"),
        2: (torch.bfloat16, "bfloat16")
    }

    dtype, dtype_name = dtype_map.get(dtype_id, (torch.float16, "float16"))

    # 添加摘要信息
    state.add_summary("DType", dtype_name)
    state.add_summary("NumTokens", str(num_tokens))
    state.add_summary("HiddenSize", str(hidden_size))

    # 计算元素数量
    total_elements = num_tokens * hidden_size
    state.add_element_count(total_elements)

    # 计算内存访问量（读取和写入）
    # fused_add_rms_norm 操作：
    # - 读取 input, residual, weight
    # - 写入 input (in-place), residual (in-place)
    element_size = 2 if dtype in [torch.float16, torch.bfloat16] else 4
    read_bytes = total_elements * element_size * 2 + hidden_size * element_size  # input + residual + weight
    write_bytes = total_elements * element_size * 2  # input + residual (both modified in-place)
    state.add_global_memory_reads(read_bytes)
    state.add_global_memory_writes(write_bytes)

    # epsilon 参数
    epsilon = 1e-6

    # 在流上分配测试数据
    with torch.cuda.stream(tc_s):
        # 创建测试张量
        hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device=f'cuda:{dev_id}')
        residual = torch.randn((num_tokens, hidden_size), dtype=dtype, device=f'cuda:{dev_id}')
        weight = torch.randn((hidden_size,), dtype=dtype, device=f'cuda:{dev_id}')

    # 定义 launcher 函数
    def launcher(launch: bench.Launch) -> None:
        """
        单次启动函数，执行 fused_add_rms_norm 算子

        Args:
            launch: nvbench 的 Launch 对象
        """
        # 获取当前启动的流
        launch_tc_s = as_torch_cuda_stream(launch.get_stream(), dev_id)

        # 在 PyTorch 流中执行算子
        with torch.cuda.stream(launch_tc_s):
            # 调用 fused_add_rms_norm 算子
            # 注意：该算子是 in-place 操作，会直接修改 hidden_states 和 residual
            torch.ops._C.fused_add_rms_norm(
                hidden_states,
                residual,
                weight,
                epsilon
            )

    # 执行 benchmark
    # sync=True 表示 nvbench 会在每次执行后同步 GPU
    state.exec(launcher, sync=True)


if __name__ == "__main__":
    # 注册 benchmark
    benchmark = bench.register(fused_add_rms_norm_benchmark)

    # 添加测试轴（参数组合）

    # 添加数据类型轴
    benchmark.add_int64_axis("DType", [0, 1])  # 0=float16, 1=float32

    # 添加不同的大小组合，模拟不同的工作负载
    # 小规模
    benchmark.add_int64_axis("NumTokens", [128, 256])
    benchmark.add_int64_axis("HiddenSize", [512, 768])

    # 中规模
    benchmark.add_int64_axis("NumTokens", [1024, 2048])
    benchmark.add_int64_axis("HiddenSize", [1024, 2048])

    # 大规模
    benchmark.add_int64_axis("NumTokens", [4096])
    benchmark.add_int64_axis("HiddenSize", [4096])

    # 运行所有 benchmarks
    # sys.argv 包含命令行参数，nvbench 会解析这些参数
    bench.run_all_benchmarks(sys.argv)
