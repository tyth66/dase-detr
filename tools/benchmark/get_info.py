"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import argparse
from collections import OrderedDict


def _normalize_thread_env():
    for env_name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        value = os.environ.get(env_name)
        if value is None:
            continue

        try:
            if int(value) > 0:
                continue
        except ValueError:
            pass

        os.environ[env_name] = '1'


_normalize_thread_env()

from calflops import calculate_flops
from engine.core import YAMLConfig

import torch
import torch.nn as nn

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

def get_top_modules(model, top_k=5):
    """获取参数量排名前top_k的模块"""
    module_params = OrderedDict()
    
    # 递归遍历所有模块并统计参数量
    for name, module in model.named_modules():
        if name == '':  # 跳过根模块
            continue
            
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if num_params > 0:
            module_type = str(module.__class__).split('.')[-1].strip("'>")
            module_name = f"{name} ({module_type})"
            module_params[module_name] = num_params
    
    # 按参数量排序
    sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return sorted_modules, total_params

def main(args):
    """主函数：加载配置并计算模型指标"""
    # 1. 加载YAML配置文件
    cfg = YAMLConfig(args.config, resume=None)
    
    # 2. 创建用于计算FLOPs的包装模型
    class Model_for_flops(nn.Module):
        def __init__(self):
            super().__init__()
            # 从配置部署模型
            self.model = cfg.model.deploy()
            
        def forward(self, images):
            return self.model(images)
    
    # 3. 实例化模型并设置为评估模式
    model = Model_for_flops().eval()
    
    # 4. 计算模型指标
    flops, macs, Params = calculate_flops(
        model=model,
        input_shape=(1, 3, args.input_size, args.input_size),  # 使用参数化的输入尺寸
        print_detailed=False,
        output_as_string=True,
        output_precision=4)
    
    print("Top 10 Modules by Parameter Count:")
    print("{:<60} {:>15} {:>10}".format("Module Name", "Params", "% Total"))
    print("-" * 90)

    # 5. 打印参数量前5的模块
    top_modules, total_params = get_top_modules(model.model, top_k=10)
    for module_name, params in top_modules:
        percent = (params / total_params) * 100
        print(f"{module_name:<60} {params:>15,} {percent:>9.2f}%")
    
    # 6. 打印整体指标
    print(f"Model Summary:")
    print(f"FLOPs: {flops}")
    print(f"MACs:  {macs}")
    print(f"Params: {Params}\n")

if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='Model Complexity Analyzer')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to model configuration YAML file')
    parser.add_argument('--input-size', type=int, default=640,
                        help='Input image size (default: 640)')
    args = parser.parse_args()
    
    main(args)
