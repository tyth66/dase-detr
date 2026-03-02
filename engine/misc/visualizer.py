""""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
import torchvision
torchvision.disable_beta_transforms_warning()

__all__ = ['show_sample']

def show_sample(sample):
    """for coco dataset/dataloader
    """
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)

    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()
    plt.show()

def plot_loss_curves(log_file_path):
    """
    从训练日志文件(log.txt)解析数据并绘制损失曲线和验证 mAP 曲线
    
    参数:
        log_file_path: Path - 日志文件路径
    """
    # 确保文件存在
    if not log_file_path.exists():
        print(f"Warning: Log file not found at {log_file_path}")
        return

    # 读取日志数据
    log_data = []
    with open(log_file_path, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                log_data.append(log_entry)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line}")
    
    if not log_data:
        print("No valid log entries found.")
        return

    # 检查数据长度一致性
    epochs = [entry['epoch'] for entry in log_data]
    
    # 创建输出目录
    plots_dir = log_file_path.parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 提取训练损失
    train_losses = {'train_loss': [entry.get('train_loss', np.nan) for entry in log_data]}

    # 提取验证指标（仅保留 mAP@50:95 和 mAP@50）
    val_metrics = {}
    for entry in log_data:
        test_coco_eval_bbox = entry.get('test_coco_eval_bbox', [np.nan] * 12)
        if 'mAP@50:95' not in val_metrics:
            val_metrics['mAP@50:95'] = []
            val_metrics['mAP@50'] = []
        val_metrics['mAP@50:95'].append(test_coco_eval_bbox[0] if len(test_coco_eval_bbox) > 0 else np.nan)
        val_metrics['mAP@50'].append(test_coco_eval_bbox[1] if len(test_coco_eval_bbox) > 1 else np.nan)

    # 绘图调用
    plot_grouped_losses(
        epochs, train_losses,
        title="Training Losses",
        save_path=plots_dir / "training_losses.png")
    plot_validation_metrics(
        epochs, val_metrics,
        title="Validation mAP",
        save_path=plots_dir / "validation_map.png")


def plot_grouped_losses(epochs, losses, title, save_path):
    """绘制总训练损失曲线图并标出最小损失点"""
    plt.figure(figsize=(10, 6))
    
    # 只绘制总损失 train_loss
    train_loss = losses['train_loss']
    
    # 过滤掉NaN值
    valid_indices = [i for i, loss in enumerate(train_loss) if not np.isnan(loss)]
    if not valid_indices:
        print("No valid training loss data to plot.")
        plt.close()
        return
    
    valid_epochs = [epochs[i] for i in valid_indices]
    valid_losses = [train_loss[i] for i in valid_indices]
    
    # 绘制损失曲线
    plt.plot(valid_epochs, valid_losses, label='Total Training Loss', linewidth=2, alpha=0.8)
    
    # 找到最小损失点
    min_loss_idx = np.argmin(valid_losses)
    min_loss_epoch = valid_epochs[min_loss_idx]
    min_loss_value = valid_losses[min_loss_idx]
    
    # 标记最小损失点
    plt.scatter(min_loss_epoch, min_loss_value, color='red', s=100, zorder=5, 
            label=f'Min Loss: {min_loss_value:.4f}\n(Epoch {min_loss_epoch})')
    
    # 在最小损失点添加垂直参考线
    plt.axvline(x=min_loss_epoch, color='red', linestyle='--', alpha=0.5)
    
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 添加最小损失的文本标注
    plt.annotate(f'Min: {min_loss_value:.4f}', 
                xy=(min_loss_epoch, min_loss_value),
                xytext=(min_loss_epoch + len(valid_epochs)*0.05, min_loss_value),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_validation_metrics(epochs, metrics, title, save_path):
    """绘制验证指标曲线图（如 mAP@50, mAP@50:95）"""
    plt.figure(figsize=(10, 6))

    colors = {'mAP@50:95': 'red', 'mAP@50': 'blue'}
    
    for metric_name, values in metrics.items():
        # 过滤掉NaN值
        valid_indices = [i for i, val in enumerate(values) if not np.isnan(val)]
        if not valid_indices:
            print(f"No valid data for {metric_name}")
            continue
            
        valid_epochs = [epochs[i] for i in valid_indices]
        valid_values = [values[i] for i in valid_indices]
        
        color = colors.get(metric_name, None)
        plt.plot(valid_epochs, valid_values, label=metric_name, linewidth=2, color=color)
        
        # 找到最大值及其对应的epoch
        max_value = max(valid_values)
        max_epoch = valid_epochs[valid_values.index(max_value)]
        
        # 在最大值点添加标注
        plt.scatter(max_epoch, max_value, color=color, s=50, zorder=5)
        
        # 智能文本位置：避免边界冲突
        ha = 'left' if max_epoch < max(valid_epochs) * 0.8 else 'right'
        va = 'bottom' if valid_values.index(max_value) < len(valid_values) * 0.8 else 'top'
        
        plt.annotate(f'E{max_epoch}: {max_value:.3f}', 
                    xy=(max_epoch, max_value),
                    xytext=(5, 5) if ha == 'left' else (-5, 5), 
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8),
                    ha=ha, va=va,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()