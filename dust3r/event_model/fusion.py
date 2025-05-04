import torch
import torch.nn as nn
from torch.distributed import all_reduce, ReduceOp

def check_shape_consistency(tensor, name):
    # 获取张量形状并转换为张量
    shape = torch.tensor(tensor.shape, device=tensor.device, dtype=torch.long)  # 例如 [2, 768, 9, 16]
    
    # 创建一个副本用于聚合
    shape_max = shape.clone()
    shape_min = shape.clone()
    
    # 使用 all_reduce 聚合所有 rank 的形状（取最大值和最小值）
    all_reduce(shape_max, op=ReduceOp.MAX)
    all_reduce(shape_min, op=ReduceOp.MIN)
    
    # 检查最大值和最小值是否相同（即形状是否一致）
    shape_list = shape.tolist()
    if torch.equal(shape_max, shape_min):
        print(f"{name} shape consistency: {shape_list} across all ranks")
    else:
        raise RuntimeError(f"{name} shape inconsistency: {shape_list} on rank {torch.distributed.get_rank()}, "
                           f"max shape: {shape_max.tolist()}, min shape: {shape_min.tolist()}")

class ImageEventFusion(nn.Module):
    def __init__(self, event_channels=768, target_channels=1024, target_hw=(24, 24)):
        super().__init__()
        self.conv_adjust = nn.Conv2d(event_channels, target_channels, kernel_size=1)  # 调整通道数

    def forward(self, x, f_event, true_shape):
        event_feat = f_event[-1]

        upsample_layer = nn.Upsample(size=(true_shape[0][0].item() // 16, true_shape[0][1].item() // 16), mode='bilinear', align_corners=False)
        event_feat = upsample_layer(event_feat)

        event_feat = self.conv_adjust(event_feat)

        B, C, H, W = event_feat.size()
        event_feat = event_feat.view(B, C, H * W).transpose(1, 2)

        x = x + event_feat
        return x