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
        self.attention = nn.MultiheadAttention(embed_dim=target_channels, num_heads=8)
        self.norm = nn.LayerNorm(target_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, f_event, true_shape, snr_map=None):
        event_feat = f_event[-1]

        upsample_layer = nn.Upsample(size=(true_shape[0][0].item() // 16, true_shape[0][1].item() // 16), mode='bilinear', align_corners=False)
        event_feat = upsample_layer(event_feat)

        event_feat = self.conv_adjust(event_feat)

        B, C, H, W = event_feat.size()
        event_feat = event_feat.view(B, C, H * W).transpose(1, 2)

        if snr_map is not None:
            snr_map = upsample_layer(snr_map).view(B, 1, H * W).transpose(1, 2)  # [B, H*W, 1]
            snr_weight = self.sigmoid(snr_map)

            x = x * snr_weight
            event_feat = event_feat*(1 - snr_weight)


        # x = x + event_feat

        # Cross-Attention
        # x 作为 query，event_feat 作为 key 和 value
        # 为了 MultiheadAttention，调整维度为 [seq_len, batch, embed_dim]
        x = x.transpose(0, 1)  # [576, 2, 1024]
        event_feat = event_feat.transpose(0, 1)  # [576, 2, 1024]

        # 计算注意力
        attn_output, _ = self.attention(query=x, key=event_feat, value=event_feat)

        # 恢复维度
        attn_output = attn_output.transpose(0, 1)  # [2, 576, 1024]

        # 残差连接并归一化
        x = self.norm(x.transpose(0, 1) + attn_output)  # [2, 576, 1024]
        return x