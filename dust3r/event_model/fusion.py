import torch
import torch.nn as nn
from torch.distributed import all_reduce, ReduceOp
import math
from ..visualization import visualize_tensors, visualize_tensors_snr
import torch.nn.functional as F

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

class CrossAttention(nn.Module):
    """多头交叉注意力模块，用于event特征对image特征的注意力"""
    
    def __init__(self, dim_q, dim_kv, dim_embed, dim_out, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., rope=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_embed // num_heads
        self.scale = head_dim ** -0.5
        self.dim_embed = dim_embed

        self.q_proj = nn.Linear(dim_q, dim_embed, bias=qkv_bias)
        self.k_proj = nn.Linear(dim_kv, dim_embed, bias=qkv_bias)
        self.v_proj = nn.Linear(dim_kv, dim_embed, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_embed, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        
    def forward(self, query, key, value, query_pos=None, key_pos=None):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        
        # 投影查询、键和值
        q = self.q_proj(query).reshape(B, Nq, self.num_heads, self.dim_embed // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, Nk, self.num_heads, self.dim_embed // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, Nk, self.num_heads, self.dim_embed // self.num_heads).permute(0, 2, 1, 3)
        
        # 应用位置编码（如果有）
        if self.rope is not None and query_pos is not None and key_pos is not None:
            # 检查rope对象的接口
            if hasattr(self.rope, 'apply_rotary'):
                q = self.rope.apply_rotary(q, positions=query_pos)
                k = self.rope.apply_rotary(k, positions=key_pos)
            else:
                # 如果没有apply_rotary方法，则不应用位置编码
                pass
            
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力权重
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, self.dim_embed)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ImageEventFusion(nn.Module):
    def __init__(self, event_channels=768, target_channels=1024):
        super().__init__()
        self.conv_adjust = nn.Conv2d(event_channels, target_channels, kernel_size=1)  # 调整通道数
        self.attention = nn.MultiheadAttention(embed_dim=target_channels, num_heads=8)
        self.norm = nn.LayerNorm(target_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, event_feat, true_shape, snr_map=None,event_blk_idx=None):
        scale = int(math.sqrt(true_shape[0][0].item()*true_shape[0][1].item()/x.shape[1]))

        upsample_layer = nn.Upsample(size=(true_shape[0][0].item() // scale, true_shape[0][1].item() // scale), mode='bilinear', align_corners=False)
        event_feat = upsample_layer(event_feat)

        event_feat = self.conv_adjust(event_feat)

        B, C, H, W = event_feat.size()
        event_feat = event_feat.view(B, C, H * W).transpose(1, 2)

        old_x = x.clone()
        old_event_feat = event_feat.clone()

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

        visualize_tensors_snr(old_x, old_event_feat, snr_map, attn_output, save_path=f"visualization/tensorSNR{event_blk_idx}_visualization.png",true_shape=true_shape)
        return x

class EventImageFusion(nn.Module):
    def __init__(self, event_channels=768, target_channels=1024):
        super().__init__()
        # self.attention = nn.MultiheadAttention(embed_dim=target_channels, num_heads=8)
        self.attention = CrossAttention(dim_q=event_channels, dim_kv=target_channels, dim_embed=256, dim_out=event_channels)
        self.norm = nn.LayerNorm(event_channels)
        self.image_norm = nn.LayerNorm(target_channels)  # 新增：用于归一化图像特征
        self.sigmoid = nn.Sigmoid()
    
        self._init_weights()

    def _init_weights(self):
        # 初始化注意力模块
        # 使用kaiming初始化，更适合注意力机制
        nn.init.kaiming_normal_(self.attention.q_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.attention.k_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.attention.v_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.attention.proj.weight, mode='fan_out', nonlinearity='linear')
        
        # 初始化注意力模块的偏置
        if self.attention.q_proj.bias is not None:
            nn.init.zeros_(self.attention.q_proj.bias)
        if self.attention.k_proj.bias is not None:
            nn.init.zeros_(self.attention.k_proj.bias)
        if self.attention.v_proj.bias is not None:
            nn.init.zeros_(self.attention.v_proj.bias)
        if self.attention.proj.bias is not None:
            nn.init.zeros_(self.attention.proj.bias)

        # 初始化LayerNorm
        # 使用较小的初始值来避免训练初期的不稳定性
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)
        nn.init.constant_(self.image_norm.weight, 1.0)
        nn.init.constant_(self.image_norm.bias, 0.0)

        # 可选：添加一个小的扰动来打破对称性
        with torch.no_grad():
            self.norm.weight.add_(torch.randn_like(self.norm.weight) * 0.01)
            self.image_norm.weight.add_(torch.randn_like(self.image_norm.weight) * 0.01)

    def forward(self, x, event_feat):
        # 首先对图像特征进行归一化
        x = self.image_norm(x)  # 新增：归一化图像特征
        # TODO: lack enevt feature norm

        old_x = x.clone()
        old_event_feat = event_feat.clone()

        # if event_blk_idx==3 and snr_map is not None:
        #     snr_map = upsample_layer(snr_map).view(B, 1, H * W).transpose(1, 2)  # [B, H*W, 1]
        #     snr_weight = self.sigmoid(snr_map)
        #     output = x * snr_weight + event_feat * (1 - snr_weight)
        #     # visualize_tensors_snr(old_x, old_event_feat, snr_map, output, save_path=f"visualization/tensor{event_blk_idx}_visualization.png",true_shape=true_shape)
        #     return output

        # x = x + event_feat

        # Cross-Attention
        # x 作为 query，event_feat 作为 key 和 value
        # 为了 MultiheadAttention，调整维度为 [seq_len, batch, embed_dim]
        # x = x.transpose(0, 1)  # [576, 2, 1024]
        # event_feat = event_feat.transpose(0, 1)  # [576, 2, 1024]

        # 计算注意力
        # attn_output, _ = self.attention(query=event_feat, key=x, value=x)
        attn_output = self.attention(query=event_feat, key=x, value=x)

        # 恢复维度
        # attn_output = attn_output.transpose(0, 1)  # [2, 576, 1024]

        # 残差连接并归一化
        # event_feat = self.norm(event_feat.transpose(0, 1) + attn_output)  # [2, 576, 1024]
        event_feat = self.norm(event_feat+ attn_output)  # [2, 576, 1024]

        # visualize_tensors(old_x, old_event_feat, attn_output, event_feat, save_path=f"visualization/tensor{event_blk_idx}_visualization.png",true_shape=true_shape)
        return event_feat