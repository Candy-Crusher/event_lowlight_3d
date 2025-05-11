import torch
import torch.nn as nn
from torch.distributed import all_reduce, ReduceOp
import math
from ..visualization import visualize_tensors, visualize_tensors_snr
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        
    def forward(self, query, key, value, query_pos=None, key_pos=None, return_attention_weights=False):
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
        
        if return_attention_weights:
            # 返回注意力权重（平均所有头的注意力）
            attn_weights = attn.mean(dim=1)  # [B, Nq, Nk]
            return x, attn_weights
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
        # 多头注意力
        self.attention = CrossAttention(
            dim_q=event_channels, 
            dim_kv=target_channels, 
            dim_embed=256, 
            dim_out=event_channels,
            num_heads=8  # 添加多头
        )
        
        # 特征归一化层
        self.norm = nn.LayerNorm(event_channels)
        self.image_norm = nn.LayerNorm(target_channels)
        
        # 特征增强模块
        self.feature_enhance = nn.Sequential(
            nn.Linear(event_channels, event_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(event_channels * 2, event_channels),
            nn.LayerNorm(event_channels)
        )
        
        # 注意力权重可视化
        self.attention_weights = None
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        def _init_module(module):
            if isinstance(module, nn.Linear):
                # 使用xavier初始化，更适合注意力机制
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # 初始化所有模块
        self.apply(_init_module)
        
        # 特别初始化注意力模块
        nn.init.xavier_uniform_(self.attention.q_proj.weight)
        nn.init.xavier_uniform_(self.attention.k_proj.weight)
        nn.init.xavier_uniform_(self.attention.v_proj.weight)
        nn.init.xavier_uniform_(self.attention.proj.weight)
        
        # 初始化注意力模块的偏置
        if self.attention.q_proj.bias is not None:
            nn.init.zeros_(self.attention.q_proj.bias)
        if self.attention.k_proj.bias is not None:
            nn.init.zeros_(self.attention.k_proj.bias)
        if self.attention.v_proj.bias is not None:
            nn.init.zeros_(self.attention.v_proj.bias)
        if self.attention.proj.bias is not None:
            nn.init.zeros_(self.attention.proj.bias)

    def forward(self, x, event_feat):
        # 特征归一化
        x = self.image_norm(x)
        event_feat = self.norm(event_feat)
        
        # 保存原始特征用于残差连接
        residual = event_feat
        
        # 计算注意力
        attn_output, self.attention_weights = self.attention(
            query=event_feat, 
            key=x, 
            value=x,
            return_attention_weights=True
        )
        
        # 特征增强
        enhanced_feat = self.feature_enhance(attn_output)
        
        # 残差连接
        output = residual + enhanced_feat
        
        # 最终归一化
        output = self.norm(output)
        
        # 训练时进行梯度裁剪
        if self.training:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        return output
    
    def visualize_attention(self, save_path=None):
        """可视化注意力权重"""
        if self.attention_weights is not None and save_path is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.attention_weights.detach().cpu().numpy())
            plt.colorbar()
            plt.savefig(save_path)
            plt.close()
    
    def get_feature_stats(self, x, event_feat):
        """获取特征统计信息"""
        stats = {
            'x_mean': x.mean().item(),
            'x_std': x.std().item(),
            'event_feat_mean': event_feat.mean().item(),
            'event_feat_std': event_feat.std().item(),
            'attention_weights_mean': self.attention_weights.mean().item() if self.attention_weights is not None else 0
        }
        return stats

class FeatureFusionLayer(nn.Module):
    """特征融合层，支持多种融合策略"""
    
    def __init__(self, dim, fusion_type='attention', num_heads=8):
        super().__init__()
        self.dim = dim
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            # 注意力融合
            self.attention = CrossAttention(
                dim_q=dim,
                dim_kv=dim,
                dim_embed=dim,
                dim_out=dim,
                num_heads=num_heads
            )
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            
        elif fusion_type == 'concat':
            # 拼接融合
            self.fusion_conv = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
            )
            
        elif fusion_type == 'adaptive':
            # 自适应融合
            self.fusion_weights = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, 2),
                nn.Softmax(dim=-1)
            )
            
        # 特征增强
        self.enhance = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        def _init_module(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        self.apply(_init_module)
        
    def forward(self, x1, x2, snr_map=None):
        """
        融合两个特征
        
        Args:
            x1: 第一个特征 [B, L, C]
            x2: 第二个特征 [B, L, C]
            snr_map: SNR map [B, L, 1] 或 None
            
        Returns:
            融合后的特征 [B, L, C]
        """
        if self.fusion_type == 'attention':
            # 注意力融合
            x1 = self.norm1(x1)
            x2 = self.norm2(x2)
            
            # 计算注意力
            attn_output = self.attention(query=x1, key=x2, value=x2)
            
            # 残差连接
            fused = x1 + attn_output
            
        elif self.fusion_type == 'concat':
            # 拼接融合
            concat_feat = torch.cat([x1, x2], dim=-1)
            fused = self.fusion_conv(concat_feat)
            
        elif self.fusion_type == 'adaptive':
            # 自适应融合
            concat_feat = torch.cat([x1, x2], dim=-1)
            weights = self.fusion_weights(concat_feat)  # [B, L, 2]
            fused = weights[:, :, 0:1] * x1 + weights[:, :, 1:2] * x2
            
        # 如果提供了SNR map，使用它来调整融合权重
        if snr_map is not None:
            snr_weight = torch.sigmoid(snr_map)
            fused = fused * snr_weight + x1 * (1 - snr_weight)
            
        # 特征增强
        enhanced = self.enhance(fused)
        
        # 残差连接
        output = fused + enhanced
        
        return output