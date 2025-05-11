import torch
import torch.nn as nn
import torch.nn.functional as F
from dust3r.utils.goem_opt import WarpImage

class IntensityBasedEventLoss(nn.Module):
    """
    基于图像强度变化的事件损失函数
    根据事件相机的物理原理，事件由像素强度变化引起
    
    实现逻辑：
    1. 使用目标坐标1和2以及初始坐标1和2来检索t和t'之间的图像和事件流
    2. 计算intensity(tgt_coord) - intensity(coord)，这应当被event_stream(coord)所监督
    """
    def __init__(self):
        super().__init__()
        self.image_warper = WarpImage()
        
    def forward(self, img1, img2, flow, event_voxels, valid_mask=None, threshold=0.1):
        """
        计算基于图像强度变化的事件损失
        
        Args:
            img1: 第一帧图像 [B, 3, H, W]
            img2: 第二帧图像 [B, 3, H, W]
            flow: 从img1到img2的光流 [B, 2, H, W]
            event_voxels: 事件体素表示 [B, C, H, W]
            valid_mask: 有效区域掩码 [B, 1, H, W]
            threshold: 强度变化阈值，低于此值的变化不会触发事件
            
        Returns:
            loss: 事件一致性损失值
        """
        B, _, H, W = flow.shape
        
        # 将RGB图像转换为灰度图
        gray1 = 0.299 * img1[:, 0:1] + 0.587 * img1[:, 1:2] + 0.114 * img1[:, 2:3]
        gray2 = 0.299 * img2[:, 0:1] + 0.587 * img2[:, 1:2] + 0.114 * img2[:, 2:3]
        
        # 根据光流将img2变形到img1的坐标系
        warped_gray2 = self.image_warper(gray2, flow)
        
        # 计算强度变化: I(tgt_coord) - I(coord)
        intensity_change = warped_gray2 - gray1
        
        # 使用tanh函数进行标准化，以捕获极性特征
        normalized_change = torch.tanh(intensity_change * 5.0)  # 乘以5以增强对比度
        
        # 创建预测事件体素表示
        # 将标准化后的强度变化转换为事件表示
        # 假设event_voxels有C个通道
        C = event_voxels.shape[1]
        
        # 创建一个简单的事件表示，基于强度变化的极性和幅度
        pred_events = torch.zeros_like(event_voxels)
        
        # 正极性事件 (intensity增加)
        positive_change = F.relu(normalized_change)
        # 负极性事件 (intensity减少)
        negative_change = F.relu(-normalized_change)
        
        # 分配到不同时间片(通道)，这里使用简化方法
        # 实际上，事件的时间分布更复杂，这里只是一种近似
        if C >= 2:
            # 至少有2个通道时，分配正负极性
            pred_events[:, 0] = positive_change.squeeze(1)  # 正极性
            pred_events[:, 1] = negative_change.squeeze(1)  # 负极性
            
            # 如果有更多通道，可以根据强度大小分配到不同时间片
            if C > 2:
                # 将剩余通道平均分配
                remaining_channels = C - 2
                for i in range(remaining_channels):
                    weight = (i + 1) / (remaining_channels + 1)
                    pred_events[:, i+2] = (positive_change * weight + negative_change * (1-weight)).squeeze(1)
        else:
            # 只有1个通道时，直接使用绝对变化
            pred_events[:, 0] = (positive_change + negative_change).squeeze(1)
        
        # 创建事件掩码，光流大于阈值的区域才可能产生事件
        flow_magnitude = torch.sqrt(flow[:, 0, ...]**2 + flow[:, 1, ...]**2)
        event_mask = (flow_magnitude > threshold).float().unsqueeze(1)
        
        # 计算损失
        if valid_mask is not None:
            # 扩展掩码到所有事件通道
            valid_mask_expanded = valid_mask.expand(-1, C, -1, -1)
            
            # 只在有效区域和潜在事件区域计算损失
            combined_mask = valid_mask_expanded * event_mask.expand(-1, C, -1, -1)
            mask_sum = combined_mask.sum() + 1e-6
            
            # L1损失
            l1_loss = F.l1_loss(
                pred_events * combined_mask, 
                event_voxels * combined_mask, 
                reduction='sum'
            ) / mask_sum
            
            # 相关性损失 - 计算预测和真实事件数据的相关性
            pred_flat = (pred_events * combined_mask).reshape(B, C, -1)
            gt_flat = (event_voxels * combined_mask).reshape(B, C, -1)
            
            # 对每个批次和通道计算相关性，并取平均值
            corr_loss = 0
            for b in range(B):
                for c in range(C):
                    if combined_mask[b, c].sum() > 0:
                        pred_norm = pred_flat[b, c] - pred_flat[b, c].mean()
                        gt_norm = gt_flat[b, c] - gt_flat[b, c].mean()
                        corr = (pred_norm * gt_norm).sum() / (torch.norm(pred_norm) * torch.norm(gt_norm) + 1e-6)
                        corr_loss += (1.0 - corr) / (B * C)
            
            # 组合损失
            loss = l1_loss + 0.5 * corr_loss
        else:
            # 如果没有有效掩码，仅使用L1损失
            loss = F.l1_loss(pred_events, event_voxels, reduction='mean')
        
        return loss, pred_events

# 为了兼容性，恢复原来的compute_event_loss函数
def compute_event_loss(ego_flow, event_gt, imgs=None, valid_mask=None, threshold=0.1):
    """
    计算事件一致性损失，适用于MVSEC数据集的事件体素格式
    
    Args:
        ego_flow: 从姿态和深度计算得到的光流 [B, 2, H, W] 或 [2, H, W]
        event_gt: 真实事件数据 [B, C, H, W] 或 [C, H, W]（MVSEC使用5通道的体素网格表示）
        imgs: 用于计算强度变化的图像对 [B, 2, 3, H, W]，如果为None则回退到基于流的方法
        valid_mask: 有效区域掩码 [B, 1, H, W] 或 [H, W]
        threshold: 事件触发阈值
            
    Returns:
        event_loss: 事件一致性损失值
    """
    # 检查并标准化输入维度
    if ego_flow.dim() == 3:  # [2, H, W]
        ego_flow = ego_flow.unsqueeze(0)  # 添加批处理维度 [1, 2, H, W]
    
    if event_gt.dim() == 3:  # [C, H, W]
        event_gt = event_gt.unsqueeze(0)  # 添加批处理维度 [1, C, H, W]
    
    if valid_mask is not None and valid_mask.dim() == 2:  # [H, W]
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)  # 添加批处理维度 [1, 1, H, W]
    
    # 确保所有输入都有正确的批处理维度
    B, _, H, W = ego_flow.shape
    C = event_gt.shape[1]
    
    # 检查并解决尺寸不匹配问题
    event_H, event_W = event_gt.shape[-2:]
    if event_H != H or event_W != W:
        print(f"警告：事件尺寸 ({event_H}, {event_W}) 与光流尺寸 ({H}, {W}) 不匹配，正在调整...")
        # 调整事件数据以匹配光流尺寸
        event_gt = torch.nn.functional.interpolate(
            event_gt, size=(H, W), mode='bilinear', align_corners=False
        )
        print(f"事件数据已调整为尺寸: ({H}, {W})")
    
    if imgs is not None:
        # 使用强度变化方法计算损失
        loss_fn = IntensityBasedEventLoss()
        
        # 确保img有正确的维度 [B, 2, 3, H, W]
        if imgs.dim() == 4:  # [2, 3, H, W]
            imgs = imgs.unsqueeze(0)
        
        img1, img2 = imgs[:, 0], imgs[:, 1]
        
        # 检查图像尺寸是否与光流匹配
        img_H, img_W = img1.shape[-2:]
        if img_H != H or img_W != W:
            print(f"警告：图像尺寸 ({img_H}, {img_W}) 与光流尺寸 ({H}, {W}) 不匹配，正在调整...")
            # 调整图像数据以匹配光流尺寸
            img1 = torch.nn.functional.interpolate(img1, size=(H, W), mode='bilinear', align_corners=False)
            img2 = torch.nn.functional.interpolate(img2, size=(H, W), mode='bilinear', align_corners=False)
            print(f"图像数据已调整为尺寸: ({H}, {W})")
            
        loss, _ = loss_fn(img1, img2, ego_flow, event_gt, valid_mask, threshold)
        return loss
    else:
        # 回退到基于流的方法
        # 计算光流大小和方向
        flow_magnitude = torch.sqrt(ego_flow[:, 0, ...]**2 + ego_flow[:, 1, ...]**2)
        flow_direction = torch.atan2(ego_flow[:, 1, ...], ego_flow[:, 0, ...])
        
        # 创建事件预测掩码，根据光流大小判断是否有事件
        event_mask = (flow_magnitude > threshold).float().unsqueeze(1)
        
        # 根据光流大小和方向生成模拟的事件体素
        pred_events = torch.zeros_like(event_gt)
        
        # 将光流转换为事件激活
        for c in range(C):
            weight = (c + 0.5) / C  # 将通道分布在[0.5/C, 1.5/C, ..., (C-0.5)/C]
            channel_activation = flow_magnitude * torch.cos(flow_direction - weight * torch.pi)
            pred_events[:, c] = F.relu(channel_activation)
            
        # 标准化预测事件，使平均值与真实事件相匹配
        if event_gt.sum() > 0:
            pred_sum = pred_events.sum()
            gt_sum = event_gt.sum()
            if pred_sum > 0:
                pred_events = pred_events * (gt_sum / pred_sum)
        
        # 计算预测事件体素与真实事件体素之间的差异
        if valid_mask is not None:
            valid_mask_expanded = valid_mask.expand(-1, C, -1, -1)
            combined_mask = valid_mask_expanded * event_mask.expand(-1, C, -1, -1)
            mask_sum = combined_mask.sum() + 1e-6
            
            l1_loss = F.l1_loss(
                pred_events * combined_mask, 
                event_gt * combined_mask, 
                reduction='sum'
            ) / mask_sum
            
            # 相关性损失
            pred_flat = (pred_events * combined_mask).reshape(B, C, -1)
            gt_flat = (event_gt * combined_mask).reshape(B, C, -1)
            
            corr_loss = 0
            for b in range(B):
                for c in range(C):
                    if combined_mask[b, c].sum() > 0:
                        pred_norm = pred_flat[b, c] - pred_flat[b, c].mean()
                        gt_norm = gt_flat[b, c] - gt_flat[b, c].mean()
                        corr = (pred_norm * gt_norm).sum() / (torch.norm(pred_norm) * torch.norm(gt_norm) + 1e-6)
                        corr_loss += (1.0 - corr) / (B * C)
            
            loss = l1_loss + 0.5 * corr_loss
        else:
            loss = F.l1_loss(pred_events, event_gt, reduction='mean')
        
        return loss 