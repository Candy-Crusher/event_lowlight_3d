import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class SpatialConsistencyLoss(nn.Module):
    """
    空间一致性损失：确保增强后的图像与原始图像在局部梯度上保持一致
    参考自Zero-DCE论文
    """
    def __init__(self, patch_size=4):
        super(SpatialConsistencyLoss, self).__init__()
        self.patch_size = patch_size
        # 创建平均卷积核，形状为(out_channels, in_channels, kH, kW)
        # 确保形状符合F.conv2d的要求
        self.avg_kernel = torch.ones(1, 1, patch_size, patch_size, dtype=torch.float32) / (patch_size * patch_size)
        self.avg_kernel.requires_grad = False  # 卷积核不需要学习

    def _to_gray(self, img_rgb):
        # (B, C, H, W) -> (B, 1, H, W)
        # 使用亮度转换公式
        return 0.299 * img_rgb[:, 0:1, :, :] + 0.587 * img_rgb[:, 1:2, :, :] + 0.114 * img_rgb[:, 2:3, :, :]

    def _get_gradients(self, img_gray):
        # 对图像进行填充以保持卷积后尺寸不变
        img_padded_x = F.pad(img_gray, (0, 1, 0, 0), mode='replicate')  # 右侧填充
        img_padded_y = F.pad(img_gray, (0, 0, 0, 1), mode='replicate')  # 底部填充

        # 计算水平和垂直梯度
        grad_x = torch.abs(img_padded_x[:, :, :, 1:] - img_padded_x[:, :, :, :-1])
        grad_y = torch.abs(img_padded_y[:, :, 1:, :] - img_padded_y[:, :, :-1, :])
        return grad_x, grad_y

    def forward(self, enhanced_image, low_light_image):
        # 确保卷积核在正确的设备上
        if self.avg_kernel.device != enhanced_image.device:
            self.avg_kernel = self.avg_kernel.to(enhanced_image.device)

        # 转换为灰度图
        enhanced_gray = self._to_gray(enhanced_image)
        low_light_gray = self._to_gray(low_light_image)

        # 使用卷积计算局部平均强度
        padding_size = self.patch_size // 2
        # 不再squeeze卷积核，直接使用原始形状(out_channels, in_channels, kH, kW)
        kernel_for_conv = self.avg_kernel

        # 修复卷积操作，确保卷积核形状为(out_channels, in_channels, kH, kW)
        avg_enhanced = F.conv2d(enhanced_gray, kernel_for_conv, stride=1, padding=padding_size)
        avg_low_light = F.conv2d(low_light_gray, kernel_for_conv, stride=1, padding=padding_size)
        
        # 计算梯度
        grad_enhanced_x, grad_enhanced_y = self._get_gradients(avg_enhanced)
        grad_low_light_x, grad_low_light_y = self._get_gradients(avg_low_light)

        # 计算损失
        loss_x = F.mse_loss(grad_enhanced_x, grad_low_light_x)
        loss_y = F.mse_loss(grad_enhanced_y, grad_low_light_y)
        
        return loss_x + loss_y

class ExposureControlLoss(nn.Module):
    """
    曝光控制损失：确保增强后的图像曝光度接近预设值
    参考自Zero-DCE论文
    """
    def __init__(self, patch_size=16, well_exposed_level=0.6):
        super(ExposureControlLoss, self).__init__()
        self.patch_size = patch_size
        self.well_exposed_level = well_exposed_level

    def forward(self, enhanced_image):
        # 计算灰度平均强度
        enhanced_gray_mean = torch.mean(enhanced_image, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # 计算非重叠patch的平均强度
        avg_intensity_patches = F.avg_pool2d(enhanced_gray_mean, 
                                             kernel_size=self.patch_size, 
                                             stride=self.patch_size)
        
        # 计算与理想曝光度的L1损失
        loss = torch.mean(torch.abs(avg_intensity_patches - self.well_exposed_level))
        return loss

class ColorConstancyLoss(nn.Module):
    """
    颜色恒常性损失：确保增强后的图像在各颜色通道间保持平衡
    参考自Zero-DCE论文
    """
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, enhanced_image):
        # 计算每个通道的平均值
        mean_R = torch.mean(enhanced_image[:, 0, :, :], dim=[1, 2])  # (B)
        mean_G = torch.mean(enhanced_image[:, 1, :, :], dim=[1, 2])  # (B)
        mean_B = torch.mean(enhanced_image[:, 2, :, :], dim=[1, 2])  # (B)
        
        # 计算通道间差异的平方和，然后取批次平均
        loss_R_G = torch.pow(mean_R - mean_G, 2)
        loss_R_B = torch.pow(mean_R - mean_B, 2)
        loss_G_B = torch.pow(mean_G - mean_B, 2)
        
        loss = torch.mean(loss_R_G + loss_R_B + loss_G_B)
        return loss

class IlluminationSmoothnessLoss(nn.Module):
    """
    照明平滑损失：确保增强的照明是平滑的
    参考自Zero-DCE论文
    """
    def __init__(self, l2_tv=True):
        super(IlluminationSmoothnessLoss, self).__init__()
        self.l2_tv = l2_tv

    def forward(self, enhanced_illumination):
        # 计算水平和垂直梯度
        delta_h = enhanced_illumination[:, :, 1:, :] - enhanced_illumination[:, :, :-1, :]  # (B, 1, H-1, W)
        delta_w = enhanced_illumination[:, :, :, 1:] - enhanced_illumination[:, :, :, :-1]  # (B, 1, H, W-1)
        
        if self.l2_tv:
            # 根据原论文，裁剪以确保相同大小的求和 (H-1, W-1)
            abs_delta_h_cropped = torch.abs(delta_h[:, :, :, :-1])
            abs_delta_w_cropped = torch.abs(delta_w[:, :, :-1, :])
            loss = torch.mean(torch.pow(abs_delta_h_cropped + abs_delta_w_cropped, 2))
        else:
            # L1 TV
            loss = torch.mean(torch.abs(delta_h)) + torch.mean(torch.abs(delta_w))
            
        return loss

class ReflectanceSmoothnessLoss(nn.Module):
    def __init__(self, l2_tv=True): # Default to L2 TV (sum of squared gradients)
        super(ReflectanceSmoothnessLoss, self).__init__()
        self.l2_tv = l2_tv

    def forward(self, reflectance):
        # reflectance is (B, 3, H, W)
        # Horizontal gradients
        delta_h = reflectance[:, :, 1:, :] - reflectance[:, :, :-1, :]
        # Vertical gradients
        delta_w = reflectance[:, :, :, 1:] - reflectance[:, :, :, :-1]
        
        if self.l2_tv:
            loss = torch.mean(torch.pow(delta_h, 2)) + torch.mean(torch.pow(delta_w, 2))
        else: # L1 TV
            loss = torch.mean(torch.abs(delta_h)) + torch.mean(torch.abs(delta_w))
        return loss

def estimate_illumination_reflectance(image, epsilon=1e-6):
    """
    根据Retinex理论，估计图像的照明图和反射率图
    Args:
        image: 输入图像，形状为(B, 3, H, W)
        epsilon: 避免除零的小常数
    Returns:
        illumination: 照明图，形状为(B, 1, H, W)
        reflectance: 反射率图，形状为(B, 3, H, W)
    """
    # 简单方法：用RGB通道平均值估计照明
    illumination = torch.mean(image, dim=1, keepdim=True)  # (B, 1, H, W)
    
    # 根据Retinex理论，反射率 = 图像 / 照明
    reflectance = image / (illumination + epsilon)
    
    return illumination, reflectance

class ZeroDCELoss(nn.Module):
    """
    基于Zero-DCE的综合增强损失函数
    组合了空间一致性、曝光控制、颜色恒常性和照明平滑性损失
    """
    def __init__(self, 
                 w_spa=1.0,          # 空间一致性损失权重
                 w_exp=1.0,          # 曝光控制损失权重
                 w_col=0.5,          # 颜色恒常性损失权重
                 w_tvI=20.0,         # 照明平滑损失权重
                 w_tvR=0.01,         # 反射率平滑损失权重(可选，可设为0禁用)
                 spa_patch_size=4,   # 空间一致性损失的patch大小
                 exp_patch_size=16,  # 曝光控制损失的patch大小
                 exp_well_exposed_level=0.6, # 理想曝光度
                 tv_l2=True          # 是否使用L2 TV损失
                ):
        super(ZeroDCELoss, self).__init__()
        self.w_spa = w_spa
        self.w_exp = w_exp
        self.w_col = w_col
        self.w_tvI = w_tvI
        self.w_tvR = w_tvR
        
        # 初始化各个损失函数
        self.spatial_loss = SpatialConsistencyLoss(patch_size=spa_patch_size)
        self.exposure_loss = ExposureControlLoss(patch_size=exp_patch_size, well_exposed_level=exp_well_exposed_level)
        self.color_loss = ColorConstancyLoss()
        self.illumination_smooth_loss = IlluminationSmoothnessLoss(l2_tv=tv_l2)
        if self.w_tvR > 0:
            self.reflectance_smooth_loss = ReflectanceSmoothnessLoss(l2_tv=tv_l2)
        else:
            self.reflectance_smooth_loss = None
    
    def forward(self, enhanced_image, original_image, event_voxel=None, snr_map=None, illumination_map=None, reflectance_map=None):
        """
        计算增强图像的综合损失
        Args:
            enhanced_image: 增强后的图像，形状为(B, 3, H, W)
            original_image: 原始低光图像，形状为(B, 3, H, W)
            event_voxel: 可选，事件体素数据
            snr_map: 可选，信噪比图
            illumination_map: 增强后的照明图(B, 1, H, W)
            reflectance_map: 计算出的反射率图(B, 3, H, W)
        Returns:
            total_loss: 总损失
            losses_dict: 各组成部分损失的字典，用于日志记录
        """
        # 如果未提供照明图和反射率图，尝试估计它们
        if illumination_map is None:
            # 如果提供了SNR图，可尝试使用它
            if snr_map is not None and snr_map.shape[1] == 1:
                illumination_map = snr_map
            else:
                # 否则根据Retinex理论估计
                illumination_map, _ = estimate_illumination_reflectance(enhanced_image)
        
        if reflectance_map is None and self.w_tvR > 0:
            # 仅当需要计算反射率平滑损失时估计反射率
            _, reflectance_map = estimate_illumination_reflectance(enhanced_image)
        
        # 计算各部分损失
        loss_spa = self.spatial_loss(enhanced_image, original_image)
        loss_exp = self.exposure_loss(enhanced_image)
        loss_col = self.color_loss(enhanced_image)
        loss_tvI = self.illumination_smooth_loss(illumination_map)
        
        # 计算加权总损失
        total_loss_val = (self.w_spa * loss_spa + 
                          self.w_exp * loss_exp + 
                          self.w_col * loss_col + 
                          self.w_tvI * loss_tvI)
        
        # 如果需要，计算反射率平滑损失
        loss_tvR_val = 0.0
        if self.reflectance_smooth_loss is not None and self.w_tvR > 0 and reflectance_map is not None:
            loss_tvR = self.reflectance_smooth_loss(reflectance_map)
            total_loss_val += self.w_tvR * loss_tvR
            loss_tvR_val = loss_tvR.item() # 用于日志记录
        
        # 返回总损失和各部分损失
        losses_dict = {
            "enhancement_spatial_loss": loss_spa.item(),
            "enhancement_exposure_loss": loss_exp.item(),
            "enhancement_color_loss": loss_col.item(),
            "enhancement_illumination_loss": loss_tvI.item(),
            "enhancement_reflectance_loss": loss_tvR_val,
            "enhancement_total_loss": total_loss_val.item()
        }
        
        return total_loss_val, losses_dict

def visualize_enhancement_result(original_image, enhanced_image, illumination_map=None, snr_map=None, save_path=None):
    """
    可视化图像增强结果，包括原始图像、增强图像、照明图和SNR图（如果提供）
    Args:
        original_image: 原始图像，形状为(B, 3, H, W)或(3, H, W)
        enhanced_image: 增强后的图像，形状为(B, 3, H, W)或(3, H, W)
        illumination_map: 可选，照明图，形状为(B, 1, H, W)或(1, H, W)
        snr_map: 可选，SNR图，形状为(B, 1, H, W)或(1, H, W)
        save_path: 可选，保存路径，如果提供则保存图像
    Returns:
        vis_image: 可视化结果，形状为(H, W*n, 3)，其中n取决于提供的图像数量
    """
    import numpy as np
    import cv2
    import torch
    import os
    from torchvision.utils import make_grid
    
    # 确保输入为批次格式
    if original_image.dim() == 3:
        original_image = original_image.unsqueeze(0)
    if enhanced_image.dim() == 3:
        enhanced_image = enhanced_image.unsqueeze(0)
    if illumination_map is not None and illumination_map.dim() == 3:
        illumination_map = illumination_map.unsqueeze(0)
    if snr_map is not None and snr_map.dim() == 3:
        snr_map = snr_map.unsqueeze(0)
    
    # 准备可视化图像列表
    vis_tensors = [original_image, enhanced_image]
    
    # 添加照明图和SNR图（如果提供）
    if illumination_map is not None:
        # 将照明图复制为3通道便于可视化
        ill_vis = illumination_map.repeat(1, 3, 1, 1)
        vis_tensors.append(ill_vis)
    
    if snr_map is not None:
        # 将SNR图复制为3通道便于可视化
        snr_vis = snr_map.repeat(1, 3, 1, 1)
        vis_tensors.append(snr_vis)
    
    # 创建网格图像
    with torch.no_grad():
        # 将像素值从[0,1]缩放到[0,255]
        vis_tensors = [t.clamp(0, 1) * 255 for t in vis_tensors]
        # 使用make_grid创建网格
        grid = make_grid(torch.cat(vis_tensors, dim=0), nrow=len(vis_tensors))
        # 转换为numpy数组并从CHW转换为HWC
        grid_np = grid.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
    
    # 保存结果（如果提供路径）
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR))
    
    return grid_np 