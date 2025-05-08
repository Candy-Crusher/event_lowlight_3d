import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint  # 导入 checkpoint 功能
# from einops import rearrange
import matplotlib.pyplot as plt
import os
import numpy as np

# 通道注意力模块，类似于 EvLight 中的 eca_layer
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

# 基本卷积块，增加了归一化和激活函数
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True, norm=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm)
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=False) if activation else None
        self.norm = nn.InstanceNorm2d(out_channels) if norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

# 带有注意力机制的残差块，类似于 EvLight 中的 ECAResidualBlock
class AttentionResidualBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1, norm=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1, norm=True)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        # 应用通道注意力
        out = out * self.ca(out)
        # 应用空间注意力
        out = out * self.sa(out)
        
        # 避免使用 in-place 操作
        out = out + residual
        return out

# SNR 增强模块，类似于 EvLight 中的 SNR_enhance
class SNREnhanceModule(nn.Module):
    def __init__(self, channels, snr_threshold=0.5, depth=1):
        super(SNREnhanceModule, self).__init__()
        self.channels = channels
        self.depth = depth
        self.threshold = snr_threshold
        
        # 减少特征提取器深度和复杂度
        self.img_extractors = nn.ModuleList([ConvLayer(channels, channels) for _ in range(depth)])
        self.ev_extractors = nn.ModuleList([ConvLayer(channels, channels) for _ in range(depth)])
        
        # 简化特征融合层
        self.fusion = nn.Sequential(
            ConvLayer(channels*3, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, img_feat, event_feat, snr_map, att_feat):
        """
        img_feat: 图像特征 [B, C, H, W]
        event_feat: 事件特征 [B, C, H, W]
        snr_map: SNR 图 [B, 1, H, W]
        att_feat: 带有注意力的特征 [B, C, H, W]
        """
        # 根据阈值处理 SNR 图
        high_snr = snr_map.clone()
        high_snr[high_snr <= self.threshold] = 0.3
        high_snr[high_snr > self.threshold] = 0.7
        low_snr = 1 - high_snr
        
        # 扩展 SNR 图到特征通道数
        high_snr_expanded = high_snr.repeat(1, self.channels, 1, 1)
        low_snr_expanded = low_snr.repeat(1, self.channels, 1, 1)
        
        # 简化提取过程
        for i in range(self.depth):
            img_feat = self.img_extractors[i](img_feat)
            event_feat = self.ev_extractors[i](event_feat)
        
        # 选择高 SNR 区域的图像特征
        high_snr_feat = torch.mul(img_feat, high_snr_expanded)
        
        # 选择低 SNR 区域的事件特征
        low_snr_feat = torch.mul(event_feat, low_snr_expanded)
        
        # 特征融合
        fused_feat = self.fusion(torch.cat([high_snr_feat, low_snr_feat, att_feat], dim=1))
        
        return fused_feat

# 自适应特征融合模块
class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveFeatureFusion, self).__init__()
        
        # 空间注意力模块，用于学习空间权重
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 通道注意力模块，用于学习通道权重
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        
        # SNR引导的特征融合
        self.snr_fusion = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, img_feat, event_feat, snr_map=None):
        # 确保特征图尺寸一致
        if img_feat.shape[2:] != event_feat.shape[2:]:
            event_feat = F.interpolate(event_feat, size=img_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # 计算空间注意力权重
        avg_pool = torch.mean(torch.cat([img_feat, event_feat], dim=1), dim=1, keepdim=True)
        max_pool, _ = torch.max(torch.cat([img_feat, event_feat], dim=1), dim=1, keepdim=True)
        spatial_weights = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        
        # 计算通道注意力权重
        channel_weights = self.channel_attention(img_feat + event_feat)
        
        # 如果有SNR map，进行SNR引导的融合
        if snr_map is not None:
            # 确保SNR map尺寸正确
            if snr_map.shape[2:] != img_feat.shape[2:]:
                snr_map = F.interpolate(snr_map, size=img_feat.shape[2:], mode='bilinear', align_corners=False)
            
            # 使用SNR map生成融合权重
            snr_weights = self.snr_fusion(snr_map)
            
            # 结合SNR权重进行融合
            fused_feat = img_feat * spatial_weights * channel_weights * snr_weights + \
                        event_feat * spatial_weights * channel_weights * (1 - snr_weights)
        else:
            # 如果没有SNR map，使用原始融合方式
            fused_feat = img_feat * spatial_weights * channel_weights + \
                        event_feat * (1 - spatial_weights) * channel_weights
        
        # 最终融合
        out = self.fusion_conv(fused_feat)
        
        return out

# 改进的图像编码器
class ImageEncoder(nn.Module):
    def __init__(self, input_channels=3, base_channels=16):
        """轻量级图像编码器
        
        Args:
            input_channels: 输入通道数
            base_channels: 基础通道数
        """
        super(ImageEncoder, self).__init__()
        self.conv1 = ConvLayer(input_channels, base_channels, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = ConvLayer(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1)
        
        self.conv3 = ConvLayer(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1)
        
        self.conv4 = ConvLayer(base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        c1 = self.conv1(x)
        
        c2 = self.conv2(c1)
        
        c3 = self.conv3(c2)
        
        c4 = self.conv4(c3)
        
        return c1, c2, c3, c4

# 改进的事件编码器
class EventEncoder(nn.Module):
    def __init__(self, input_channels=5, base_channels=16):
        """轻量级事件编码器
        
        Args:
            input_channels: 输入通道数
            base_channels: 基础通道数
        """
        super(EventEncoder, self).__init__()
        self.conv1 = ConvLayer(input_channels, base_channels, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = ConvLayer(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1)
        
        self.conv3 = ConvLayer(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1)
        
        self.conv4 = ConvLayer(base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1)
        
        # 上采样路径
        self.upconv3 = ConvLayer(base_channels*8, base_channels*4, kernel_size=3, stride=1, padding=1)
        
        self.upconv2 = ConvLayer(base_channels*4, base_channels*2, kernel_size=3, stride=1, padding=1)
        
        self.upconv1 = ConvLayer(base_channels*2, base_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 下采样路径
        c1 = self.conv1(x)
        
        c2 = self.conv2(c1)
        
        c3 = self.conv3(c2)
        
        c4 = self.conv4(c3)
        
        # 上采样路径 - 避免 in-place 操作
        up3 = F.interpolate(c4, scale_factor=2, mode='bilinear', align_corners=False)
        up3 = self.upconv3(up3)
        u3 = up3 + c3  # 非 in-place 操作
        
        up2 = F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False)
        up2 = self.upconv2(up2)
        u2 = up2 + c2  # 非 in-place 操作
        
        up1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        up1 = self.upconv1(up1)
        u1 = up1 + c1  # 非 in-place 操作
        
        return u1, u2, u3, c4

# 解码器，结合了 SNR 增强模块
class Decoder(nn.Module):
    def __init__(self, base_channels=16, snr_thresholds=[0.5, 0.5, 0.5]):
        super(Decoder, self).__init__()
        
        # 减少特征融合模块复杂度
        self.fusion4 = AdaptiveFeatureFusion(base_channels*8, base_channels*8)
        self.fusion3 = AdaptiveFeatureFusion(base_channels*4, base_channels*4)
        self.fusion2 = AdaptiveFeatureFusion(base_channels*2, base_channels*2)
        self.fusion1 = AdaptiveFeatureFusion(base_channels, base_channels)
        
        # 减少SNR增强模块深度
        self.snr_enhance4 = SNREnhanceModule(base_channels*8, snr_thresholds[2], depth=1)
        self.snr_enhance3 = SNREnhanceModule(base_channels*4, snr_thresholds[1], depth=1)
        self.snr_enhance2 = SNREnhanceModule(base_channels*2, snr_thresholds[0], depth=1)
        
        # 上采样模块
        self.upconv3 = ConvLayer(base_channels*8, base_channels*4, kernel_size=3, stride=1, padding=1)
        self.upconv2 = ConvLayer(base_channels*4, base_channels*2, kernel_size=3, stride=1, padding=1)
        self.upconv1 = ConvLayer(base_channels*2, base_channels, kernel_size=3, stride=1, padding=1)
        
        # 输出层
        self.output_conv = nn.Sequential(
            ConvLayer(base_channels, 3, kernel_size=3, stride=1, padding=1, activation=False)
        )

    def generate_snr_map(self, image, blur_image, factor=10.0):
        """生成 SNR 图，简化计算"""
        # 转换为灰度图，使用固定权重避免复杂计算
        gray = image.mean(dim=1, keepdim=True)
        gray_blur = blur_image.mean(dim=1, keepdim=True)
        
        # 计算噪声
        noise = torch.abs(gray - gray_blur)
        
        # 计算 SNR，添加小值避免除零
        snr = torch.div(gray_blur, noise + 0.0001)
        
        # 归一化到 [0, 1]
        batch_size = snr.shape[0]
        snr_max, _ = torch.max(snr.view(batch_size, -1), dim=1, keepdim=True)
        snr_max = snr_max.view(batch_size, 1, 1, 1)
        snr = snr * factor / (snr_max + 0.0001)
        snr = torch.clamp(snr, min=0, max=1.0)
        
        return snr

    def forward(self, img_features, event_features, low_light_img):
        img_f1, img_f2, img_f3, img_f4 = img_features
        ev_f1, ev_f2, ev_f3, ev_f4 = event_features
        
        # 生成模糊图像用于 SNR 计算
        with torch.no_grad():
            blur_kernel = torch.ones(1, 1, 5, 5).to(low_light_img.device) / 25
            low_light_blur = F.conv2d(
                F.pad(low_light_img, (2, 2, 2, 2), mode='reflect'),
                blur_kernel.repeat(3, 1, 1, 1),
                groups=3
            )
        
        # 生成 SNR 图
        snr_map = self.generate_snr_map(low_light_img, low_light_blur)
        
        # 特征融合
        f4 = self.fusion4(img_f4, ev_f4, F.interpolate(snr_map, size=img_f4.shape[2:]))
        f4 = self.snr_enhance4(img_f4, ev_f4, F.interpolate(snr_map, size=f4.shape[2:]), f4)
        
        # 上采样并融合 - 避免原地操作
        up3 = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)
        up3 = self.upconv3(up3)
        f3 = self.fusion3(img_f3, ev_f3, F.interpolate(snr_map, size=img_f3.shape[2:]))
        u3 = up3 + f3  # 非原地操作
        u3 = self.snr_enhance3(img_f3, ev_f3, F.interpolate(snr_map, size=u3.shape[2:]), u3)
        
        up2 = F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False)
        up2 = self.upconv2(up2)
        f2 = self.fusion2(img_f2, ev_f2, F.interpolate(snr_map, size=img_f2.shape[2:]))
        u2 = up2 + f2  # 非原地操作
        u2 = self.snr_enhance2(img_f2, ev_f2, F.interpolate(snr_map, size=u2.shape[2:]), u2)
        
        up1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        up1 = self.upconv1(up1)
        f1 = self.fusion1(img_f1, ev_f1, F.interpolate(snr_map, size=img_f1.shape[2:]))
        u1 = up1 + f1  # 非原地操作
        
        # 生成增强后的图像残差
        enhanced_residual = self.output_conv(u1)
        
        return enhanced_residual, snr_map

class EasyIlluminationNet(nn.Module):
    def __init__(self, image_channels=3, event_channels=None, verbose_viz=False):  # event_channels不再使用，保留为了接口兼容
        super(EasyIlluminationNet, self).__init__()
        self.verbose_viz = verbose_viz
        
        # 照明特征提取器 - 简化版本
        self.ill_extractor = nn.Sequential(
            nn.Conv2d(
                image_channels + 1,  # 图像 + 初始照明图
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        
        # 照明映射 - 减少到单通道
        self.reduce = nn.Sequential(
            nn.Conv2d(16, 1, 1, 1, 0),
            nn.Sigmoid()  # 确保照明图范围在[0,1]，更符合论文描述
        )
        
    def _generate_illumination_map(self, img):
        """生成照明先验图"""
        return torch.max(img, dim=1, keepdim=True)[0]
        
    def _generate_snr_map(self, enhanced_img):
        """生成SNR图 - 使用更简单有效的方式"""
        # 转换为灰度图
        r, g, b = enhanced_img[:, 0:1], enhanced_img[:, 1:2], enhanced_img[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b  # 标准RGB到灰度转换
        
        # 使用均值滤波对灰度图进行去噪
        kernel_size = 5
        padding = kernel_size // 2
        denoised_gray = F.avg_pool2d(
            F.pad(gray, (padding, padding, padding, padding), mode='reflect'),
            kernel_size=kernel_size, 
            stride=1
        )
        
        # 计算噪声
        noise = torch.abs(gray - denoised_gray)
        epsilon = 1e-6  # 避免除零
        
        # 计算SNR
        snr_map = torch.div(denoised_gray, noise + epsilon)
        
        # 使用自适应归一化
        batch_size = snr_map.shape[0]
        snr_flat = snr_map.view(batch_size, -1)
        
        # 使用百分位数进行归一化，避免异常值的影响
        q1 = torch.quantile(snr_flat, 0.25, dim=1).view(batch_size, 1, 1, 1)
        q3 = torch.quantile(snr_flat, 0.75, dim=1).view(batch_size, 1, 1, 1)
        iqr = q3 - q1
        
        # 使用IQR进行归一化，对异常值更鲁棒
        snr_map = (snr_map - q1) / (iqr + epsilon)
        
        # 使用sigmoid函数进行平滑映射
        snr_map = torch.sigmoid(snr_map)
        
        # 确保没有NaN值
        snr_map = torch.nan_to_num(snr_map, nan=0.5, posinf=1.0, neginf=0.0)
        
        return snr_map
    
    def forward(self, low_light_image, event_voxel=None):  # event_voxel参数保留但不使用
        # 1. 生成照明先验图
        illumination_prior = self._generate_illumination_map(low_light_image)
        
        if self.verbose_viz:
            # 可视化照明先验图
            save_dir = "visualization/lightup_net"
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存原始低光图像
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.imshow(low_light_image[0].permute(1,2,0).cpu().detach().numpy())
            plt.title('Low Light Image')
            plt.subplot(122)
            plt.imshow(illumination_prior[0].permute(1,2,0).cpu().detach().numpy())
            plt.title('Illumination Prior')
            plt.savefig(f"{save_dir}/step1_illumination_prior.png")
            plt.close()
        
        # 2. 提取照明特征并预测照明图
        pred_illu_feature = self.ill_extractor(torch.cat((low_light_image, illumination_prior), dim=1))
        illumination_map = self.reduce(pred_illu_feature)
        
        if self.verbose_viz:
            # 可视化照明特征和照明图
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(pred_illu_feature[0,0].cpu().detach().numpy())
            plt.title('Illumination Feature (Channel 0)')
            plt.subplot(132)
            plt.imshow(illumination_map[0,0].cpu().detach().numpy())
            plt.title('Illumination Map')
            plt.subplot(133)
            plt.imshow(illumination_map[0,0].cpu().detach().numpy(), cmap='jet')
            plt.title('Illumination Map (Jet)')
            plt.savefig(f"{save_dir}/step2_illumination_map.png")
            plt.close()
        
        # 3. 初步增强图像
        enhanced_image = low_light_image * illumination_map
        
        if self.verbose_viz:
            # 可视化增强后的图像
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(low_light_image[0].permute(1,2,0).cpu().detach().numpy())
            plt.title('Low Light Image')
            plt.subplot(132)
            plt.imshow(enhanced_image[0].permute(1,2,0).cpu().detach().numpy())
            plt.title('Enhanced Image')
            plt.subplot(133)
            plt.imshow((enhanced_image[0] - low_light_image[0]).permute(1,2,0).cpu().detach().numpy())
            plt.title('Difference')
            plt.savefig(f"{save_dir}/step3_enhanced_image.png")
            plt.close()
        
        # 4. 生成SNR图
        snr_map = self._generate_snr_map(enhanced_image)
        
        if self.verbose_viz:
            # 可视化SNR图
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(snr_map[0,0].cpu().detach().numpy())
            plt.title('SNR Map')
            plt.subplot(132)
            plt.imshow(snr_map[0,0].cpu().detach().numpy(), cmap='jet')
            plt.title('SNR Map (Jet)')
            plt.subplot(133)
            plt.hist(snr_map[0,0].cpu().detach().numpy().flatten(), bins=50)
            plt.title('SNR Distribution')
            plt.savefig(f"{save_dir}/step4_snr_map.png")
            plt.close()
            
            # 保存所有中间结果到numpy文件
            np.save(f"{save_dir}/low_light_image.npy", low_light_image.cpu().detach().numpy())
            np.save(f"{save_dir}/illumination_prior.npy", illumination_prior.cpu().detach().numpy())
            np.save(f"{save_dir}/pred_illu_feature.npy", pred_illu_feature.cpu().detach().numpy())
            np.save(f"{save_dir}/illumination_map.npy", illumination_map.cpu().detach().numpy())
            np.save(f"{save_dir}/enhanced_image.npy", enhanced_image.cpu().detach().numpy())
            np.save(f"{save_dir}/snr_map.npy", snr_map.cpu().detach().numpy())
        
        return enhanced_image, snr_map

# 事件引导的低光图像增强网络
class ComplexEvLightEnhancer(nn.Module):
    def __init__(self, image_channels=3, event_channels=5, base_channels=16, snr_thresholds=[0.5, 0.5, 0.5], use_checkpoint=True):
        """复杂的事件引导低光图像增强网络
        
        Args:
            image_channels: 图像输入通道数，默认为3（RGB）
            event_channels: 事件输入通道数，默认为5
            base_channels: 基础通道数，默认为16（减少内存占用）
            snr_thresholds: SNR阈值列表
            use_checkpoint: 是否使用梯度检查点以减少内存占用
        """
        super(ComplexEvLightEnhancer, self).__init__()
        
        # 使用更轻量级的编码器和解码器
        self.image_encoder = ImageEncoder(image_channels, base_channels)
        self.event_encoder = EventEncoder(event_channels, base_channels)
        
        # 添加自适应特征融合模块
        self.fusion1 = AdaptiveFeatureFusion(base_channels, base_channels)
        self.fusion2 = AdaptiveFeatureFusion(base_channels*2, base_channels*2)
        self.fusion3 = AdaptiveFeatureFusion(base_channels*4, base_channels*4)
        self.fusion4 = AdaptiveFeatureFusion(base_channels*8, base_channels*8)
        
        self.decoder = Decoder(base_channels, snr_thresholds)
        self.use_checkpoint = use_checkpoint
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, low_light_image, event_voxel):
        # 归一化事件数据
        if event_voxel.max() > 1.0:
            event_voxel = event_voxel / 255.0 if event_voxel.max() > 255.0 else event_voxel / event_voxel.max()
        
        # 使用 checkpoint 减少内存占用
        if self.use_checkpoint and self.training:
            # 编码图像和事件数据
            img_features = checkpoint.checkpoint(self.image_encoder, low_light_image)
            event_features = checkpoint.checkpoint(self.event_encoder, event_voxel)
            
            # 自适应特征融合
            fused_features = [
                self.fusion1(img_features[0], event_features[0]),
                self.fusion2(img_features[1], event_features[1]),
                self.fusion3(img_features[2], event_features[2]),
                self.fusion4(img_features[3], event_features[3])
            ]
            
            # 解码融合特征生成增强图像残差
            enhanced_residual, snr_map = checkpoint.checkpoint(self.decoder, fused_features, fused_features, low_light_image)
        else:
            # 不使用 checkpoint
            img_features = self.image_encoder(low_light_image)
            event_features = self.event_encoder(event_voxel)
            
            # 自适应特征融合
            fused_features = [
                self.fusion1(img_features[0], event_features[0]),
                self.fusion2(img_features[1], event_features[1]),
                self.fusion3(img_features[2], event_features[2]),
                self.fusion4(img_features[3], event_features[3])
            ]
            
            enhanced_residual, snr_map = self.decoder(fused_features, fused_features, low_light_image)
        
        # 残差连接，增强图像是原始图像加上残差
        enhanced_image = torch.clamp(low_light_image + enhanced_residual, 0, 1)
        
        return enhanced_image, snr_map

# 统一接口，根据模式选择不同的增强方法
class EvLightEnhancer(nn.Module):
    def __init__(self, mode='complex', image_channels=3, event_channels=5, base_channels=16, snr_thresholds=[0.5, 0.5, 0.5], use_checkpoint=True):
        """
        低光图像增强网络统一接口

        Args:
            mode: 增强模式，'none'不进行增强，'easy'使用改进后的轻量级增强（不使用event数据），'complex'使用复杂的增强网络
            image_channels: 图像输入通道数
            event_channels: 事件输入通道数
            base_channels: 基础通道数
            snr_thresholds: SNR阈值列表
            use_checkpoint: 是否使用梯度检查点
        """
        super(EvLightEnhancer, self).__init__()
        self.mode = mode

        if mode == 'easy':
            # 使用改进后的easy增强网络（不依赖event数据）
            self.illumination_net = EasyIlluminationNet(image_channels)
        elif mode == 'complex':
            self.enhancer = ComplexEvLightEnhancer(image_channels, event_channels, base_channels, snr_thresholds, use_checkpoint)

    def forward(self, low_light_image, event_voxel=None):
        """
        前向传播

        Args:
            low_light_image: 低光图像 [B, C, H, W]
            event_voxel: 事件体素数据 [B, C, H, W]，对于'none'和'easy'模式会被忽略

        Returns:
            enhanced_image: 增强后的图像 [B, C, H, W]
            snr_map: 信噪比图 [B, 1, H, W]
        """
        if self.mode == 'none':
            batch_size, _, height, width = low_light_image.shape
            return low_light_image, torch.zeros((batch_size, 1, height, width), device=low_light_image.device)

        if self.mode == 'easy':
            # 使用改进后的easy增强网络，不使用event数据
            enhanced_image, snr_map = self.illumination_net(low_light_image)
            return enhanced_image, snr_map

        # complex模式
        return self.enhancer(low_light_image, event_voxel)

# 测试代码
if __name__ == '__main__':
    batch_size, height, width = 2, 256, 256
    low_light_image = torch.rand(batch_size, 3, height, width)
    event_voxel = torch.rand(batch_size, 5, height, width)

    modes = ['none', 'easy', 'complex']

    for mode in modes:
        print(f"\n测试 {mode} 模式:")
        model = EvLightEnhancer(mode=mode)
        if mode == 'complex':
            print("使用event数据进行增强...")
            enhanced_image, snr_map = model(low_light_image, event_voxel)
        else:
            if mode == 'easy':
                print("不使用event数据，仅基于改进后的easy增强...")
            else:
                print("不进行任何增强...")
            enhanced_image, snr_map = model(low_light_image, None)

        print(f"输入图像形状: {low_light_image.shape}")
        if mode == 'complex':
            print(f"事件数据形状: {event_voxel.shape}")
        print(f"增强后图像形状: {enhanced_image.shape}")
        print(f"SNR图形状: {snr_map.shape}")