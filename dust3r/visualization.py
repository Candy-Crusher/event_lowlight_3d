import numpy as np
import matplotlib.pyplot as plt
import math

def visualize_feature(image_feature, save_path="visualization.png", true_shape=None, dim=100):
    """
    可视化单个 image feature 的张量，显示指定维度并 reshape 为 [H, W]。

    参数:
    - image_feature: 输入特征张量, 形状 [batch, seq_len, embed_dim]
    - save_path: 保存图像的路径, 默认 "visualization.png"
    - true_shape: 真实形状, 形状 [batch, 2]，例如 [[height, width]]
    - dim: 要可视化的维度索引, 默认 100
    """
    # 选择第一个 batch 的数据
    batch_idx = 0
    if true_shape is None:
        raise ValueError("true_shape must be provided for reshaping")
    scale = math.sqrt(true_shape[batch_idx][0].item()*true_shape[batch_idx][1].item()/image_feature.shape[1])
    scale = int(scale)
    print(f"scale: {scale}")
    resize_shape = (true_shape[batch_idx][0].item() // scale, true_shape[batch_idx][1].item() // scale)
    print(f"resize_shape: {resize_shape}")
    H, W = resize_shape
    seq_len = H * W

    # 提取第一个 batch 的数据
    image_feature = image_feature[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]

    # 验证 seq_len 匹配
    if image_feature.shape[0] != seq_len:
        raise ValueError(f"seq_len ({image_feature.shape[0]}) does not match expected H*W ({seq_len})")

    # 确保 dim 在 embed_dim 范围内
    embed_dim = image_feature.shape[1]
    if dim >= embed_dim:
        raise ValueError(f"Dimension index {dim} exceeds embed_dim {embed_dim}")

    # Reshape 为 [H, W]，取指定维度
    feature_vis = image_feature[:, dim].reshape(H, W)  # [H, W]

    # 归一化到 [0, 1] 以便显示（注释掉归一化，按原代码保持）
    def normalize(tensor):
        # tensor = tensor - tensor.min()
        # tensor = tensor / (tensor.max() + 1e-8)
        return tensor

    feature_vis = normalize(feature_vis)

    # 创建单个图像
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # 可视化 image_feature（热图）
    im = ax.imshow(feature_vis, cmap='viridis')
    ax.set_title(f'Image Feature (dim {dim})')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.colorbar(im, ax=ax)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")

def visualize_tensors_snr(x, event_feat, snr_map, output_x, save_path="visualization.png", true_shape=None):
    """
    可视化 x, event_feat, snr_map 和 output_x 的张量，只显示第一个维度，reshape 为 [H, W]。

    参数:
    - x: 初始输入张量, 形状 [batch, seq_len, embed_dim]
    - event_feat: 事件特征张量, 形状与 x 一致
    - snr_map: 信噪比图, 形状 [batch, seq_len, 1]
    - output_x: 最终输出张量, 形状与 x 一致
    - save_path: 保存图像的路径, 默认 "visualization.png"
    - true_shape: 真实形状, 形状 [batch, 2]，例如 [[height, width]]
    """
    # 选择第一个 batch 的数据
    batch_idx = 0
    if true_shape is None:
        raise ValueError("true_shape must be provided for reshaping")
    resize_shape = (true_shape[batch_idx][0].item() // 16, true_shape[batch_idx][1].item() // 16)
    H, W = resize_shape
    seq_len = H * W

    # 提取第一个 batch 的数据
    x = x[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]
    event_feat = event_feat[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]
    output_x = output_x[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]
    if snr_map is not None:
        snr_map = snr_map[batch_idx, :, 0].detach().cpu().numpy()  # [seq_len]

    # 验证 seq_len 匹配
    if x.shape[0] != seq_len:
        raise ValueError(f"seq_len ({x.shape[0]}) does not match expected H*W ({seq_len})")

    # Reshape 为 [H, W]，取第一个维度
    x_vis = x[:, 100].reshape(H, W)  # [H, W]
    event_feat_vis = event_feat[:, 100].reshape(H, W)  # [H, W]
    output_x_vis = output_x[:, 100].reshape(H, W)  # [H, W]
    if snr_map is not None:
        snr_map = snr_map.reshape(H, W)  # [H, W]

    # 归一化到 [0, 1] 以便显示
    def normalize(tensor):
        # tensor = tensor - tensor.min()
        # tensor = tensor / (tensor.max() + 1e-8)
        return tensor

    x_vis = normalize(x_vis)
    event_feat_vis = normalize(event_feat_vis)
    output_x_vis = normalize(output_x_vis)
    if snr_map is not None:
        snr_map = normalize(snr_map)

    # 创建 2x2 的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 可视化 x（热图）
    im0 = axes[0].imshow(x_vis, cmap='viridis')
    axes[0].set_title('Input x (dim 0)')
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    plt.colorbar(im0, ax=axes[0])

    # 可视化 event_feat（热图）
    im1 = axes[1].imshow(event_feat_vis, cmap='viridis')
    axes[1].set_title('Event Feature (dim 0)')
    axes[1].set_xlabel('Width')
    axes[1].set_ylabel('Height')
    plt.colorbar(im1, ax=axes[1])

    # 可视化 snr_map（热图）
    if snr_map is not None:
        im2 = axes[2].imshow(snr_map, cmap='gray')
        axes[2].set_title('SNR Map (dim 0)')
        axes[2].set_xlabel('Width')
        axes[2].set_ylabel('Height')
        plt.colorbar(im2, ax=axes[2])
    else:
        axes[2].axis('off')

    # 可视化 output_x（热图）
    im3 = axes[3].imshow(output_x_vis, cmap='viridis')
    axes[3].set_title('Output x (dim 0)')
    axes[3].set_xlabel('Width')
    axes[3].set_ylabel('Height')
    plt.colorbar(im3, ax=axes[3])

    # 手动调整间距
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")

def visualize_tensors(x, event_feat, atten, output_x, save_path="visualization.png", true_shape=None):
    """
    可视化 x, event_feat, snr_map 和 output_x 的张量，只显示第一个维度，reshape 为 [H, W]。

    参数:
    - x: 初始输入张量, 形状 [batch, seq_len, embed_dim]
    - event_feat: 事件特征张量, 形状与 x 一致
    - atten: 注意力图, 形状与 x 一致
    - output_x: 最终输出张量, 形状与 x 一致
    - save_path: 保存图像的路径, 默认 "visualization.png"
    - true_shape: 真实形状, 形状 [batch, 2]，例如 [[height, width]]
    """
    # 选择第一个 batch 的数据
    batch_idx = 0
    if true_shape is None:
        raise ValueError("true_shape must be provided for reshaping")
    resize_shape = (true_shape[batch_idx][0].item() // 16, true_shape[batch_idx][1].item() // 16)
    H, W = resize_shape
    seq_len = H * W

    # 提取第一个 batch 的数据
    x = x[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]
    event_feat = event_feat[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]
    atten = atten[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]
    output_x = output_x[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]
    print(f"x.shape: {x.shape}, event_feat.shape: {event_feat.shape}, atten.shape: {atten.shape}, output_x.shape: {output_x.shape}")

    # 验证 seq_len 匹配
    if x.shape[0] != seq_len:
        raise ValueError(f"seq_len ({x.shape[0]}) does not match expected H*W ({seq_len})")

    # Reshape 为 [H, W]，取第一个维度
    x_vis = x[:, 100].reshape(H, W)  # [H, W]
    event_feat_vis = event_feat[:, 100].reshape(H, W)  # [H, W]
    atten_vis = atten[:, 100].reshape(H, W)  # [H, W]
    output_x_vis = output_x[:, 100].reshape(H, W)  # [H, W]

    # 归一化到 [0, 1] 以便显示
    def normalize(tensor):
        # tensor = tensor - tensor.min()
        # tensor = tensor / (tensor.max() + 1e-8)
        return tensor

    x_vis = normalize(x_vis)
    event_feat_vis = normalize(event_feat_vis)
    atten_vis = normalize(atten_vis)
    output_x_vis = normalize(output_x_vis)

    # 创建 2x2 的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 可视化 x（热图）
    im0 = axes[0].imshow(x_vis, cmap='viridis')
    axes[0].set_title('Input x (dim 0)')
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    plt.colorbar(im0, ax=axes[0])

    # 可视化 event_feat（热图）
    im1 = axes[1].imshow(event_feat_vis, cmap='viridis')
    axes[1].set_title('Event Feature (dim 0)')
    axes[1].set_xlabel('Width')
    axes[1].set_ylabel('Height')
    plt.colorbar(im1, ax=axes[1])

    # 可视化 atten（热图）
    im2 = axes[2].imshow(atten_vis, cmap='viridis')
    axes[2].set_title('Attention Map (dim 0)')
    axes[2].set_xlabel('Width')
    axes[2].set_ylabel('Height')
    plt.colorbar(im2, ax=axes[2])

    # 可视化 output_x（热图）
    im3 = axes[3].imshow(output_x_vis, cmap='viridis')
    axes[3].set_title('Output x (dim 0)')
    axes[3].set_xlabel('Width')
    axes[3].set_ylabel('Height')
    plt.colorbar(im3, ax=axes[3])

    # 手动调整间距
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")

def visualize_image_snr(old_image, new_image, snr_map, save_path="image_snr_visualization.png"):
    """
    可视化 old_image, new_image 和 snr_map。

    参数:
    - old_image: 原始图像张量, 形状 [B, 3, H, W]
    - new_image: 增强后的图像张量, 形状 [B, 3, H, W]
    - snr_map: 信噪比图, 形状 [B, 1, H, W]
    - save_path: 保存图像的路径, 默认 "image_snr_visualization.png"
    """
    # 选择第一个 batch 的数据
    batch_idx = 0
    old_image = old_image[batch_idx].detach().cpu().numpy()  # [3, H, W]
    new_image = new_image[batch_idx].detach().cpu().numpy()  # [3, H, W]
    snr_map = snr_map[batch_idx, 0].detach().cpu().numpy()  # [H, W]

    # 将图像从 [C, H, W] 转换为 [H, W, C]，以便显示
    old_image = old_image.transpose(1, 2, 0)  # [H, W, 3]
    new_image = new_image.transpose(1, 2, 0)  # [H, W, 3]

    # 归一化到 [0, 1] 以便显示
    def normalize(tensor):
        tensor = tensor - tensor.min()
        tensor = tensor / (tensor.max() + 1e-8)
        return tensor

    old_image = normalize(old_image)
    new_image = normalize(new_image)
    snr_map = normalize(snr_map)

    # 创建 1x3 的子图布局
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()

    # 可视化 old_image（RGB 图像）
    axes[0].imshow(old_image)
    axes[0].set_title('Old Image')
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    axes[0].axis('off')  # 隐藏坐标轴

    # 可视化 new_image（RGB 图像）
    axes[1].imshow(new_image)
    axes[1].set_title('New Image')
    axes[1].set_xlabel('Width')
    axes[1].set_ylabel('Height')
    axes[1].axis('off')  # 隐藏坐标轴

    # 可视化 snr_map（灰度热图）
    im2 = axes[2].imshow(snr_map, cmap='gray')
    axes[2].set_title('SNR Map')
    axes[2].set_xlabel('Width')
    axes[2].set_ylabel('Height')
    plt.colorbar(im2, ax=axes[2])

    # 手动调整间距
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")
