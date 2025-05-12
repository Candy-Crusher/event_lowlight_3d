import h5py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from scipy.ndimage import gaussian_filter

class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool, use_weight: bool=True):
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.nb_channels = channels
        self.normalize = normalize
        self.use_weight = use_weight

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(pol.device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = time
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()

            if pol.min() == 0:
                value = 2*pol-1
            else:
                value = pol

            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        if self.use_weight:
                            interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())
                        else:
                            interp_weights = value

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid

def events_to_voxel_grid(voxel_grid, bin, x, y, p, t, device: str='cpu'):
    t = (t - t[0]).astype('float32')
    t = (t/t[-1])
    x = x.astype('float32')
    y = y.astype('float32')
    pol = p.astype('float32') # -1 1
    return voxel_grid.convert(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(pol),
        torch.from_numpy(t))

def read_voxel_hdf5(file_path,key='event_voxels'):
    with h5py.File(file_path, 'r') as f:
        voxel_data = f[key][:]
    # B, H, W = voxel_data.shape
    # voxel_data = voxel_data.reshape(3, 2, H, W).sum(axis=1)  # New shape: (B//2, H, W)
    return voxel_data

def resize_event_voxel(event_voxel, target_size, mode='bilinear'):
    """
    调整事件数据的空间分辨率(H, W)，保留时间维度(C)。
    
    Args:
        event_voxel (torch.Tensor): 事件数据，形状为 [C, H, W]
        target_size (tuple): 目标分辨率 (target_h, target_w)
        mode (str): 插值模式，'bilinear' 或 'nearest'
    
    Returns:
        torch.Tensor: 调整后的数据，形状为 [C, target_h, target_w]
    """
    C, H, W = event_voxel.shape
    target_h, target_w = target_size
    
    # 使用 interpolate 调整空间维度，保持 C 维度不变
    event_voxel = F.interpolate(event_voxel.unsqueeze(0), size=(target_h, target_w), mode=mode, align_corners=False if mode == 'bilinear' else None)
    return event_voxel.squeeze(0)  # 移除临时添加的 batch 维度

def crop_event(event_voxel, size, square_ok=False, crop=True, mode='bilinear'):
    """
    裁剪或调整事件数据，与 crop_img 对齐。
    
    Args:
        event_voxel (torch.Tensor): 事件数据，形状为 [C, H, W]
        size (int): 目标大小(与 crop_img 的 size 参数一致，例如 224 或 512)
        square_ok (bool): 是否允许输出为正方形
        crop (bool): 是否裁剪(True)或调整大小(False)
        mode (str): 插值模式，'bilinear' 或 'nearest'
    
    Returns:
        torch.Tensor: 裁剪或调整后的数据
    """
    C, H1, W1 = event_voxel.shape
    
    # Step 1: 调整大小，与 crop_img 保持一致
    if size == 224:
        # 短边调整到 224
        scale = size / min(H1, W1)
        target_h = round(H1 * scale)
        target_w = round(W1 * scale)
        event_voxel = resize_event_voxel(event_voxel, (target_h, target_w), mode=mode)
    else:
        # 长边调整到 512
        scale = size / max(H1, W1)
        target_h = round(H1 * scale)
        target_w = round(W1 * scale)
        event_voxel = resize_event_voxel(event_voxel, (target_h, target_w), mode=mode)

    # Step 2: 裁剪或调整到目标区域
    C, H, W = event_voxel.shape
    cx, cy = W // 2, H // 2

    if size == 224:
        # 裁剪为 224x224
        half = min(cx, cy)
        left = cx - half
        top = cy - half
        right = cx + half
        bottom = cy + half
        event_voxel = event_voxel[:, top:bottom, left:right]
    else:
        # 裁剪为 512x384 或调整大小
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not square_ok and W == H:
            halfh = 3 * halfw // 4  # 调整为 3:4 比例
        
        if crop:
            left = cx - halfw
            top = cy - halfh
            right = cx + halfw
            bottom = cy + halfh
            event_voxel = event_voxel[:, top:bottom, left:right]
        else:
            # 调整大小而不是裁剪
            target_size = (2 * halfw, 2 * halfh)
            event_voxel = resize_event_voxel(event_voxel, target_size, mode=mode)

    return event_voxel

def detect_harris_corners_and_gradients(image, patch_size=5, k=0.04, threshold=0.01):
    """
    检测Harris角点并计算梯度
    """
    # 1. 计算图像梯度
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # 2. 计算梯度乘积
    Ixx = Ix * Ix
    Ixy = Iy * Ix
    Iyy = Iy * Iy
    
    # 3. 使用高斯核进行平滑
    Ixx = gaussian_filter(Ixx, sigma=1)
    Ixy = gaussian_filter(Ixy, sigma=1)
    Iyy = gaussian_filter(Iyy, sigma=1)
    
    # 4. 计算Harris响应
    det = (Ixx * Iyy) - (Ixy * Ixy)
    trace = Ixx + Iyy
    harris_response = det - k * (trace ** 2)
    
    # 5. 非极大值抑制
    corners = []
    patches = []
    gradients = []
    
    # 获取局部最大值
    local_max = cv2.dilate(harris_response, None)
    corner_mask = (harris_response == local_max) & (harris_response > threshold * harris_response.max())
    corner_points = np.where(corner_mask)
    
    # 6. 提取角点周围的patch和梯度
    half_patch = patch_size // 2
    for i in range(len(corner_points[0])):
        y, x = corner_points[0][i], corner_points[1][i]
        
        # 确保patch不超出图像边界
        if (y >= half_patch and y < image.shape[0] - half_patch and 
            x >= half_patch and x < image.shape[1] - half_patch):
            
            # 提取patch
            patch = image[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
            
            # 计算该点的梯度
            grad_x = Ix[y, x].mean()  # 使用mean()来获取标量值
            grad_y = Iy[y, x].mean()  # 使用mean()来获取标量值
            
            corners.append([x, y])
            patches.append(patch)
            gradients.append([grad_x, grad_y])
    
    return np.array(corners), np.array(patches), np.array(gradients)

def visualize_corners_and_gradients(image, corners, gradients):
    """
    可视化角点和梯度
    """
    plt.figure(figsize=(12, 4))
    
    # 显示原始图像
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    # 显示角点
    plt.subplot(132)
    plt.imshow(image, cmap='gray')
    plt.scatter(corners[:, 0], corners[:, 1], c='r', s=20)
    plt.title('Detected Corners')
    
    # 显示梯度
    plt.subplot(133)
    plt.imshow(image, cmap='gray')
    
    # 确保梯度和角点数量匹配
    if len(corners) > 0 and len(gradients) > 0:
        # 归一化梯度以便更好地可视化
        norm = np.sqrt(gradients[:, 0]**2 + gradients[:, 1]**2)
        norm[norm == 0] = 1  # 避免除以零
        normalized_gradients = gradients / norm[:, np.newaxis]
        
        # 使用quiver来显示梯度
        plt.quiver(corners[:, 0], corners[:, 1], 
                  normalized_gradients[:, 0], normalized_gradients[:, 1],
                  color='r', scale=50)
    plt.title('Gradients at Corners')
    
    plt.tight_layout()
    plt.show()

def compute_gradient_flow_dot_product(corners, gradients, ego_flow):
    """
    计算每个角点处的梯度和光流的点积，保持梯度信息用于loss计算
    
    Args:
        corners: 角点坐标 [N, 2]
        gradients: 梯度向量 [N, 2]
        ego_flow: 光流场 [2, H, W] (CUDA tensor)
    
    Returns:
        dot_products: 点积结果 [N] (保持梯度信息)
    """
    dot_products = []
    
    for i in range(len(corners)):
        x, y = corners[i].astype(int)
        gx, gy = gradients[i]
        
        # 获取该点的光流
        flow_x = ego_flow[0, y, x]  # x方向光流
        flow_y = ego_flow[1, y, x]  # y方向光流
        
        # 计算点积 (保持梯度信息)
        dot_product = gx * flow_x + gy * flow_y
        dot_products.append(dot_product)
    
    return torch.stack(dot_products)  # 返回tensor而不是numpy数组

def visualize_gradient_flow_dot_product(image, corners, gradients, ego_flow, patch_flow, dot_products, event_repr_at_corners=None, save_dir='visualization'):
    """
    可视化梯度和光流的点积结果并保存到本地
    
    Args:
        image: 输入图像
        corners: 角点坐标 [N, 2]
        gradients: 梯度向量 [N, 2]
        ego_flow: 光流场 [2, H, W]
        patch_flow: 角点处的光流 [2, N]
        dot_products: 点积结果 [N]
        event_repr_at_corners: 角点处的事件表示 [N]，可选
        save_dir: 保存目录
    """
    # 为了可视化，我们需要分离梯度并转换为numpy
    image = image.detach().cpu().numpy() if torch.is_tensor(image) else image
    corners = corners.detach().cpu().numpy() if torch.is_tensor(corners) else corners
    gradients = gradients.detach().cpu().numpy() if torch.is_tensor(gradients) else gradients
    ego_flow = ego_flow.detach().cpu().numpy() if torch.is_tensor(ego_flow) else ego_flow
    patch_flow = patch_flow.detach().cpu().numpy() if torch.is_tensor(patch_flow) else patch_flow
    dot_products_np = dot_products.detach().cpu().numpy() if torch.is_tensor(dot_products) else dot_products
    if event_repr_at_corners is not None:
        event_repr_np = event_repr_at_corners.detach().cpu().numpy() if torch.is_tensor(event_repr_at_corners) else event_repr_at_corners
    
    # 确保patch_flow的维度正确 [2, N] -> [N, 2]
    if patch_flow.shape[0] == 2:
        patch_flow = patch_flow.T
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建图形和GridSpec
    if event_repr_at_corners is not None:
        fig = plt.figure(figsize=(30, 5))
        gs = fig.add_gridspec(1, 5)
    else:
        fig = plt.figure(figsize=(24, 5))
        gs = fig.add_gridspec(1, 4)
    
    # 1. 显示原始图像和角点
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.scatter(corners[:, 0], corners[:, 1], c='r', s=20)
    ax1.set_title('Corner Positions')
    
    # 2. 显示梯度和光流
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(image, cmap='gray')
    # 归一化梯度
    norm = np.sqrt(gradients[:, 0]**2 + gradients[:, 1]**2)
    norm[norm == 0] = 1
    normalized_gradients = gradients / norm[:, np.newaxis]
    
    # 绘制梯度
    ax2.quiver(corners[:, 0], corners[:, 1], 
              normalized_gradients[:, 0], normalized_gradients[:, 1],
              color='r', scale=50, label='Gradient')
    
    # 绘制光流
    for i in range(len(corners)):
        x, y = corners[i].astype(int)
        flow_x = ego_flow[0, y, x]  # x方向光流
        flow_y = ego_flow[1, y, x]  # y方向光流
        ax2.arrow(x, y, flow_x, flow_y, 
                 color='b', alpha=0.5, 
                 head_width=2, head_length=2,
                 label='Flow' if i==0 else "")
    
    ax2.set_title('Gradients and Flow')
    ax2.legend()
    
    # 3. 显示点积结果
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(image, cmap='gray')
    scatter = ax3.scatter(corners[:, 0], corners[:, 1], 
                         c=dot_products_np, cmap='coolwarm', s=50)
    ax3.set_title('Gradient-Flow Dot Product')
    
    # 添加colorbar，并调整位置
    cbar = plt.colorbar(scatter, ax=ax3, pad=0.1)
    cbar.set_label('Dot Product')
    
    # 4. 显示patch flow
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(image, cmap='gray')
    # 归一化patch flow
    norm = np.sqrt(patch_flow[:, 0]**2 + patch_flow[:, 1]**2)
    norm[norm == 0] = 1
    normalized_patch_flow = patch_flow / norm[:, np.newaxis]
    
    # 绘制patch flow
    ax4.quiver(corners[:, 0], corners[:, 1], 
              normalized_patch_flow[:, 0], normalized_patch_flow[:, 1],
              color='g', scale=50)
    ax4.set_title('Patch Flow')
    
    # 5. 如果提供了事件表示，显示事件表示
    if event_repr_at_corners is not None:
        ax5 = fig.add_subplot(gs[0, 4])
        ax5.imshow(image, cmap='gray')
        scatter = ax5.scatter(corners[:, 0], corners[:, 1], 
                            c=event_repr_np, cmap='coolwarm', s=50)
        ax5.set_title('Event Representation')
        
        # 添加colorbar，并调整位置
        cbar = plt.colorbar(scatter, ax=ax5, pad=0.1)
        cbar.set_label('Event Value')
    
    # 调整布局
    plt.subplots_adjust(wspace=0.3, right=0.95)
    
    # 保存图像
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'gradient_flow_dot_product_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存点积数据
    data_save_path = os.path.join(save_dir, f'dot_products_{timestamp}.npy')
    save_dict = {
        'corners': corners,
        'gradients': gradients,
        'dot_products': dot_products_np,
        'ego_flow': ego_flow,
        'patch_flow': patch_flow
    }
    if event_repr_at_corners is not None:
        save_dict['event_repr'] = event_repr_np
    np.save(data_save_path, save_dict)
    
    # 打印统计信息
    print(f"点积范围: [{dot_products_np.min():.2f}, {dot_products_np.max():.2f}]")
    print(f"平均点积: {dot_products_np.mean():.2f}")
    print(f"点积标准差: {dot_products_np.std():.2f}")
    if event_repr_at_corners is not None:
        print(f"事件表示范围: [{event_repr_np.min():.2f}, {event_repr_np.max():.2f}]")
        print(f"平均事件表示: {event_repr_np.mean():.2f}")
        print(f"事件表示标准差: {event_repr_np.std():.2f}")
    print(f"图像已保存至: {save_path}")
    print(f"数据已保存至: {data_save_path}")

def normalized_l2_loss(gt, pred, eps=1e-8):
    """
    归一化L2距离损失
    gt: torch.Tensor, shape [N] 或 [B, N]
    pred: torch.Tensor, shape [N] 或 [B, N]
    """
    # 保证为float
    gt = gt.float()
    pred = pred.float()
    
    # 归一化
    gt_norm = gt / (torch.norm(gt, p=2, dim=-1, keepdim=True) + eps)
    pred_norm = pred / (torch.norm(pred, p=2, dim=-1, keepdim=True) + eps)
    
    # L2距离的平方
    loss = torch.sum((gt_norm - pred_norm) ** 2, dim=-1)
    return loss.mean()  # 如果有batch，取均值；否则就是标量

def visualize_flow(flow, image=None, save_dir='visualization', title='Flow Visualization'):
    """
    可视化光流场
    
    Args:
        flow: 光流场 [2, H, W]
        image: 原始图像，可选
        save_dir: 保存目录
        title: 图像标题
    """
    # 转换为numpy数组
    flow = flow.detach().cpu().numpy() if torch.is_tensor(flow) else flow
    if image is not None:
        image = image.detach().cpu().numpy() if torch.is_tensor(image) else image
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建图形和子图
    fig = plt.figure(figsize=(20, 5))
    if image is not None:
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
    else:
        ax2 = fig.add_subplot(121)
        ax3 = fig.add_subplot(122)
    
    # 1. 显示原始图像（如果有）
    if image is not None:
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Original Image')
    
    # 2. 显示光流场（箭头表示）
    if image is not None:
        ax2.imshow(image, cmap='gray')
    
    # 创建网格点
    h, w = flow.shape[1:]
    y, x = np.mgrid[0:h:20, 0:w:20]  # 每20个像素采样一个点
    
    # 获取采样点的光流值
    flow_x = flow[0, y.flatten(), x.flatten()]  # 展平坐标
    flow_y = flow[1, y.flatten(), x.flatten()]
    
    # 计算光流幅值用于颜色映射
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    
    # 绘制光流场
    quiver = ax2.quiver(x.flatten(), y.flatten(), flow_x, flow_y, magnitude,
              cmap='viridis', scale=50,
              width=0.003, headwidth=3)
    
    # 添加colorbar
    cbar = fig.colorbar(quiver, ax=ax2, pad=0.1)
    cbar.set_label('Flow Magnitude')
    ax2.set_title('Flow Field (Arrows)')
    
    # 3. 显示光流场（颜色映射）
    # 计算光流的幅值和方向
    magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
    angle = np.arctan2(flow[1], flow[0])
    
    # 将角度归一化到[0, 1]范围
    angle = (angle + np.pi) / (2 * np.pi)
    
    # 创建HSV颜色映射
    hsv = np.zeros((h, w, 3))
    hsv[..., 0] = angle  # 色调：表示方向
    hsv[..., 1] = 1.0    # 饱和度：设为最大
    hsv[..., 2] = np.clip(magnitude / magnitude.max(), 0, 1)  # 亮度：表示幅值
    
    # 转换为RGB
    rgb = plt.cm.hsv(hsv[..., 0])
    rgb[..., 3] = hsv[..., 2]  # 使用alpha通道来表示幅值
    
    # 显示颜色映射的光流场
    ax3.imshow(rgb)
    ax3.set_title('Flow Field (Color Map)')
    
    # 添加方向图例
    legend_elements = [
        plt.Line2D([0], [0], color='red', label='Right'),
        plt.Line2D([0], [0], color='green', label='Up'),
        plt.Line2D([0], [0], color='blue', label='Left'),
        plt.Line2D([0], [0], color='yellow', label='Down')
    ]
    ax3.legend(handles=legend_elements, loc='upper right')
    
    # 调整布局
    plt.subplots_adjust(wspace=0.3, right=0.95)
    
    # 保存图像
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'flow_visualization_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"光流可视化已保存至: {save_path}")
    print(f"光流幅值范围: [{magnitude.min():.2f}, {magnitude.max():.2f}]")
    print(f"平均光流幅值: {magnitude.mean():.2f}")
    print(f"光流幅值标准差: {magnitude.std():.2f}")