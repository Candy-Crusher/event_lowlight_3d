import h5py
import torch
import torch.nn.functional as F

def read_voxel_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        voxel_data = f['event_voxels'][:]
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