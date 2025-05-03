import h5py
import numpy as np
from tqdm import tqdm
import torch
import os


class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool):
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.nb_channels = channels
        self.normalize = normalize

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
                        interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

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

def events_to_voxel_grid(bin, x, y, p, t, device: str='cpu'):
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

def mvsecRectifyFrames(frames, x_map, y_map):
    """
    Rectifies the spatial coordinates of input frames using mapping matrices (vectorized).
    CAUTION: Ensure frames and maps correspond to the same side (e.g., DAVIS/left or DAVIS/right)!

    :param frames: np.array of shape [N, H, W] containing frames (e.g., depth maps)
    :param x_map: np.array of shape [H, W] containing rectified x-coordinates
    :param y_map: np.array of shape [H, W] containing rectified y-coordinates
    :return: rectified frames, np.array of shape [N, H, W] with invalid pixels marked as NaN
    """
    print("\nRectifying frame coordinates (vectorized)...")
    N, H, W = frames.shape
    rectified_frames = np.full((N, H, W), np.nan, dtype=np.float32)  # Initialize with NaN
    
    # Validate map dimensions
    if x_map.shape != (H, W) or y_map.shape != (H, W):
        raise ValueError(f"Expected x_map and y_map of shape ({H}, {W}), got {x_map.shape} and {y_map.shape}")
    
    # Generate pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.ravel()  # Shape: [H*W]
    v = v.ravel()  # Shape: [H*W]
    
    # Get rectified coordinates for all pixels (load maps once)
    u_rect = x_map[v, u]  # Shape: [H*W]
    v_rect = y_map[v, u]  # Shape: [H*W]
    
    # Filter valid mappings (non-NaN)
    valid = (~np.isnan(u_rect) & ~np.isnan(v_rect))
    u_valid = u[valid]
    v_valid = v[valid]
    u_rect_valid = u_rect[valid]
    v_rect_valid = v_rect[valid]
    
    # Round rectified coordinates
    u_rect_int = np.round(u_rect_valid).astype(int)
    v_rect_int = np.round(v_rect_valid).astype(int)
    
    # Ensure rounded coordinates are within bounds
    valid_bounds = (0 <= u_rect_int) & (u_rect_int < W) & (0 <= v_rect_int) & (v_rect_int < H)
    if not np.all(valid_bounds):
        print(f"Warning: {np.sum(~valid_bounds)} out-of-bounds rectified coordinates detected")
        u_valid = u_valid[valid_bounds]
        v_valid = v_valid[valid_bounds]
        u_rect_int = u_rect_int[valid_bounds]
        v_rect_int = v_rect_int[valid_bounds]
    
    # Process each frame
    for i in tqdm(range(N), desc="Rectifying frames"):
        frame = frames[i]
        
        # Get values for valid pixels
        values = frame[v_valid, u_valid]
        valid_values = ~np.isnan(values)
        u_rect_int_valid = u_rect_int[valid_values]
        v_rect_int_valid = v_rect_int[valid_values]
        values_valid = values[valid_values]
        
        # Handle occlusions: keep smallest depth (closest)
        # Create a unique index for each rectified pixel
        indices = v_rect_int_valid * W + u_rect_int_valid
        sort_idx = np.argsort(values_valid)  # Sort by depth (ascending)
        indices_sorted = indices[sort_idx]
        values_sorted = values_valid[sort_idx]
        u_rect_sorted = u_rect_int_valid[sort_idx]
        v_rect_sorted = v_rect_int_valid[sort_idx]
        
        # Keep the first occurrence (smallest depth) for each unique index
        _, unique_idx = np.unique(indices_sorted, return_index=True)
        unique_u_rect = u_rect_sorted[unique_idx]
        unique_v_rect = v_rect_sorted[unique_idx]
        unique_values = values_sorted[unique_idx]
        
        # Assign values to rectified frame
        rectified_frames[i, unique_v_rect, unique_u_rect] = unique_values
    
    return rectified_frames

def post_process_frames(rectified_frames, inpaint_radius=10, max_iterations=3, use_interpolation=False):
    """
    Post-process rectified frames to fill NaN regions using inpainting or interpolation.
    
    :param rectified_frames: np.array of shape [N, H, W] with NaN for invalid pixels
    :param inpaint_radius: radius for inpainting neighborhood (larger for bigger holes)
    :param max_iterations: number of inpainting iterations for large holes
    :param use_interpolation: if True, use bilinear interpolation instead of inpainting
    :return: processed frames with NaN regions filled
    """
    processed_frames = rectified_frames.copy()
    N, H, W = rectified_frames.shape
    
    if use_interpolation:
        # Bilinear interpolation to fill NaN regions
        for i in tqdm(range(N)):
            frame = rectified_frames[i]
            mask = np.isnan(frame)
            if np.any(mask):
                # Create a grid of valid coordinates
                x, y = np.meshgrid(np.arange(W), np.arange(H))
                valid = ~mask
                points = np.stack([y[valid], x[valid]], axis=-1)
                values = frame[valid]
                from scipy.interpolate import griddata
                # Interpolate NaN regions
                interpolated = griddata(points, values, (y, x), method='linear', fill_value=0)
                processed_frames[i] = np.where(mask, interpolated, frame)
    else:
        # Iterative inpainting with OpenCV (Telea method)
        for i in tqdm(range(N)):
            frame = rectified_frames[i]
            for _ in range(max_iterations):
                mask = np.isnan(frame).astype(np.uint8)
                if not np.any(mask):
                    break
                frame_inp = np.where(np.isnan(frame), 0, frame)
                frame = cv2.inpaint(frame_inp, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
            processed_frames[i] = frame
    
    return processed_frames

# Example usage
# rectified_frames = mvsecRectifyFrames(frames, x_map, y_map)
# rectified_frames_filled = post_process_frames(rectified_frames)

def mvsecLoadRectificationMaps(Lx_path, Ly_path, Rx_path, Ry_path):
    """
    Loads the rectification maps for further calibration of DAVIS' spike events coordinates.

    :param Lx_path: path of the .txt file containing the mapping of the x coordinate for the left DAVIS camera
    :param Ly_path:                     ..                              y        ..          left
    :param Rx_path:                     ..                              x        ..          right
    :param Ry_path:                     ..                              y        ..          right
    :return: all corresponding mapping matrices in the form of a numpy array
    """
    print("\nloading rectification maps...")
    Lx_map = np.loadtxt(Lx_path)
    Ly_map = np.loadtxt(Ly_path)
    Rx_map = np.loadtxt(Rx_path)
    Ry_map = np.loadtxt(Ry_path)
    return Lx_map, Ly_map, Rx_map, Ry_map


def mvsecRectifyEvents(events, x_map, y_map):
    """
    Rectifies the spatial coordinates of the input spike events in accordance to the given mapping matrices.
    CAUTION: make sure events and maps correspond to the same side (DAVIS/left or DAVIS/right) !

    :param events: a list of spike events to the format [X, Y, TIME, POLARITY]
    :param x_map: np.array obtained by mvsecLoadRectificationMaps() function
    :param y_map:                       ..
    :return: rectified events, in the same format as the input events
    """
    # print("\nrectifying spike coordinates...")
    rect_events = []
    for event in tqdm(events):
        x = int(event[0])
        y = int(event[1])
        x_rect = x_map[y, x]
        y_rect = y_map[y, x]
        rect_events.append([x_rect, y_rect, event[2], event[3]])

    # convert to np.array and remove spikes falling outside of the Lidar field of view (fov)
    rect_events = np.array(rect_events)
    rect_events = rect_events[(rect_events[:, 0] >= 0)
                              & (rect_events[:, 0] <= 346)
                              & (rect_events[:, 1] >= 0)
                              & (rect_events[:, 1] <= 260)]
    return rect_events


root_dir = '/mnt/sdc/lxy/datasets/MVSEC/OpenDataLab___MVSEC/raw/MVSEC/hdf5/'
# scenario = 'outdoor_night'
scenario = 'indoor_flying'
split = '4'
save_root = '/mnt/sdc/lxy/datasets/MVSEC/processed_raw/' + f'{scenario}/{scenario}{split}/'
timestamp_root = save_root + 'index_It_left.txt'
data_path = root_dir + f'{scenario}/{scenario}{split}_data.hdf5'
gt_path = root_dir + f'{scenario}/{scenario}{split}_gt.hdf5'

with h5py.File(data_path, 'r') as data, h5py.File(gt_path, 'r') as gt:
    data = data['davis']['left']
    gt = gt['davis']['left']

    print(f"Data keys: {list(data.keys())}")
    print(f"GT keys: {list(gt.keys())}")
    image_raw = data['image_raw'][:]
    image_raw_ts = data['image_raw_ts'][:]
    image_raw_event_inds = data['image_raw_event_inds'][:]
    blended_image_rect = gt['blended_image_rect'][:]
    blended_image_rect_ts = gt['blended_image_rect_ts'][:]
    # depth_image_rect = gt['depth_image_rect'][:]
    # depth_image_rect_ts = gt['depth_image_rect_ts'][:]
    pose_ts = gt['pose_ts'][:]
    pose = gt['pose'][:]
    Levents = data['events'][:]
    # remove events occurring during take-off and landing of the drone as well
    # Levents = Levents[(Levents[:, 2] > depth_image_rect_ts[0] - 0.05) & (Levents[:, 2] < depth_image_rect_ts[-1])]
    # rectify the spatial coordinates of spike events and get rid of events falling outside of the 346x260 fov
    Lx_path = root_dir + '{}/{}_calib/{}_left_x_map.txt'.format(scenario, scenario, scenario)
    Ly_path = root_dir + '{}/{}_calib/{}_left_y_map.txt'.format(scenario, scenario, scenario)
    Rx_path = root_dir + '{}/{}_calib/{}_right_x_map.txt'.format(scenario, scenario, scenario)
    Ry_path = root_dir + '{}/{}_calib/{}_right_y_map.txt'.format(scenario, scenario, scenario)
    Lx_map, Ly_map, Rx_map, Ry_map = mvsecLoadRectificationMaps(Lx_path, Ly_path, Rx_path, Ry_path) 
    print(f"Image raw: {image_raw.shape}")
    print(f"Image raw timestamps: {image_raw_ts.shape}")
    print(f"Image raw event indices: {image_raw_event_inds.shape}")
    # print(f"Depth image rect: {depth_image_rect.shape}")
    # print(f"Depth image rect timestamps: {depth_image_rect_ts.shape}")
    print(f"Blended image rect: {blended_image_rect.shape}")
    print(f"Blended image rect timestamps: {blended_image_rect_ts.shape}")
    print(f"Pose timestamps: {pose_ts.shape}")
    print(f"Pose: {pose.shape}")

with open(timestamp_root, 'r') as f:
    # the save format is f.write(f"{img_indices[j]}\t{np.where(depth_image_rect_ts==filtered_depth_ts[j])[0][0]}\t{filtered_image_ts[j]:.18e}\t{filtered_depth_ts[j]:.18e}\n")
    lines = f.readlines()
    # Extract the first column from each line
    img_indices = [int(line.split('\t')[0]) for line in lines]
    image_timestamps = [float(line.split()[1]) for line in lines]

# check if any duplicate indices
unique_indices, counts = np.unique(img_indices, return_counts=True)
for i in range(len(counts)):
    if counts[i] > 1:
        print(f"Duplicate index {unique_indices[i]} found {counts[i]} times")
# check delta t
delta_t_img = np.diff(image_timestamps)
print('Delta t image:', delta_t_img.max(), delta_t_img.min(), delta_t_img.mean())

voxel_grid = VoxelGrid(channels=5, height=260, width=346, normalize=True)

event_root = save_root + 'event_left/event_voxel_left/'
event_stream_root = save_root + 'event_left/event_stream_left/'
event_template = '_event.hdf5'
os.makedirs(event_root, exist_ok=True)
os.makedirs(event_stream_root, exist_ok=True)

event_stream_path = save_root + 'event_left/' + 'all' + event_template 
with h5py.File(event_stream_path, 'w') as f: 
    f.create_dataset('event_stream', data=Levents) 

iter=0 
event_indices = image_raw_event_inds[img_indices]
for i in tqdm(range(len(event_indices)-1)): 
    event_id_start = event_indices[i] 
    event_id_end = event_indices[i+1] 
    # print(f"Processing events from {event_id_start} to {event_id_end}") 
    events = Levents[event_id_start:event_id_end] 
    event_stream_filename = event_stream_root + f'{i:06d}_{i+1:06d}' + event_template 
    with h5py.File(event_stream_filename, 'w') as f: 
        f.create_dataset('event_stream', data=events) 
    event_x = events[:, 0] 
    event_y = events[:, 1] 
    event_t = events[:, 2] 
    event_p = events[:, 3] 
    event_representation = events_to_voxel_grid(bin=5, x=event_x, y=event_y, p=event_p, t=event_t) 
    event_filename = event_root + f'{i:06d}_{i+1:06d}' + event_template 
    # print(f"Saving event representation to {event_filename}") 
    # print(event_representation.shape) 
    with h5py.File(event_filename, 'w') as f: 
        f.create_dataset('event_voxels', data=event_representation.cpu().numpy()) 
    iter += 1