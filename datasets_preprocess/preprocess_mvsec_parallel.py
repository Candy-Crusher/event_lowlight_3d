import os
import h5py
import numpy as np
import torch
import cv2
from tqdm import tqdm
from scipy.interpolate import griddata
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
class MVSECConfig:
    """Configuration for MVSEC dataset processing."""
    ROOT_DIR = '/mnt/sdc/lxy/datasets/MVSEC/OpenDataLab___MVSEC/raw/MVSEC/hdf5/'
    PROCESSED_DIR = '/mnt/sdc/lxy/datasets/MVSEC/processed/'
    SCENARIO = 'outdoor_day'
    SPLIT = '2'
    
    # Camera intrinsics
    FX = 223.9940010790056
    FY = 223.61783486959376
    CX = 170.7684322973841
    CY = 128.18711828338436
    
    # Voxel grid parameters
    VOXEL_CHANNELS = 5
    HEIGHT = 260
    WIDTH = 346
    NORMALIZE_VOXELS = True
    
    # Post-processing parameters
    INPAINT_RADIUS = 10
    MAX_INPAINT_ITERATIONS = 3
    
    # Parallel processing
    MAX_WORKERS = 4

    @classmethod
    def get_paths(cls, scenario: str, split: str):
        """Generate file paths for the given scenario and split."""
        data_path = os.path.join(cls.ROOT_DIR, f'{scenario}/{scenario}{split}_data.hdf5')
        gt_path = os.path.join(cls.ROOT_DIR, f'{scenario}/{scenario}{split}_gt.hdf5')
        save_root = os.path.join(cls.PROCESSED_DIR, f'{scenario}/{scenario}{split}/')
        paths = {
            'data_path': data_path,
            'gt_path': gt_path,
            'save_root': save_root,
            'timestamp_root': os.path.join(save_root, 'index_It_Dt_left.txt'),
            'depth_root': os.path.join(save_root, 'depth_left/'),
            'depth_filtered_root': os.path.join(save_root, 'depth_filtered_left/'),
            'depth_template': '_left_depth.npy',
            'blended_image_root': os.path.join(save_root, 'blended_image_left/'),
            'blended_image_template': '_left_blended_image.png',
            'image_root': os.path.join(save_root, 'image_left/'),
            'image_template': '_left.png',
            'pose_timg_root': os.path.join(save_root, 'pose_left.txt'),
            'pose_tdepth_root': os.path.join(save_root, 'pose_tdepth_left.txt'),
            'pose_origin_root': os.path.join(save_root, 'pose_origin_left.txt'),
            'event_root': os.path.join(save_root, 'event_left/event_voxel_left/'),
            'event_template': '_event.hdf5',
            'lx_map': os.path.join(cls.ROOT_DIR, f'{scenario}/{scenario}_calib/{scenario}_left_x_map.txt'),
            'ly_map': os.path.join(cls.ROOT_DIR, f'{scenario}/{scenario}_calib/{scenario}_left_y_map.txt'),
            'rx_map': os.path.join(cls.ROOT_DIR, f'{scenario}/{scenario}_calib/{scenario}_right_x_map.txt'),
            'ry_map': os.path.join(cls.ROOT_DIR, f'{scenario}/{scenario}_calib/{scenario}_right_y_map.txt'),
        }
        return paths

# Event Representation Classes
class EventRepresentation:
    """Base class for event representations."""
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        raise NotImplementedError

class VoxelGrid(EventRepresentation):
    """Voxel grid representation for events."""
    def __init__(self, channels: int, height: int, width: int, normalize: bool, device: str = 'cpu'):
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, device=device)
        self.nb_channels = channels
        self.normalize = normalize
        self.device = device

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        """Convert events to a voxel grid."""
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.voxel_grid.shape
        voxel_grid = self.voxel_grid.clone().to(self.device)

        t_norm = (C - 1) * (time - time[0]) / (time[-1] - time[0] + 1e-6)
        x0, y0, t0 = x.int(), y.int(), t_norm.int()

        value = torch.where(pol.min() == 0, 2 * pol - 1, pol)

        for xlim in [x0, x0 + 1]:
            for ylim in [y0, y0 + 1]:
                for tlim in [t0, t0 + 1]:
                    mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < C)
                    interp_weights = value * (1 - (xlim - x).abs()) * (1 - (ylim - y).abs()) * (1 - (tlim - t_norm).abs())
                    index = H * W * tlim.long() + W * ylim.long() + xlim.long()
                    voxel_grid.view(-1).put_(index[mask], interp_weights[mask], accumulate=True)

        if self.normalize:
            mask = torch.nonzero(voxel_grid, as_tuple=True)
            if mask[0].size(0) > 0:
                mean, std = voxel_grid[mask].mean(), voxel_grid[mask].std()
                if std > 0:
                    voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                else:
                    voxel_grid[mask] -= mean

        return voxel_grid

def events_to_voxel_grid(x, y, p, t, channels: int, height: int, width: int, normalize: bool, device: str = 'cpu'):
    """Convert event arrays to voxel grid."""
    t = (t - t[0]) / (t[-1] - t[0] + 1e-6) if t[-1] != t[0] else t * 0
    voxel_grid = VoxelGrid(channels, height, width, normalize, device)
    return voxel_grid.convert(
        torch.from_numpy(x.astype(np.float32)).to(device),
        torch.from_numpy(y.astype(np.float32)).to(device),
        torch.from_numpy(p.astype(np.float32)).to(device),
        torch.from_numpy(t.astype(np.float32)).to(device)
    )

# MVSEC Processing Class
class MVSECProcessor:
    """Processor for MVSEC dataset rectification and event handling."""
    @staticmethod
    def load_rectification_maps(lx_path: str, ly_path: str, rx_path: str, ry_path: str):
        """Load rectification maps for DAVIS cameras."""
        print("\nLoading rectification maps...")
        return tuple(np.loadtxt(path) for path in [lx_path, ly_path, rx_path, ry_path])

    @staticmethod
    def rectify_frames(frames: np.ndarray, x_map: np.ndarray, y_map: np.ndarray) -> np.ndarray:
        """
        Rectify spatial coordinates of input frames using mapping matrices (vectorized).
        
        :param frames: np.array of shape [N, H, W] (e.g., depth maps)
        :param x_map: np.array of shape [H, W] with rectified x-coordinates
        :param y_map: np.array of shape [H, W] with rectified y-coordinates
        :return: rectified frames, shape [N, H, W] with invalid pixels as NaN
        """
        print("\nRectifying frame coordinates (vectorized)...")
        N, H, W = frames.shape
        rectified_frames = np.full((N, H, W), np.nan, dtype=np.float32)

        if x_map.shape != (H, W) or y_map.shape != (H, W):
            raise ValueError(f"Expected x_map, y_map of shape ({H}, {W}), got {x_map.shape}, {y_map.shape}")

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u, v = u.ravel(), v.ravel()
        u_rect, v_rect = x_map[v, u], y_map[v, u]

        valid = (~np.isnan(u_rect) & ~np.isnan(v_rect))
        u_valid, v_valid = u[valid], v[valid]
        u_rect_valid, v_rect_valid = u_rect[valid], v_rect[valid]

        u_rect_int = np.round(u_rect_valid).astype(int)
        v_rect_int = np.round(v_rect_valid).astype(int)

        valid_bounds = (0 <= u_rect_int) & (u_rect_int < W) & (0 <= v_rect_int) & (v_rect_int < H)
        if not np.all(valid_bounds):
            print(f"Warning: {np.sum(~valid_bounds)} out-of-bounds rectified coordinates detected")
            u_valid, v_valid = u_valid[valid_bounds], v_valid[valid_bounds]
            u_rect_int, v_rect_int = u_rect_int[valid_bounds], v_rect_int[valid_bounds]

        for i in tqdm(range(N), desc="Rectifying frames"):
            frame = frames[i]
            values = frame[v_valid, u_valid]
            valid_values = ~np.isnan(values)
            u_rect_int_valid = u_rect_int[valid_values]
            v_rect_int_valid = v_rect_int[valid_values]
            values_valid = values[valid_values]

            indices = v_rect_int_valid * W + u_rect_int_valid
            sort_idx = np.argsort(values_valid)
            indices_sorted = indices[sort_idx]
            values_sorted = values_valid[sort_idx]
            u_rect_sorted = u_rect_int_valid[sort_idx]
            v_rect_sorted = v_rect_int_valid[sort_idx]

            _, unique_idx = np.unique(indices_sorted, return_index=True)
            rectified_frames[i, v_rect_sorted[unique_idx], u_rect_sorted[unique_idx]] = values_sorted[unique_idx]

        return rectified_frames

    @staticmethod
    def post_process_frames(frames: np.ndarray, inpaint_radius: int = 10, max_iterations: int = 3, use_interpolation: bool = False) -> np.ndarray:
        """
        Post-process rectified frames to fill NaN regions.
        
        :param frames: np.array of shape [N, H, W] with NaN for invalid pixels
        :param inpaint_radius: radius for inpainting neighborhood
        :param max_iterations: number of inpainting iterations
        :param use_interpolation: if True, use bilinear interpolation
        :return: processed frames with NaN regions filled
        """
        processed_frames = frames.copy()
        N, H, W = frames.shape

        if use_interpolation:
            for i in tqdm(range(N), desc="Interpolating frames"):
                frame = frames[i]
                mask = np.isnan(frame)
                if np.any(mask):
                    x, y = np.meshgrid(np.arange(W), np.arange(H))
                    valid = ~mask
                    points = np.stack([y[valid], x[valid]], axis=-1)
                    values = frame[valid]
                    interpolated = griddata(points, values, (y, x), method='linear', fill_value=0)
                    processed_frames[i] = np.where(mask, interpolated, frame)
        else:
            for i in tqdm(range(N), desc="Inpainting frames"):
                frame = frames[i]
                for _ in range(max_iterations):
                    mask = np.isnan(frame).astype(np.uint8)
                    if not np.any(mask):
                        break
                    frame_inp = np.where(np.isnan(frame), 0, frame)
                    frame = cv2.inpaint(frame_inp, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
                processed_frames[i] = frame

        return processed_frames

    @staticmethod
    def rectify_events(events: np.ndarray, x_map: np.ndarray, y_map: np.ndarray) -> np.ndarray:
        """
        Rectify spatial coordinates of spike events.
        
        :param events: np.array of shape [N, 4] with [x, y, time, polarity]
        :param x_map: np.array of shape [H, W] with rectified x-coordinates
        :param y_map: np.array of shape [H, W] with rectified y-coordinates
        :return: rectified events, shape [M, 4] within FoV
        """
        print("\nRectifying spike coordinates...")
        H, W = x_map.shape
        rect_events = np.zeros_like(events)
        for i in tqdm(range(len(events)), desc="Rectifying events"):
            x, y = int(events[i, 0]), int(events[i, 1])
            rect_events[i] = [x_map[y, x], y_map[y, x], events[i, 2], events[i, 3]]

        valid = (rect_events[:, 0] >= 0) & (rect_events[:, 0] <= W) & (rect_events[:, 1] >= 0) & (rect_events[:, 1] <= H)
        return rect_events[valid]

    @staticmethod
    def process_event_block(file_index: int, event_id_start: int, event_id_end: int, events: np.ndarray, x_map: np.ndarray, y_map: np.ndarray, 
                           event_root: str, event_template: str, voxel_grid: VoxelGrid) -> str:
        """Process a block of events and save as voxel grid."""
        print(f"Processing events from {event_id_start} to {event_id_end}")
        events_block = events[event_id_start:event_id_end]
        rect_events = MVSECProcessor.rectify_events(events_block, x_map, y_map)
        
        if len(rect_events) == 0:
            print(f"No valid events in block {event_id_start}-{event_id_end}")
            return None

        event_voxel = events_to_voxel_grid(
            x=rect_events[:, 0],
            y=rect_events[:, 1],
            p=rect_events[:, 3],
            t=rect_events[:, 2],
            channels=voxel_grid.nb_channels,
            height=voxel_grid.voxel_grid.shape[1],
            width=voxel_grid.voxel_grid.shape[2],
            normalize=voxel_grid.normalize,
            device=voxel_grid.device
        )
        
        event_filename = os.path.join(event_root, f'{file_index:06d}_{file_index+1:06d}{event_template}')
        print(f"Saving event representation to {event_filename}")
        with h5py.File(event_filename, 'w') as f:
            f.create_dataset('event_voxels', data=event_voxel.cpu().numpy())
        return event_filename

# Main Processing Function
def process_mvsec_dataset(config: MVSECConfig = MVSECConfig):
    """Process MVSEC dataset for rectification and event voxelization."""
    paths = config.get_paths(config.SCENARIO, config.SPLIT)
    
    # Create output directories
    for dir_path in [paths['depth_root'], paths['depth_filtered_root'], paths['blended_image_root'], 
                     paths['image_root'], paths['event_root']]:
        os.makedirs(dir_path, exist_ok=True)

    # Load data
    print("\nLoading HDF5 files...")
    with h5py.File(paths['data_path'], 'r') as data_file, h5py.File(paths['gt_path'], 'r') as gt_file:
        data = data_file['davis']['left']
        gt = gt_file['davis']['left']
        
        image_raw = data['image_raw'][:]
        image_raw_ts = data['image_raw_ts'][:]
        image_raw_event_inds = data['image_raw_event_inds'][:]
        blended_image_rect = gt['blended_image_rect'][:]
        blended_image_rect_ts = gt['blended_image_rect_ts'][:]
        depth_image_rect = gt['depth_image_rect'][:]
        depth_image_rect_ts = gt['depth_image_rect_ts'][:]
        pose_ts = gt['pose_ts'][:]
        pose = gt['pose'][:]
        events = data['events'][:]

        print(f"Data keys: {list(data.keys())}")
        print(f"GT keys: {list(gt.keys())}")
        print(f"Image raw: {image_raw.shape}")
        print(f"Image raw timestamps: {image_raw_ts.shape}")
        print(f"Image raw event indices: {image_raw_event_inds.shape}")
        print(f"Depth image rect: {depth_image_rect.shape}")
        print(f"Depth image rect timestamps: {depth_image_rect_ts.shape}")
        print(f"Blended image rect: {blended_image_rect.shape}")
        print(f"Blended image rect timestamps: {blended_image_rect_ts.shape}")
        print(f"Pose timestamps: {pose_ts.shape}")
        print(f"Pose: {pose.shape}")

    # Load timestamps
    print("\nLoading timestamps...")
    with open(paths['timestamp_root'], 'r') as f:
        lines = f.readlines()
        img_indices = [int(line.split('\t')[0]) for line in lines]
        depth_indices = [int(line.split('\t')[1]) for line in lines]
        image_timestamps = [float(line.split('\t')[2]) for line in lines]
        depth_timestamps = [float(line.split('\t')[3]) for line in lines]

    # Validate timestamps
    unique_indices, counts = np.unique(img_indices, return_counts=True)
    for idx, count in zip(unique_indices, counts):
        if count > 1:
            print(f"Duplicate index {idx} found {count} times")

    for j, (img_t, depth_t) in enumerate(zip(image_timestamps, depth_timestamps)):
        t_diff = abs(img_t - depth_t) * 1e3
        if t_diff > 20:
            print(f"Timestamp mismatch at index {j}: {t_diff:.2f} ms")

    delta_t_img = np.diff(image_timestamps)
    delta_t_depth = np.diff(depth_timestamps)
    print(f"Delta t image: max={delta_t_img.max():.6f}, min={delta_t_img.min():.6f}, mean={delta_t_img.mean():.6f}")
    print(f"Delta t depth: max={delta_t_depth.max():.6f}, min={delta_t_depth.min():.6f}, mean={delta_t_depth.mean():.6f}")

    # Load rectification maps
    lx_map, ly_map, rx_map, ry_map = MVSECProcessor.load_rectification_maps(
        paths['lx_map'], paths['ly_map'], paths['rx_map'], paths['ry_map']
    )

    # Initialize voxel grid
    voxel_grid = VoxelGrid(
        channels=config.VOXEL_CHANNELS,
        height=config.HEIGHT,
        width=config.WIDTH,
        normalize=config.NORMALIZE_VOXELS,
        device='cpu'  # Change to 'cuda' if GPU available
    )

    # Process events in parallel
    print("\nProcessing events in parallel...")
    event_indices = image_raw_event_inds  # Assuming event indices align with image_raw_event_inds
    chunk_size = max(1, len(event_indices) // config.MAX_WORKERS)
    chunks = [(i,event_indices[i], event_indices[min(i + 1, len(event_indices) - 1)]) 
              for i in range(0, len(event_indices) - 1, chunk_size)]

    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                MVSECProcessor.process_event_block,
                file_index, start_idx, end_idx, events, lx_map, ly_map,
                paths['event_root'], paths['event_template'], voxel_grid
            ) for file_index,start_idx, end_idx in chunks
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing event chunks"):
            try:
                result = future.result()
                if result:
                    print(f"Finished processing: {result}")
                else:
                    print("No valid events in chunk")
            except Exception as e:
                print(f"Error occurred: {e}")

    # Example: Rectify and post-process depth frames (uncomment to use)
    # rectified_depth = MVSECProcessor.rectify_frames(depth_image_rect, lx_map, ly_map)
    # processed_depth = MVSECProcessor.post_process_frames(
    #     rectified_depth,
    #     inpaint_radius=config.INPAINT_RADIUS,
    #     max_iterations=config.MAX_INPAINT_ITERATIONS,
    #     use_interpolation=False
    # )

if __name__ == "__main__":
    process_mvsec_dataset()