
import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import glob
import cv2

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2,imreadlowlight_cv2
from dust3r.utils.event import read_voxel_hdf5
from dust3r.utils.misc import get_stride_distribution

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

def depth_read(filename):
    depth = np.load(filename)
    return depth

def xyzqxqyqxqw_to_c2w(xyzqxqyqxqw):
    xyzqxqyqxqw = np.array(xyzqxqyqxqw, dtype=np.float32)
    #NOTE: we need to convert x_y_z coordinate system to z_x_y coordinate system
    z, x, y = xyzqxqyqxqw[:3]
    qz, qx, qy, qw = xyzqxqyqxqw[3:]
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    c2w[:3, 3] = np.array([x, y, z])
    return c2w

class MVSEC(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='data/tartanair',
                 dset='train',
                 use_augs=False,
                 S=2,
                 strides=[8],
                 clip_step=2,
                 quick=False,
                 verbose=False,
                 dist_type=None,
                 *args, 
                 **kwargs
                 ):

        TRAIN_RANGE = [17,8541]
        TEST_REANGE = [10368, 12179]
        print('loading mvsec dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'mvsec'
        self.split = dset
        self.S = S # number of frames
        self.verbose = verbose

        self.use_augs = use_augs
        self.dset = 'Hard'

        self.rgb_paths = []
        self.event_paths = []
        self.depth_paths = []
        self.normal_paths = []
        self.traj_paths = []
        self.annotations = []
        self.full_idxs = []
        self.sample_stride = []
        self.strides = strides

        self.sequences = []
        self.sequences.append(os.path.join(dataset_location)) #'data/MVSEC/processed/outdoor_day/outdoor_day2'
        
        if self.split == 'train':
            self.split_index = TRAIN_RANGE
        elif self.split == 'test':
            self.split_index = TEST_REANGE
        for seq in self.sequences:
            if self.verbose: 
                print('seq', seq)

            rgb_path = os.path.join(seq, 'image_left')
            event_voxel_path = os.path.join(seq, 'event_left', 'event_voxel_left')
            depth_path = os.path.join(seq, 'depth_left')
            caminfo_path = os.path.join(seq, 'pose_left.txt')
            caminfo = []
            # Read the file line by line
            with open(caminfo_path, 'r') as f:
                for line in f:
                    # Split the line into values
                    values = line.strip('\t').split()
                    
                    # The remaining values are the flattened pose matrix
                    pose_flattened = np.array([float(v) for v in values])
                    
                    # Reshape the flattened pose into a 4x4 matrix
                    pose_matrix = pose_flattened.reshape(4, 4)
                    caminfo.append(pose_matrix)
            caminfo = np.array(caminfo)
            
            for stride in strides:
                for ii in range(self.split_index[0],self.split_index[1]-1-self.S*stride+1, clip_step):
                    full_idx = ii + np.arange(self.S)*stride
                    self.rgb_paths.append([os.path.join(rgb_path, '%06d_left.png' % idx) for idx in full_idx])
                    self.event_paths.append([os.path.join(event_voxel_path, '%06d_%06d_event.hdf5' % (idx, idx + 1)) for idx in full_idx])
                    self.depth_paths.append([os.path.join(depth_path, '%06d_left_depth.npy' % idx) for idx in full_idx])
                    self.annotations.append(caminfo[full_idx])
                    self.full_idxs.append(full_idx)
                    self.sample_stride.append(stride)
                if self.verbose:
                    sys.stdout.write('.')
                    sys.stdout.flush()

        
        fx = 223.9940010790056      # focal length x
        fy = 223.61783486959376     # focal length y
        cx = 170.7684322973841      # optical center x
        cy = 128.18711828338436     # optical center y
        width = 346
        height = 260

        self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.stride_counts = {}
        self.stride_idxs = {}
        for stride in strides:
            self.stride_counts[stride] = 0
            self.stride_idxs[stride] = []
        for i, stride in enumerate(self.sample_stride):
            self.stride_counts[stride] += 1
            self.stride_idxs[stride].append(i)
        print('stride counts:', self.stride_counts)
        
        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))
    
    def _resample_clips(self, strides, dist_type):

        # Get distribution of strides, and sample based on that
        dist = get_stride_distribution(strides, dist_type=dist_type)
        dist = dist / np.max(dist)
        max_num_clips = self.stride_counts[strides[np.argmax(dist)]]
        num_clips_each_stride = [min(self.stride_counts[stride], int(dist[i]*max_num_clips)) for i, stride in enumerate(strides)]
        print('resampled_num_clips_each_stride:', num_clips_each_stride)
        resampled_idxs = []
        for i, stride in enumerate(strides):
            resampled_idxs += np.random.choice(self.stride_idxs[stride], num_clips_each_stride[i], replace=False).tolist()

        self.rgb_paths = [self.rgb_paths[i] for i in resampled_idxs]
        self.event_paths = [self.event_paths[i] for i in resampled_idxs]
        self.depth_paths = [self.depth_paths[i] for i in resampled_idxs]
        self.annotations = [self.annotations[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]

    def __len__(self):
        return len(self.rgb_paths)
    
    def _get_views(self, index, resolution, rng):

        rgb_paths = self.rgb_paths[index]
        event_paths = self.event_paths[index]
        depth_paths = self.depth_paths[index]
        full_idx = self.full_idxs[index]
        annotations = self.annotations[index]

        views = []
        for i in range(2):
            impath = rgb_paths[i]
            eventpath = event_paths[i]
            depthpath = depth_paths[i]

            # load camera params
            # pose is saved as flatten matrix
            camera_pose = np.array(annotations[i], dtype=np.float32)
            # camera_pose = np.linalg.inv(camera_pose)

            # load image and depth
            rgb_image = imread_cv2(impath, options=cv2.IMREAD_GRAYSCALE)
            rgb_image = cv2.cvtColor(rgb_image,cv2.COLOR_GRAY2RGB)

            # rgb_image = imreadlowlight_cv2(impath, brightness_factor=0.5, gamma=1.5, noise_std=0.05, as_uint8=True, lightup="clahe")
            event_voxel = read_voxel_hdf5(eventpath)    # 6 H W
            depthmap = depth_read(depthpath).astype(np.float32)
            # set nan pixel to zero depth
            depthmap[np.isnan(depthmap)] = 0

            rgb_image, depthmap, intrinsics, event_voxel = self._crop_resize_if_necessary(
                rgb_image, depthmap, self.intrinsics, resolution, rng=rng, info=impath, event_voxel=event_voxel)
            views.append(dict(
                img=rgb_image,
                event_voxel=event_voxel,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=rgb_paths[i].split('/')[-5]+'-'+rgb_paths[i].split('/')[-3],
                instance=osp.split(rgb_paths[i])[1],
            ))
        return views

if __name__ == "__main__":

    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    use_augs = False
    S = 2
    strides = [1,2,3,4,5,6,7,8,9]
    clip_step = 2
    quick = False  # Set to True for quick testing


    def visualize_scene(idx):
        views = dataset[idx]
        assert len(views) == 2
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 1)
        label = views[0]['label']
        instance = views[0]['instance']
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                        focal=views[view_idx]['camera_intrinsics'][0, 0],
                        color=(255, 0, 0),
                        image=colors,
                        cam_size=cam_size)
        path = f"./tmp/tartanair/tartanair_scene_{label}_{instance}.glb"
        return viz.save_glb(path)

    dataset = MVSEC(
        use_augs=use_augs,
        S=S,
        strides=strides,
        clip_step=clip_step,
        quick=quick,
        verbose=False,
        resolution=(512,384), 
        dist_type='linear_9_1',
        aug_crop=16)
    
    idxs = np.arange(0, len(dataset)-1, (len(dataset)-1)//10)
    for idx in idxs:
        print(f"Visualizing scene {idx}...")
        visualize_scene(idx)