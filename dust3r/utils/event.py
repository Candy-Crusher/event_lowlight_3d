import h5py

def read_voxel_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        voxel_data = f['event_voxels'][:]
    # B, H, W = voxel_data.shape
    # voxel_data = voxel_data.reshape(3, 2, H, W).sum(axis=1)  # New shape: (B//2, H, W)
    return voxel_data