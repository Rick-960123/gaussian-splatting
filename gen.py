import numpy as np
import cv2
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
from scene.tum import TUMDataset
def read_depth_image(depth_file):
    depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    return depth_image

def read_pose_file(pose_file):
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            pose = list(map(float, line.strip().split()))
            poses.append(pose)
    return np.array(poses)

def read_rgb_image(rgb_file):
    rgb_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
    return rgb_image
def depth_to_point_cloud(depth_image, rgb_image, pose, fx, fy, cx, cy, scale=5000.0):
    height, width = depth_image.shape
    point_cloud = []

    for v in range(height):
        for u in range(width):
            z = depth_image[v, u] / scale
            if z == 0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            point = np.array([x, y, z, 1.0])
            point = pose @ point
            b, g, r = rgb_image[v, u]
            t = pose[0:3, 3]
            n = (t - point[0:3]) / np.linalg.norm(t - point[0:3])
            point_cloud.append([point[0], point[1], point[2], r/255.0, g/255.0, b/255.0, n[0], n[1], n[2]])

    return point_cloud

def save_point_cloud_to_ply(point_cloud, ply_file):
    tmp = np.array(point_cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tmp[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(tmp[:,3:6])
    pcd.normals  = o3d.utility.Vector3dVector(tmp[:,6:])
    # 保存为PLY文件
    o3d.io.write_point_cloud(ply_file, pcd)

def downsample_point_cloud(point_cloud, voxel_size=0.05):
    tmp = np.array(point_cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tmp[:,:3])
    pcd.colors = o3d.utility.Vector3dVector((tmp[:,3:6]))
    pcd.normals  = o3d.utility.Vector3dVector(tmp[:,6:])
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.hstack((downsampled_pcd.points, downsampled_pcd.colors, downsampled_pcd.normals)) 

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
def main():
    poses, colors, depths, T_poses = TUMDataset("/home/rick/Datasets/Custom/").load_poses()
    ply_output_dir = "/home/rick/Datasets/Custom"
    fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6

    downsampled_point_cloud = np.array([])
    for i, depth_file in enumerate(depths):
        if not depth_file.endswith('.png') or i % 10 != 0:
            continue
        
        rgb_image = read_rgb_image(os.path.join(colors[i]))
        depth_image = read_depth_image(os.path.join(depth_file))
        pose = T_poses[i]

        point_cloud = depth_to_point_cloud(depth_image, rgb_image, pose, fx, fy, cx, cy)
        point_cloud_dowm = downsample_point_cloud(point_cloud, voxel_size=0.02)

        if downsampled_point_cloud.__len__() == 0:
            downsampled_point_cloud = point_cloud_dowm
        else:
            downsampled_point_cloud = np.vstack((downsampled_point_cloud, point_cloud_dowm))

    downsampled_point_cloud = downsample_point_cloud(downsampled_point_cloud, voxel_size=0.05)
    ply_file = os.path.join(ply_output_dir, 'points3D.ply')
    save_point_cloud_to_ply(downsampled_point_cloud, ply_file)
    print(f'Saved {ply_file}')

if __name__ == '__main__':
    main()
