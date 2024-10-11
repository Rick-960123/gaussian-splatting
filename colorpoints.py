import numpy as np
import cv2
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
from scene.tum import TUMDataset
import torch
import torch.nn as nn
import torch.optim as optim
from colorModel import *

def read_ply(path):
        pcd = o3d.io.read_point_cloud(path)
        return pcd.points

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

def save_point_cloud_to_ply(point_cloud, ply_file):
    tmp = np.array(point_cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tmp[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(tmp[:,3:6])
    # pcd.normals  = o3d.utility.Vector3dVector(tmp[:,6:])
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
    ply_output_dir = "/home/rick/Datasets/Custom"
    K = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0,0,1]])

    poses, colors, depths, poseList = TUMDataset(ply_output_dir).load_poses()
    points = read_ply(os.path.join(ply_output_dir, "points3D.ply"))
    
    clrmodel = colorModel(points, K)
    optimizer = optim.Adam([
        {'params': clrmodel._xyz},
        {'params': clrmodel._rgb},
        {'params': clrmodel.R},
        {'params': clrmodel.t}
    ], lr=0.01)

    num_epochs = 5
    
    for epoch in range(num_epochs):
        for i, depth_file in enumerate(colors):
            if not depth_file.endswith('.png'):
                continue
            if i > 100: 
                break

            clrmodel.setNext(read_rgb_image(os.path.join(colors[i])), poseList[i]) 

            if epoch > 150:
                # 收敛后优化 R 和 t
                clrmodel.R.requires_grad = True
                clrmodel.t.requires_grad = True
    
            optimizer.zero_grad()
            loss = projection_loss(clrmodel)
            loss.backward()
            optimizer.step()

            if epoch % 1 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

    rgb = clrmodel._rgb.detach().cpu().numpy()
    points = np.hstack((points, rgb))
    save_point_cloud_to_ply(points, os.path.join(ply_output_dir, "test.ply"))

if __name__ == '__main__':
    main()




