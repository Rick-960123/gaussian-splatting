import numpy as np
import torch
import open3d as o3d
import os

# 定义旋转矩阵 R
R = torch.tensor([[1, 0, 0],
                  [0, -1, 0],
                  [0, 0, 1]], dtype=torch.float32)

def read_poses_from_txt(file_path):
    # 读取txt文件中的位姿
    poses = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split()
            if data[0] == "#":
                continue
            timestamp = float(data[0])
            tx, ty, tz = float(data[1]), float(data[2]), float(data[3])
            qx, qy, qz, qw = float(data[4]), float(data[5]), float(data[6]), float(data[7])
            pose = {
                'timestamp': timestamp,
                'translation': torch.tensor([tx, ty, tz]),
                'quaternion': torch.tensor([qw, qx, qy, qz])  # w, x, y, z
            }
            poses.append(pose)
    return poses

def quaternion_to_rotation_matrix(quaternion):
    # 四元数转旋转矩阵
    qw, qx, qy, qz = quaternion
    R = torch.tensor([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]
    ])
    return R

def transform_pose_to_left_handed(pose):
    # 将右手系位姿转换到左手系
    translation = pose['translation']
    quaternion = pose['quaternion']
    
    # 转换旋转矩阵
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    transformed_rotation_matrix = torch.matmul(R, torch.matmul(rotation_matrix, R.T))
    
    # 转换平移向量
    transformed_translation = torch.matmul(R, translation)
    
    return transformed_rotation_matrix, transformed_translation

def read_point_cloud_from_ply(file_path):
    # 读取ply文件中的点云数据
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return torch.tensor(points, dtype=torch.float32)

def write_transformed_poses_to_txt(poses, p, file_path):
    # 将转换后的位姿写入txt文件
    with open(file_path, 'w') as file:
        index = 0
        for pose in poses:
            rotation_matrix, translation = pose
            qw, qx, qy, qz = rotation_matrix_to_quaternion(rotation_matrix)
            line = f"{p[index]['timestamp']} {translation[0].item()} {translation[1].item()} {translation[2].item()} {qx} {qy} {qz} {qw}\n"
            file.write(line)
            index+=1
    print(123)
def rotation_matrix_to_quaternion(R):
    # 旋转矩阵转四元数
    qw = torch.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    return qw, qx, qy, qz

def write_point_cloud_to_ply(points, file_path):
    # 将点云数据写入ply文件
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.numpy())
    pcd.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3))*0.2)
    pcd.normals  = o3d.utility.Vector3dVector(np.zeros((points.shape[0], 3)))
    o3d.io.write_point_cloud(file_path, pcd)

def transform_point_cloud_to_left_handed(points):
    # 将右手系点云转换到左手系
    transformed_points = torch.matmul(points, R.T)
    return transformed_points

# 示例文件路径
# pose_file_path = "/home/rick/Datasets/Custom_tum/groundtruth_camera.txt"
point_cloud_file_path = "/home/rick/Datasets/Custom_tum/points3D.ply"
# transformed_pose_file_path = "/home/rick/Datasets/Custom_tum/transformed_poses.txt"
transformed_point_cloud_file_path = "/home/rick/Datasets/Custom_tum/transformed_point_cloud.ply"

# 读取位姿和点云
# poses = read_poses_from_txt(pose_file_path)
points = read_point_cloud_from_ply(point_cloud_file_path)

# 转换位姿和点云
# transformed_poses = [transform_pose_to_left_handed(pose) for pose in poses]
# transformed_points = transform_point_cloud_to_left_handed(points)

# 保存转换后的位姿和点云
# write_transformed_poses_to_txt(transformed_poses, poses, transformed_pose_file_path)
write_point_cloud_to_ply(points, transformed_point_cloud_file_path)

print("Transformation complete.")
