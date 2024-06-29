import open3d as o3d
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation
import struct

def read_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return pcd.points

def render_color():
    points = read_ply("/home/rick/Datasets/Custom_tum/points/184087.380093.ply")
    points = np.hstack((points, np.array([[0,0,0] for i in range(len(points))])))
    img = cv2.imread("/home/rick/Datasets/Custom_tum/rgb/184087.380093.png")
    K = np.array([2071.184147  ,0 , 2051.995468, 0, 2071.184147,  1589.171711, 0, 0, 1]).reshape((3,3))

    for p in points:
        if (p[2] < 0):
            continue
        p_xyz = np.array([p[:3]]).transpose()
        uv = K @ p_xyz / p_xyz[2][0]
        u = round(uv[0][0])
        v = round(uv[1][0])
        if u > 0 and u < img.shape[1] and v > 0 and v < img.shape[0]:
            p[5] = img[v][u][0]/255.0
            p[4] = img[v][u][1]/255.0
            p[3] = img[v][u][2]/255.0

    tmp = np.array(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tmp[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(tmp[:,3:])
    pcd.normals  = o3d.utility.Vector3dVector(np.zeros((tmp.shape[0], 3)))
    # 保存为PLY文件
    o3d.io.write_point_cloud("/home/rick/Datasets/Custom_tum/color.ply", pcd)

class Pose:
    tdpose_format = '<I d d d d f f f f'  # Struct format string
    struct_size = struct.calcsize(tdpose_format)
    def __init__(self, id, timestamp, posX, posY, posZ, qX, qY, qZ, qW):
        self.id = id
        self.timestamp = timestamp
        self.posX = posX
        self.posY = posY
        self.posZ = posZ
        self.qX = qX
        self.qY = qY
        self.qZ = qZ
        self.qW = qW
        self.R = Rotation.from_quat(np.array([self.qX, self.qY, self.qZ, self.qW])).as_matrix()
        self.t = np.array([self.posX, self.posY, self.posZ]).transpose()
        pose_T = np.eye(4)
        pose_T[:3,:3] = self.R
        pose_T[:3,3] = self.t
        self.T = pose_T
        self.T_inv = np.linalg.inv(self.T)

def transform_points_world_to_body(points, cur_pose):
    points = np.array(points).transpose()
    r = cur_pose.R
    t = cur_pose.t.reshape((3,1))
    points = (r@points + t).transpose()
    return points

def save_point_cloud_to_ply(ply_file, point_cloud):
    tmp = np.array(point_cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tmp[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((tmp.shape[0], 3)))
    pcd.normals  = o3d.utility.Vector3dVector(np.zeros((tmp.shape[0], 3)))
    o3d.io.write_point_cloud(ply_file, pcd)
    
def body2world():
    pose = None
    all_points = np.array([])
    with open("/home/rick/Datasets/Custom/groundtruth.txt") as f:
        index = 0
        while True:
            index += 1
            line = f.readline()
            if index == 30:
                break
            if line.startswith("#"):
                continue
            line = line.split(" ")
            timestamp = line[0]
            points = read_ply(f"/home/rick/Datasets/Custom/point/{timestamp}.ply")
            p = [float(i) for i in line]
            pose = Pose(index,*p)
            if len(all_points) == 0:
                all_points = transform_points_world_to_body(points, pose)
            else:
                np.vstack((all_points, transform_points_world_to_body(points, pose)))
    save_point_cloud_to_ply("./test.ply",all_points)

body2world()