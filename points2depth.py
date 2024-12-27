import cv2
import numpy as np
import queue
import os
import open3d as o3d
import laspy
import torch
import struct
from scipy.spatial.transform import Rotation

class Camera:
    def __init__(self,id, model, width, height, fx, fy, cx, cy):
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.fovx = 2 * np.arctan(width / (2 * fx))
        self.fovy = 2 * np.arctan(height / (2 * fy))
        self.matrix = np.array([[self.fx, 0, self.cx],
                                 [0, self.fy, self.cy],
                                 [0, 0, 1]])

    def getlist(self):
        return [self.id,
                self.model,
                self.width,
                self.height,
                self.fx,
                self.fy,
                self.cx,
                self.cy]

class ImageFrame:
    def __init__(self, timestamp=0.0, img=None):
        self.timestamp = timestamp
        self.img = img


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

class PreProcess:
    def __init__(self, pose_path, las_path, video_path, video_timestamp_path, T_c2b, save_path, camera:Camera):
        self._pose_file = open(pose_path, 'rb')  # Assuming binary read mode for pose file
        self._p_idx = 0
        self._imgIdx = 0
        self._lastVideoTime = 0
        self._index = 0
        self.save_path = save_path
        self.image_cache = queue.Queue()
        self._videoTimeList = []
        self._T_c2b = T_c2b
        self._camera_time_error = 18
        self._camera = camera

        with open(video_timestamp_path, 'r') as f:
            for line in f:
                self._videoTimeList.append(float(line.strip()) + self._camera_time_error)

        # Ensure directories exist
        os.makedirs(os.path.join(save_path, "depth"), exist_ok=True)
        self.depth_txt = os.path.join(self.save_path, "depth.txt")

        if os.path.exists(self.depth_txt):
            os.remove(self.depth_txt)

        self._cap = cv2.VideoCapture(video_path)
        _las = laspy.read(las_path)
        self._points = self.transform_points_body_to_camera(_las.xyz)

    def poseNext(self):
        data = self._pose_file.read(Pose.struct_size)
        if not data:
            return None

        pose_data = struct.unpack(Pose.tdpose_format, data)
        pose = Pose(*pose_data)

        return pose
    
    def get_depth_image(self, pose):
        points = torch.tensor(self._points, dtype=torch.float32) 
        ones = torch.ones((points.shape[0], 1), dtype=torch.float32)
        points_homogeneous = torch.cat([points, ones], dim=1)  # shape (N, 4)

        extrinsic = torch.tensor(pose.T_inv, dtype=torch.float32)  # shape (4, 4)
        points_camera = points_homogeneous @ extrinsic.T  # shape (N, 4)
        
        intrinsic = torch.tensor(self._camera.matrix, dtype=torch.float32)  # shape (3, 3)
        
        points_camera = points_camera[:, :3]
        points_image = points_camera @ intrinsic.T  # shape (N, 3)
        
        points_image[:, :2] /= points_image[:, 2:3]
        
        depth = torch.norm(points_camera, dim=1)  # shape (N,)
        
        depth_image = torch.full((self._camera.height, self._camera.width), float('inf'), dtype=torch.float32)
        
        x_coords = points_image[:, 0].long()
        y_coords = points_image[:, 1].long()
        
        valid_mask = (x_coords >= 0) & (x_coords < self._camera.width) & (y_coords >= 0) & (y_coords < self._camera.height)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        depth = depth[valid_mask]
        
        depth_image[y_coords, x_coords] = torch.min(depth_image[y_coords, x_coords], depth)
        
        depth_image[depth_image == float('inf')] = 0
        depth_image = (depth_image * 1000).clamp(10, 100000)
        
        depth_image_numpy = depth_image.numpy()
        return depth_image_numpy.astype(np.uint16)
    
    def save_point_cloud_to_ply(self, ply_file, point_cloud):
        tmp = np.array(point_cloud)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tmp[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((tmp.shape[0], 3)))
        pcd.normals  = o3d.utility.Vector3dVector(np.zeros((tmp.shape[0], 3)))
        o3d.io.write_point_cloud(ply_file, pcd)
    
    def frustum_filter(self, pose_inv, near_plane=0.01, far_plane=50, aspect_ratio=1000):
        # 转换点到相机坐标系
        p_hom = pose_inv[:3, :3] @ self._points.T + pose_inv[:3, 3].reshape(-1, 1)

        valid_mask = (p_hom[2, :] > near_plane) & (p_hom[2, :] < far_plane)    
        # 计算x和y的视角范围
        tan_half_hfov = np.tan(self._camera.fovx / 2)
        tan_half_vfov = np.tan(self._camera.fovy / 2)
        valid_mask &= (p_hom[0, :] > -p_hom[2, :] * tan_half_hfov) & (p_hom[0, :] < p_hom[2, :] * tan_half_hfov)
        valid_mask &= (p_hom[1, :] > -p_hom[2, :] * tan_half_vfov) & (p_hom[1, :] < p_hom[2, :] * tan_half_vfov)
        p_hom = p_hom.transpose()
        p_hom = p_hom[valid_mask]
        self.save_point_cloud_to_ply("./test.ply", p_hom)
        return p_hom

    def transform_points_body_to_camera(self, points):
        points = np.array(points).transpose()

        T_b2c = np.linalg.inv(self._T_c2b)
        r = T_b2c[:3,:3]
        t = T_b2c[:3,3].reshape((3,1))

        points = (r@points + t).transpose()
        return points
    
    def run(self):
        while True:
            cur_pose = self.poseNext()
            depth_img = self.get_depth_image(cur_pose)
            cv2.imwrite(os.path.join(self.save_path, f"depth/{cur_pose.timestamp}.png"), depth_img)

            with open( self.depth_txt, 'a') as ofs_depth:

                if self._index == 0:
                    ofs_depth.write("#  timestamp filename\n")
                ofs_depth.write(f"{cur_pose.timestamp} depth/{cur_pose.timestamp}.png\n")
                
            self._index += 1

            print(self._index)

            if self._index == 50:
                break
        return True


if __name__ == "__main__":
    base_path = "/home/rick/Datasets/SN_00250/SLAM_PRJ_001"
    pose_path = os.path.join(base_path, "2024-04-23_13-46-50_570/optimised_2024-04-23_14-18-25_662.bin")
    las_path = os.path.join(base_path, "2024-04-23_13-46-50_570/optimised_2024-04-23_14-18-25_662.las")

    video_path = os.path.join(base_path, "OPTICAL_CAM/optcam_1.h265")
    video_timestamp_path = os.path.join(base_path, "OPTICAL_CAM/optcam_1.ts")
    save_path = os.path.join(base_path, "depth")

    T = np.array([-0.037767,
                            -0.001235,
                            -0.999282,
                            0.059832,
                            -0.999215,
                            -0.011807,
                            0.037780,
                            -0.001428,
                            -0.011845,
                            0.999924,
                            -0.000787,
                            0.017868,
                            0.000000,
                            0.000000,
                            0.000000,
                            1.000000]).reshape((4,4))
    
    tmp_T = np.array([1,0,0,0,
                         0,-1,0,0,
                         0,0,-1,0,
                         0,0,0,1]).reshape((4,4)) 
    
    camera_pose =  T @ tmp_T
    
    camera = Camera(0, "PINHOLE", 4000, 3000, 2071.184147, 2071.184147, 2051.995468, 1589.171711)
    pp = PreProcess(pose_path, las_path, video_path, video_timestamp_path, camera_pose, save_path, camera)
    pp.run()