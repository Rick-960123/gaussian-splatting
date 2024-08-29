import cv2
import numpy as np
import queue
import os
import open3d as o3d
import laspy
import torch
import struct
from scipy.spatial.transform import Rotation, Slerp

class IMUPose:
    tdpose_format = '<d d d d d d d d d d d d d d d d d d d d d d'  # Struct format string
    struct_size = struct.calcsize(tdpose_format)
    def __init__(self, timestamp, yaw, pitch, roll, vx, vy, vz, px, py, pz,
                 gyro_x_drift, gyro_y_drift, gyro_z_drift,
                 acc_x_drift, acc_y_drift, acc_z_drift,
                 scallor_gyro_x, scallor_gyro_y, scallor_gyro_z,
                 scallor_acc_x, scallor_acc_y, scallor_acc_z):
        self.timestamp = timestamp
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.px = px
        self.py = py
        self.pz = pz
        self.gyro_x_drift = gyro_x_drift
        self.gyro_y_drift = gyro_y_drift
        self.gyro_z_drift = gyro_z_drift
        self.acc_x_drift = acc_x_drift
        self.acc_y_drift = acc_y_drift
        self.acc_z_drift = acc_z_drift
        self.scallor_gyro_x = scallor_gyro_x
        self.scallor_gyro_y = scallor_gyro_y
        self.scallor_gyro_z = scallor_gyro_z
        self.scallor_acc_x = scallor_acc_x
        self.scallor_acc_y = scallor_acc_y
        self.scallor_acc_z = scallor_acc_z

        self.R_sci = Rotation.from_euler("zyx", np.array([self.roll, self.pitch, self.yaw]))
        self.R = self.R_sci.as_matrix()
        self.q = self.R_sci.as_quat()
        self.t = np.array([self.px, self.py, self.pz]).transpose()
        pose_T = np.eye(4)
        pose_T[:3,:3] = self.R
        pose_T[:3,3] = self.t
        self.T = pose_T
        self.T_inv = np.linalg.inv(self.T)

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
        self.R_sci = Rotation.from_quat(np.array([self.qX, self.qY, self.qZ, self.qW]))
        self.R = self.R_sci.as_matrix()
        self.q = self.R_sci.as_quat()
        self.t = np.array([self.posX, self.posY, self.posZ]).transpose()
        pose_T = np.eye(4)
        pose_T[:3,:3] = self.R
        pose_T[:3,3] = self.t
        self.T = pose_T
        self.T_inv = np.linalg.inv(self.T)

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

    def getlist(self):
        return [self.id,
                self.model,
                self.width,
                self.height,
                self.fx,
                self.fy,
                self.cx,
                self.cy]

class Lidar:
    def __init__(self, id, bias, sigma):
       pass
    
class Imu:
    def __init__(self, id, bias, sigma):
        self.id = id
        self.bias = bias
        self.sigma = sigma
    
class ImageFrame:
    def __init__(self, timestamp=0.0, img=None, pose=None):
        self.timestamp = timestamp
        self.img = img
        self.pose = pose

class LidarFrame:
    def __init__(self, timestamp=0.0, points=None, pose=None):
        self.timestamp = timestamp
        self.points = points
        self.pose = pose

class ImuFrame:
    tdpose_format = '<d f f f f f f f f f f'  # Struct format string
    struct_size = struct.calcsize(tdpose_format)
    def __init__(self, timestamp, Accel_x, Accel_y, Accel_z, Gyro_x, Gyro_y, Gyro_z, Q0, Q1, Q2, Q3):
        self.timestamp = timestamp
        self.Accel_x = Accel_x
        self.Accel_y = Accel_y
        self.Accel_z = Accel_z
        self.Gyro_x = Gyro_x
        self.Gyro_y = Gyro_y
        self.Gyro_z = Gyro_z
        self.Q0 = Q0
        self.Q1 = Q1
        self.Q2 = Q2
        self.Q3 = Q3


class PreProcess:
    def __init__(self, pose_path, las_path, imu_pose_path, video_path, video_timestamp_path, T_c2b, save_path, camera, duration=200):
        self._pose_file = open(pose_path, 'rb')  # Assuming binary read mode for pose file
        self._imu_pose_file = open(imu_pose_path, 'rb')  # Assuming binary read mode for pose file

        self._cap = cv2.VideoCapture(video_path)
        self._las = laspy.read(las_path)

        self._point_idx = 0
        self._img_idx = 0
        self._pose_idx = 0
        self._imu_pose_idx = 0
        self._img_pose_idx = 0

        self._lastVideoTime = 0
        self._save_path = save_path
        self._videoTimeList = []
        self._T_c2b = T_c2b
        self._camera_time_error = 18
        self._camera = camera
        self._imu_pose_list = []
        self._whole_points = None

        with open(video_timestamp_path, 'r') as f:
            for line in f:
                self._videoTimeList.append(float(line.strip()) + self._camera_time_error)
        
        while True:
            cur_pose = self.imuPoseNext()
            if cur_pose is None:
                break
            self._imu_pose_list.append(cur_pose)
        
        self._stop_time = self._imu_pose_list[0].timestamp + duration

        # Ensure directories exist
        os.makedirs(os.path.join(self._save_path, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self._save_path, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self._save_path, "point"), exist_ok=True)

        self.pose_img_txt = os.path.join(self._save_path, "groundtruth.txt")
        self.pose_lidar_txt = os.path.join(self._save_path, "groundtruth_lidar.txt")
        self.rgb_txt = os.path.join(self._save_path, "rgb.txt")
        self.depth_txt = os.path.join(self._save_path, "depth.txt")
        self.point_txt = os.path.join(self._save_path, "point.txt")
        self.camera_txt =  os.path.join(self._save_path, "cameras.txt")

    def imgNext(self):
        cur_image = ImageFrame()
        first_pose = self._imu_pose_list[0]
        latest_pose = self._imu_pose_list[-1]

        while True:
            if not self._cap.isOpened():
                return None
            
            ret, frame = self._cap.read()
            if ret and frame is not None:
                cur_image.img = frame
                if self._img_idx < len(self._videoTimeList):
                    cur_image.timestamp = self._videoTimeList[self._img_idx]
                    self._img_idx += 1
                    self._lastVideoTime = cur_image.timestamp
                else:
                    print(f"\n\nend of video time list {self._lastVideoTime}\n")
            else:
                print(f"\n\nend of video {self._lastVideoTime}\n")
                return None
                        
            if cur_image.timestamp >= first_pose.timestamp and cur_image.timestamp <= latest_pose.timestamp:
                break

        while self._img_pose_idx < len(self._imu_pose_list):
            before_pose = self._imu_pose_list[self._img_pose_idx]
            after_pose = self._imu_pose_list[self._img_pose_idx + 1]
            self._img_pose_idx += 1
            if before_pose.timestamp <= cur_image.timestamp and cur_image.timestamp <= after_pose.timestamp:
                cur_image.pose = self.insert_pose(cur_image.timestamp, before_pose, after_pose)
                break

        return cur_image

    def pointsNext(self, cur_pose):
        points = []

        while self._point_idx < self._las.header.point_count:
            if self._las.gps_time[self._point_idx] < cur_pose.timestamp:
                points.append([self._las.x[self._point_idx], self._las.y[self._point_idx], self._las.z[self._point_idx]])
                self._point_idx += 1
            else:
                break
        points = np.array(points)
        return points
    
    def getAllPoints(self):
        points = self._las.xyz
        return points
    
    def readImu(self, path):
        framelist = []
        with open(path) as f:
            data = f.read(ImuFrame.struct_size)
            if not data:
                return None

            imu_data = struct.unpack(ImuFrame.tdpose_format, data)
            framelist.append(ImuFrame(*imu_data)) 

        return True

    def insert_pose(self, timestamp, before_pose, after_pose):
        rate = (timestamp - before_pose.timestamp) / (after_pose.timestamp - before_pose.timestamp)
        t = (after_pose.t - before_pose.t)/rate + before_pose.t
        key_rotations = Rotation.from_quat([before_pose.q, after_pose.q])
        slerp = Slerp(np.array([before_pose.timestamp, after_pose.timestamp]), key_rotations)
        q = slerp(np.array([timestamp]))[0].as_quat()
        cur_pose = Pose(100000, timestamp, *t, *q)
        return cur_pose

    def imuPoseNext(self):
        data = self._imu_pose_file.read(IMUPose.struct_size)
        if not data:
            return None

        pose_data = struct.unpack(IMUPose.tdpose_format, data)
        imu_pose = IMUPose(*pose_data)
        pose = Pose(self._imu_pose_idx, imu_pose.timestamp, *(imu_pose.t), *(imu_pose.q))
        self._imu_pose_idx += 1

        return pose
    
    def poseNext(self):
        data = self._pose_file.read(Pose.struct_size)
        if not data:
            return None

        pose_data = struct.unpack(Pose.tdpose_format, data)
        pose = Pose(*pose_data)

        return pose
    
    def save_point_cloud_to_ply(self, ply_file, point_cloud):
        tmp = np.array(point_cloud)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tmp[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((tmp.shape[0], 3)))
        pcd.normals  = o3d.utility.Vector3dVector(np.zeros((tmp.shape[0], 3)))
        o3d.io.write_point_cloud(ply_file, pcd)

    def filter_point_cloud(self, points, size=0.1):
        tmp = np.array(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tmp[:,:3])
        down_sampled_pcd = pcd.voxel_down_sample(size)
        return down_sampled_pcd.points

    def transform_points_world_to_body(self, points, cur_pose):
        points = np.array(points).transpose()
        r = cur_pose.R.transpose()
        t = cur_pose.T_inv[:3:,3].reshape((3,1))
        points = (r@points + t).transpose()
        return points
    
    def transform_pose_world_to_camera(self, cur_pose):
        camera_pose_in_world = cur_pose.T @ self._T_c2b
        camera_pose_in_new_world = np.linalg.inv(self._T_c2b) @ camera_pose_in_world
        pose_r = camera_pose_in_new_world[:3,:3]
        pose_t = camera_pose_in_new_world[:3,3]
        pose_q =  Rotation.from_matrix(pose_r).as_quat()
        trans_pose = Pose(cur_pose.id, cur_pose.timestamp, pose_t[0], pose_t[1], pose_t[2], pose_q[0], pose_q[1], pose_q[2], pose_q[3])
        return trans_pose
    
    def transform_points_body_to_camera(self, points):
        points = np.array(points).transpose()

        T_b2c = np.linalg.inv(self._T_c2b)
        r = T_b2c[:3,:3]
        t = T_b2c[:3,3].reshape((3,1))

        points = (r@points + t).transpose()
        return points

    def read_ply(self, path):
        pcd = o3d.io.read_point_cloud(path)
        return pcd.points

    def get_depth_image(self, pose, points):
        points = torch.tensor(np.array(points), dtype=torch.float32) 
        ones = torch.ones((points.shape[0], 1), dtype=torch.float32)
        points_homogeneous = torch.cat([points, ones], dim=1)  # shape (N, 4)

        extrinsic = torch.tensor(pose.T_inv, dtype=torch.float32)  # shape (4, 4)
        points_camera = points_homogeneous @ extrinsic.T  # shape (N, 4)
        
        intrinsic = torch.tensor(self._camera.matrix, dtype=torch.float32)  # shape (3, 3)
        
        points_camera = points_camera[:, :3]
        points_image = points_camera @ intrinsic.T  # shape (N, 3)
        
        points_image[:, :2] /= points_image[:, 2:3]
        
        depth = points_camera[:, 2]  # shape (N,)
        
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

    def get_depth_o3d(self, pose, points):
        pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.array(points), dtype=o3d.core.Dtype.Float32))
        intrinsic = o3d.core.Tensor([[self._camera.fx, 0, self._camera.cx],
                                    [0, self._camera.fy, self._camera.cy],
                                    [0, 0, 1]], dtype=o3d.core.Dtype.Float32)

        extrinsic = o3d.core.Tensor(np.array(pose.T_inv, dtype=np.float32))
        depth_image = pcd.project_to_depth_image(width=self._camera.width, 
                                                height=self._camera.height, 
                                                intrinsics=intrinsic,
                                                extrinsics=extrinsic,
                                                depth_scale=1000.0,
                                                depth_max=100.0)
        
        depth_image = np.array(depth_image.cpu())
        depth_map_cm_uint16 = depth_image.astype(np.uint16)
        return depth_map_cm_uint16
    
    def save_whole_points(self):
        points = []
        idx = 0

        while idx < self._las.header.point_count:
            if self._las.gps_time[idx] < self._stop_time:
                points.append([self._las.x[idx], self._las.y[idx], self._las.z[idx]])
                idx += 1
            else:
                break

        self._whole_points = self.transform_points_body_to_camera(points)
        self.save_point_cloud_to_ply(os.path.join(self._save_path, "points3D_density.ply"), self._whole_points)
        points = self.filter_point_cloud(self._whole_points)
        self.save_point_cloud_to_ply(os.path.join(self._save_path, "points3D.ply"), points)

    def save_lidar_frame(self):
        if os.path.exists(self.point_txt):
            os.remove(self.point_txt)
        if os.path.exists(self.pose_lidar_txt):
            os.remove(self.pose_lidar_txt)

        index = 0

        while True:
            cur_pose = self.poseNext()
            if cur_pose is None:
                break
            
            if cur_pose.timestamp > self._stop_time:
                break

            cur_points = self.pointsNext(cur_pose)

            if len(cur_points) == 0:
                continue

            self.save_point_cloud_to_ply(os.path.join(self._save_path, f"point/{cur_pose.timestamp}.ply"), self.transform_points_body_to_camera(self.transform_points_world_to_body(cur_points, cur_pose)))

            with open( self.pose_lidar_txt, 'a') as ofs_pose, \
                open( self.point_txt, 'a') as ofs_point:
                if index == 0:  # 只在第一次写入时添加标题
                    ofs_pose.write("# timestamp tx ty tz qx qy qz qw\n")
                    ofs_point.write("# timestamp filename\n")
                camera_pose = self.transform_pose_world_to_camera(cur_pose)
                ofs_pose.write(f"{camera_pose.timestamp} {camera_pose.posX} {camera_pose.posY} {camera_pose.posZ} "
                        f"{camera_pose.qX} {camera_pose.qY} {camera_pose.qZ} {camera_pose.qW}\n")
                ofs_point.write(f"{cur_pose.timestamp} point/{cur_pose.timestamp}.ply\n")
            index += 1

            print(f"lidar frame id: {index}")
    
    def save_camera_frame(self):
        if os.path.exists(self.depth_txt):
            os.remove(self.depth_txt)
        if os.path.exists(self.rgb_txt):
            os.remove(self.rgb_txt)
        if os.path.exists(self.pose_img_txt):
            os.remove(self.pose_img_txt)

        index = 0

        while True:
            cur_image = self.imgNext()
            
            if cur_image == None:
                break

            if cur_image.timestamp > self._stop_time:
                break

            # 保存图像和点云
            cv2.imwrite(os.path.join(self._save_path, f"rgb/{cur_image.timestamp}.png"), cur_image.img)

            if self._whole_points: 
                depth_img = self.get_depth_o3d(cur_image.pose, self._whole_points)
                cv2.imwrite(os.path.join(self._save_path, f"depth/{cur_image.timestamp}.png"), depth_img)

            with open( self.pose_img_txt, 'a') as ofs_pose, \
                open( self.rgb_txt, 'a') as ofs_rgb, \
                open( self.depth_txt, 'a') as ofs_depth:

                if index == 0:  # 只在第一次写入时添加标题
                    ofs_pose.write("# timestamp tx ty tz qx qy qz qw\n")
                    ofs_rgb.write("# timestamp filename\n")
                    ofs_depth.write("# timestamp filename\n")
                    
                ofs_rgb.write(f"{cur_image.pose.timestamp} rgb/{cur_image.pose.timestamp}.png\n")
                ofs_depth.write(f"{cur_image.timestamp} depth/{cur_image.pose.timestamp}.png\n")

                camera_pose = self.transform_pose_world_to_camera(cur_image.pose)
                ofs_pose.write(f"{camera_pose.timestamp} {camera_pose.posX} {camera_pose.posY} {camera_pose.posZ} "
                        f"{camera_pose.qX} {camera_pose.qY} {camera_pose.qZ} {camera_pose.qW}\n")
            
            index += 1

            print(f"camera frame id: {index}")
        
    def save_camera_info(self):
        if os.path.exists(self.camera_txt):
            os.remove(self.camera_txt)
        cameraInfo = " ".join([str(num) for num in self._camera.getlist()])
        with open(self.camera_txt, 'a') as ofs_camera:
            ofs_camera.write(f"{cameraInfo}\n")

    def run(self):
        # self.save_whole_points()
        self.save_camera_frame()
        self.save_camera_info()
        self.save_lidar_frame()
        return True


 
if __name__ == "__main__":
    imu_pose_path = "/home/rick/Datasets/办公室一圈/SLAM_PRJ_001/2024-04-23_13-46-50_570/IMUPOS.bin"
    pose_path = "/home/rick/Datasets/办公室一圈/SLAM_PRJ_001/2024-04-23_13-46-50_570/optimised_2024-04-23_14-18-25_662.bin"
    las_path = "/home/rick/Datasets/办公室一圈/SLAM_PRJ_001/2024-04-23_13-46-50_570/optimised_2024-04-23_14-18-25_662.las"

    imu_path = "/home/rick/Datasets/办公室一圈/SLAM_PRJ_001/20240312-030641_Lp_Imu.fmimr"
    video_path = "/home/rick/Datasets/办公室一圈/SLAM_PRJ_001/OPTICAL_CAM/optcam_1.h265"
    video_timestamp_path = "/home/rick/Datasets/办公室一圈/SLAM_PRJ_001/OPTICAL_CAM/optcam_1.ts"
    save_path = "/home/rick/Datasets/Custom"

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
    
    T_c2b =  T @ tmp_T
    
    camera = Camera(0, "PINHOLE", 4000, 3000, 2071.184147, 2071.184147, 2051.995468, 1589.171711)
    pp = PreProcess(pose_path, las_path, imu_pose_path, video_path, video_timestamp_path, T_c2b, save_path, camera)
    pp.run()