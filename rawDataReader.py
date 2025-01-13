import cv2
import numpy as np
import os
import open3d as o3d
import laspy
import torch
import struct
import yaml
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

        self.R_sci = Rotation.from_euler("xyz", np.array([self.roll, self.pitch, self.yaw]))
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
    def __init__(self, yaml_path, crop_image=False):
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
            camera_param = config['parameters']["param"]['camera_instrinsic_parameters_opt_cam']
            
            # 转换矩阵
            T_camera2body = np.array(config['parameters']["param"]['T_imu2optcam_refine']).reshape(4, 4)

            tmp_T = np.array([1,0,0,0,
                        0,-1,0,0,
                        0,0,-1,0,
                        0,0,0,1]).reshape((4,4)) 
            T_camera2body = T_camera2body @ tmp_T
    
            self.T_c2b =  T_camera2body

            self.id = 1
            self.model = "PINHOLE"
            self.width_d = camera_param[0]
            self.height_d = camera_param[1]
            self.cx_d = camera_param[3]
            self.cy_d = camera_param[4]
            self.fx_d = camera_param[5]
            self.fy_d = camera_param[5]
            self.distortion = camera_param[6:]

        self._crop_image = crop_image
        if self._crop_image:
            self._crop_right = 150
            self._crop_left = 0
            self._crop_top = 150
            self._crop_bottom = 0
        else:
            self._crop_right = 0
            self._crop_left = 0
            self._crop_top = 0
            self._crop_bottom = 0

        self.height = self.height_d - self._crop_top - self._crop_bottom
        self.width = self.width_d - self._crop_left - self._crop_right
        self.cx = self.width_d / 2 - self._crop_left
        self.cy = self.height_d / 2 - self._crop_top

        self.fx = self.fx_d
        self.fy = self.fy_d

        self.utc_gps_time_diff = 18.0
        self.initUndistortMap()

    def getCameraPose(self, cur_pose):
        camera_pose_in_world = cur_pose.T @ self.T_c2b
        pose_r = camera_pose_in_world[:3,:3]
        pose_t = camera_pose_in_world[:3,3]
        pose_q =  Rotation.from_matrix(pose_r).as_quat()
        return Pose(cur_pose.id, cur_pose.timestamp, pose_t[0], pose_t[1], pose_t[2], pose_q[0], pose_q[1], pose_q[2], pose_q[3])
    
    def calibrateImage(self, image):
        """
        图像去畸变
        Args:
            image: OpenCV格式的输入图像
        Returns:
            undistorted: 去畸变后的图像
        """
        new_image = cv2.remap(image, self.undistort_map[:,:,0], self.undistort_map[:,:,1], cv2.INTER_LINEAR)
        if new_image.shape[0] != self.height_d or new_image.shape[1] != self.width_d:
            assert False, "image shape error"
        if self._crop_image:
            new_image = new_image[self._crop_top:self.height_d-self._crop_bottom, 
                                          self._crop_left:self.width_d-self._crop_right]
        return new_image
    
    def initUndistortMap(self):
        self.undistort_map = np.zeros((self.height_d, self.width_d, 2), dtype=np.float32)
        
        center = True
        # 获取图像尺寸
        rows, cols = self.height_d, self.width_d
        
        # 初始化参数
        x0 = self.cx_d
        y0 = self.cy_d
        k1, k2, k3, k4 = self.distortion[:4]
        p1, p2 = self.distortion[4:6]
        alpha, beta = self.distortion[6:8]
        k5, k6 = 0, 0  # 额外的畸变参数设为0
        
        # 计算中心偏移
        dx0 = x0 - 0.5 * (self.width_d - 1)
        dy0 = y0 - 0.5 * (self.height_d - 1)
        
        if not center:
            dx0 = dy0 = 0
        
        # 对每个像素进行处理
        for ir in range(rows):
            for ic in range(cols):
                r = rows - 1 - ir  # 图像坐标系转换
                x = float(ic)
                y = float(r)
                
                # 添加偏移
                x += dx0
                y += dy0
                ix, iy = x, y
                
                # 迭代求解去畸变映射
                dx = [0, 0]
                dy = [0, 0]
                
                for k in range(100):
                    cx = x - x0
                    cy = y - y0
                    cxy = cx * cy
                    cx2 = cx * cx
                    cy2 = cy * cy
                    
                    r2 = cx2 + cy2
                    r4 = r2 * r2
                    r6 = r2 * r4
                    r8 = r4 * r4
                    r10 = r4 * r6
                    r12 = r6 * r6
                    
                    t0 = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12
                    t1 = r2 + 2 * cx2
                    t2 = r2 + 2 * cy2
                    
                    dx[0] = dx[1]
                    dy[0] = dy[1]
                    
                    dx[1] = cx * t0 + p1 * t1 + 2 * p2 * cxy + alpha * cx + beta * cy
                    dy[1] = cy * t0 + p2 * t2 + 2 * p1 * cxy
                    
                    x = ix - dx[1]
                    y = iy - dy[1]
                    
                    if abs(dx[1] - dx[0]) < 0.05 and abs(dy[1] - dy[0]) < 0.05:
                        break
                
                # 坐标系转换回来
                r = rows - 1 - r
                y = rows - 1 - y
                
                # 检查边界条件
                if x >= 0 and x < cols and y >= 0 and y < rows:
                    self.undistort_map[r, ic] = [x, y]

    def getList(self):
        return [self.id,
                self.model,
                self.width,
                self.height,
                self.fx,
                self.fy,
                self.cx,
                self.cy]
    
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

class CommonTools:
    @classmethod
    def insertPose(self, timestamp, before_pose, after_pose):
        rate = (timestamp - before_pose.timestamp) / (after_pose.timestamp - before_pose.timestamp)
        t = (after_pose.t - before_pose.t) * rate + before_pose.t
        key_rotations = Rotation.from_quat([before_pose.q, after_pose.q])
        slerp = Slerp(np.array([before_pose.timestamp, after_pose.timestamp]), key_rotations)
        q = slerp(np.array([timestamp]))[0].as_quat()
        cur_pose = Pose(100000, timestamp, *t, *q)
        return cur_pose
    
    @classmethod
    def savePointCloudToPly(self, ply_file, point_cloud):
        tmp = np.array(point_cloud)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tmp[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((tmp.shape[0], 3)))
        pcd.normals  = o3d.utility.Vector3dVector(np.zeros((tmp.shape[0], 3)))
        o3d.io.write_point_cloud(ply_file, pcd)
    
    @classmethod
    def filterPointCloud(self, points, size=0.1):
        tmp = np.array(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tmp[:,:3])
        down_sampled_pcd = pcd.voxel_down_sample(size)
        return down_sampled_pcd.points

    @classmethod
    def getPointsInBody(self, points, cur_pose):
        points = np.array(points).transpose()
        r = cur_pose.R.transpose()
        t = cur_pose.T_inv[:3:,3].reshape((3,1))
        points = (r@points + t).transpose()
        return points
    
    @classmethod
    def getPointsInCamera(self, points, T_c2b):
        points = np.array(points).transpose()

        T_b2c = np.linalg.inv(T_c2b)
        r = T_b2c[:3,:3]
        t = T_b2c[:3,3].reshape((3,1))

        points = (r@points + t).transpose()
        return points

    @classmethod
    def getDepthImage(self, camera_pose, points, camera):
        points = torch.tensor(np.array(points), dtype=torch.float32) 
        ones = torch.ones((points.shape[0], 1), dtype=torch.float32)
        points_homogeneous = torch.cat([points, ones], dim=1)  # shape (N, 4)

        extrinsic = torch.tensor(camera_pose.T_inv, dtype=torch.float32)  # shape (4, 4)
        points_camera = points_homogeneous @ extrinsic.T  # shape (N, 4)
        
        intrinsic = torch.tensor(camera.matrix, dtype=torch.float32)  # shape (3, 3)
        
        points_camera = points_camera[:, :3]
        points_image = points_camera @ intrinsic.T  # shape (N, 3)
        
        points_image[:, :2] /= points_image[:, 2:3]
        
        depth = points_camera[:, 2]  # shape (N,)
        
        depth_image = torch.full((camera.height, camera.width), float('inf'), dtype=torch.float32)
        
        x_coords = points_image[:, 0].long()
        y_coords = points_image[:, 1].long()
        
        valid_mask = (x_coords >= 0) & (x_coords < camera.width) & (y_coords >= 0) & (y_coords < camera.height)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        depth = depth[valid_mask]
        
        depth_image[y_coords, x_coords] = torch.min(depth_image[y_coords, x_coords], depth)
        
        depth_image[depth_image == float('inf')] = 0
        depth_image = (depth_image * 1000).clamp(10, 100000)
        
        depth_image_numpy = depth_image.numpy()
        return depth_image_numpy.astype(np.uint16)

    @classmethod
    def getDepthO3d(self, camera_pose, points, camera):
        pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.array(points), dtype=o3d.core.Dtype.Float32))
        intrinsic = o3d.core.Tensor([[camera.fx, 0, camera.cx],
                                    [0, camera.fy, camera.cy],
                                    [0, 0, 1]], dtype=o3d.core.Dtype.Float32)

        extrinsic = o3d.core.Tensor(np.array(camera_pose.T_inv, dtype=np.float32))
        depth_image = pcd.project_to_depth_image(width=camera.width, 
                                                height=camera.height, 
                                                intrinsics=intrinsic,
                                                extrinsics=extrinsic,
                                                depth_scale=1000.0,
                                                depth_max=100.0)
        
        depth_image = np.array(depth_image.cpu())
        depth_map_cm_uint16 = depth_image.astype(np.uint16)
        return depth_map_cm_uint16
    
    @classmethod
    def readPly(self, path):
        pcd = o3d.io.read_point_cloud(path)
        return pcd.points

class RawDataReader:
    def __init__(self, pose_path, las_path, imu_pose_path, video_path, video_timestamp_path, yaml_path, crop_image=False):
        self._point_idx = 0
        self._img_idx = 0
        self._pose_idx = 0
        self._imu_pose_idx = 0
        self._img_pose_idx = 0
        self._imu_frame_idx = 0
        self._last_video_time = 0

        self._common_tools = CommonTools()
        self._camera = Camera(yaml_path, crop_image)
        self._video_cap = cv2.VideoCapture(video_path)

        self._las = self.readLas(las_path)
        self._video_time_list = self.readVideoTime(video_timestamp_path)
        self._pose_list = self.readPose(pose_path)
        self._imu_pose_list = self.readImuPose(imu_pose_path)

    def readLas(self, path):
        las = laspy.read(path)
        return las
    
    def readPose(self, path):
        pose_list = []
        with open(path, 'rb') as pose_file:
            while True:
                data = pose_file.read(Pose.struct_size)
                if not data:
                    break
                pose_data = struct.unpack(Pose.tdpose_format, data)
                pose = Pose(*pose_data)
                pose_list.append(pose)
        return pose_list
    
    def readImuPose(self, path):
        imu_pose_list = []
        with open(path, 'rb') as imu_pose_file:
            while True:
                data = imu_pose_file.read(IMUPose.struct_size)
                if not data:
                    break

                pose_data = struct.unpack(IMUPose.tdpose_format, data)
                imu_pose = IMUPose(*pose_data)
                pose = Pose(self._imu_pose_idx, imu_pose.timestamp, *(imu_pose.t), *(imu_pose.q))
                self._imu_pose_idx += 1
                if pose is None:
                    break
                imu_pose_list.append(pose)
        return imu_pose_list

    def readVideoTime(self, path):
        video_time_list = []
        with open(path, 'r') as f:
            for line in f:
                #UTC时间 转换为 GPS时间
                video_time_list.append(float(line.strip()) + self._camera.utc_gps_time_diff)
        return video_time_list

    def lidarNext(self):
        if self._pose_idx >= len(self._pose_list):
            return None
        
        points = []
        pose = self._pose_list[self._pose_idx]
        self._pose_idx += 1
        lidar_frame = LidarFrame(pose.timestamp, points, pose)
        while self._point_idx < self._las.header.point_count:
            if self._las.gps_time[self._point_idx] < pose.timestamp:
                points.append([self._las.x[self._point_idx], self._las.y[self._point_idx], self._las.z[self._point_idx]])
                self._point_idx += 1
            else:
                break
        lidar_frame.points = np.array(points)
        return lidar_frame

    def imgNext(self):
        cur_image = ImageFrame()
        first_pose = self._imu_pose_list[0]
        latest_pose = self._imu_pose_list[-1]

        while True:
            if not self._video_cap.isOpened():
                return None
            
            ret, frame = self._video_cap.read()
            if ret and frame is not None:
                cur_image.img = frame
                if self._img_idx < len(self._video_time_list):
                    cur_image.timestamp = self._video_time_list[self._img_idx]
                    self._img_idx += 1
                    self._last_video_time = cur_image.timestamp
                else:
                    print(f"\nend of video time list {self._last_video_time}\n")
                    return None
            else:
                print(f"\nend of video {self._last_video_time}\n")
                return None
                        
            if cur_image.timestamp >= first_pose.timestamp and cur_image.timestamp <= latest_pose.timestamp:
                break

        while self._img_pose_idx < len(self._imu_pose_list):
            before_pose = self._imu_pose_list[self._img_pose_idx]
            after_pose = self._imu_pose_list[self._img_pose_idx + 1]
            self._img_pose_idx += 1
            if before_pose.timestamp <= cur_image.timestamp and cur_image.timestamp <= after_pose.timestamp:
                cur_image.pose = self._common_tools.insertPose(cur_image.timestamp, before_pose, after_pose)
                break
        
        cur_image.img = self._camera.calibrateImage(cur_image.img)
        return cur_image
    
    def test(self):
        while True:
            img_data = self.imgNext()
            if img_data is None:
                break
            print(img_data.timestamp)
        while True:
            lidar_data = self.lidarNext()
            if lidar_data is None:
                break
            print(lidar_data.timestamp)
    
if __name__ == "__main__":

    base_path = "/home/rick/Datasets/slam2000-雪乡情-正走"
    save_path = os.path.join(base_path)

    
    pose_path = os.path.join(base_path, "V5-2024-11-05_15-28-23_808/optimised_2024-11-05_15-30-42_602.bin")
    las_path = os.path.join(base_path, "V5-2024-11-05_15-28-23_808/optimised_2024-11-05_15-30-42_602.las")
    imu_pose_path = os.path.join(base_path, "V5-2024-11-05_15-28-23_808/IMUPOS.bin")

    video_path = os.path.join(base_path, "SLAM_PRJ_001/OPTICAL_CAM/optcam_1.h265")
    video_timestamp_path = os.path.join(base_path, "SLAM_PRJ_001/OPTICAL_CAM/optcam_1.ts")
    yaml_path = os.path.join(base_path, "SLAM_PRJ_001/slam_calib.yaml")
    
    reader = RawDataReader(pose_path, las_path, imu_pose_path, video_path, video_timestamp_path, yaml_path)
    reader.test()