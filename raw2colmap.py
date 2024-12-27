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
    def __init__(self, yaml_path, crop_image=True):
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
            camera_param = config['parameters']["param"]['camera_instrinsic_parameters_opt_cam']
            
            # 转换矩阵
            T_imu2clrcam = np.array(config['parameters']["param"]['T_imu2optcam_refine']).reshape(4, 4)
            tmp_T = np.array([1,0,0,0,
                         0,-1,0,0,
                         0,0,-1,0,
                         0,0,0,1]).reshape((4,4)) 
    
            self.T_c2b =  T_imu2clrcam @ tmp_T

            self.id = 1
            self.model = "PINHOLE"
            self.width_d = camera_param[0]
            self.height_d = camera_param[1]
            self.cx_d = camera_param[3]
            self.cy_d = camera_param[4]
            self.fx_d = camera_param[5]
            self.fy_d = camera_param[5]
            self.distortion = camera_param[6:]

        if crop_image:
            self.crop_top = 250
            self.crop_right = 250
        else:
            self.crop_top = 0
            self.crop_right = 0

        self.width = self.width_d - self.crop_right
        self.height = self.height_d - self.crop_top
        self.cx = self.width / 2 - self.crop_right
        self.cy = self.height / 2 - self.crop_top
        self.fx = self.fx_d
        self.fy = self.fy_d
        self.init_undistort_map()

    def calibrate_image(self, image):
        """
        图像去畸变
        Args:
            image: OpenCV格式的输入图像
        Returns:
            undistorted: 去畸变后的图像
        """
        return cv2.remap(image, self.undistort_map[:,:,0], self.undistort_map[:,:,1], cv2.INTER_LINEAR)
    
    def calibrate_and_crop_image(self, image):
        undistorted = self.calibrate_image(image)
        # 裁剪参数
        # 裁剪图像
        height, width = undistorted.shape[:2]
        cropped_image = undistorted[self.crop_top:, :(width-self.crop_right)]
        return cropped_image
    
    def init_undistort_map(self):
        self.undistort_map = np.zeros((self.height, self.width, 2), dtype=np.float32)
        
        center = True
        # 获取图像尺寸
        rows, cols = self.height, self.width
        
        # 初始化参数
        x0 = self.cx_d
        y0 = self.cy_d
        k1, k2, k3, k4 = self.distortion[:4]
        p1, p2 = self.distortion[4:6]
        alpha, beta = self.distortion[6:8]
        k5, k6 = 0, 0  # 额外的畸变参数设为0
        
        # 计算中心偏移
        dx0 = x0 - 0.5 * (self.width - 1)
        dy0 = y0 - 0.5 * (self.height - 1)
        
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

    def getlist(self):
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


class PreProcess:
    def __init__(self, pose_path, las_path, imu_pose_path, video_path, video_timestamp_path, save_path, camera, duration=200):
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
        self._T_c2b = camera.T_c2b
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

        self._colmap_dir = os.path.join(self._save_path, "colmap")
        self._sparse_dir = os.path.join(self._colmap_dir, "sparse/0")
        self._images_dir = os.path.join(self._colmap_dir, "images")
        os.makedirs(self._sparse_dir, exist_ok=True)
        os.makedirs(self._images_dir, exist_ok=True)

        # 定义COLMAP文件路径
        self._colmap_cameras = os.path.join(self._sparse_dir, "cameras.bin")
        self._colmap_images = os.path.join(self._sparse_dir, "images.bin")
        self._colmap_points3D = os.path.join(self._sparse_dir, "points3D.bin")
        self._colmap_points3D_density = os.path.join(self._sparse_dir, "points3D_density.ply")

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
        
        cur_image.img = self._camera.calibrate_and_crop_image(cur_image.img)
        return cur_image
    
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
        t = (after_pose.t - before_pose.t) * rate + before_pose.t
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
        self.save_point_cloud_to_ply(self._colmap_points3D_density, self._whole_points)
        points = self.filter_point_cloud(self._whole_points)
        with open(self._colmap_points3D, "wb") as fid_pts:
            fid_pts.seek(0)
            fid_pts.write(struct.pack('Q', 0))  # 先写入0，后面再更新
            for i, point in enumerate(points):
                point_id = i + 1
                fid_pts.write(struct.pack('Q', point_id))
                fid_pts.write(struct.pack('ddd', *point))
                fid_pts.write(struct.pack('BBB', 255, 255, 255))
                fid_pts.write(struct.pack('d', 0.0))
                fid_pts.write(struct.pack('Q', 0))

            fid_pts.seek(0)
            fid_pts.write(struct.pack('Q', point_id))

    def save_camera_frame(self):
        with open(self._colmap_images, "wb") as fid:
            # 预留图像数量的位置
            fid.seek(0)
            fid.write(struct.pack('Q', 0))  # 先写入0，后面再更新
            
            image_count = 0
            while True:
                cur_image = self.imgNext()
                
                if cur_image is None or cur_image.timestamp > self._stop_time:
                    break

                # 保存图像
                image_name = f"{cur_image.timestamp}.png"
                cv2.imwrite(os.path.join(self._images_dir, image_name), cur_image.img)

                camera_pose = self.transform_pose_world_to_camera(cur_image.pose)
                
                # 写入图像参数
                image_count += 1
                fid.write(struct.pack('i', image_count))
                fid.write(struct.pack('dddd', 
                    camera_pose.qW, camera_pose.qX, 
                    camera_pose.qY, camera_pose.qZ))
                fid.write(struct.pack('ddd', 
                    camera_pose.posX, camera_pose.posY, camera_pose.posZ))
                fid.write(struct.pack('i', 1))  # camera_id
                # 写入图像名称
                fid.write(image_name.encode('utf-8'))
                fid.write(struct.pack('B', 0))
                fid.write(struct.pack('Q', 0))                
                print(f"Saved camera frame: {image_count}")

            # 更新图像总数
            fid.seek(0)
            fid.write(struct.pack('Q', image_count))
        
    def save_camera_info(self):
        with open(self._colmap_cameras, 'wb') as fid:
            fid.write(struct.pack('Q', 1))
        
            # PINHOLE相机模型ID为1
            model_id = 1
            
            # 相机参数: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4
            params = np.array([
                self._camera.fx, self._camera.fy,
                self._camera.cx, self._camera.cy,
                0,0,0,0,0,0 # k1, k2, p1, p2 k3, k4
            ])
            
            # 写入相机参数
            fid.write(struct.pack('I', 1))  # camera_id
            fid.write(struct.pack('i', model_id))
            fid.write(struct.pack('Q', self._camera.width))
            fid.write(struct.pack('Q', self._camera.height))
            fid.write(struct.pack(f'{len(params)}d', *params))

    def run(self):
        self.save_camera_frame()
        self.save_camera_info()
        self.save_whole_points()
        return True


 
if __name__ == "__main__":

    base_path = "/home/rick/Datasets/slam2000-雪乡情-正走"
    save_path = os.path.join(base_path)

    imu_pose_path = os.path.join(base_path, "V5-2024-11-05_15-28-23_808/IMUPOS.bin")
    pose_path = os.path.join(base_path, "V5-2024-11-05_15-28-23_808/optimised_2024-11-05_15-30-42_602.bin")
    las_path = os.path.join(base_path, "V5-2024-11-05_15-28-23_808/optimised_2024-11-05_15-30-42_602.las")

    imu_path = os.path.join(base_path, "SLAM_PRJ_001/20240312-030641_Lp_Imu.fmimr")
    video_path = os.path.join(base_path, "SLAM_PRJ_001/OPTICAL_CAM/optcam_1.h265")
    video_timestamp_path = os.path.join(base_path, "SLAM_PRJ_001/OPTICAL_CAM/optcam_1.ts")
    
    yaml_path = os.path.join(base_path, "SLAM_PRJ_001/slam_calib.yaml")
    
    camera = Camera(yaml_path)
    pp = PreProcess(pose_path, las_path, imu_pose_path, video_path, video_timestamp_path, save_path, camera)
    pp.run()