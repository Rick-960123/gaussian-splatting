import cv2
import numpy as np
import queue
import os
import open3d as o3d
import laspy
import struct
from scipy.spatial.transform import Rotation

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
    

class PreProcess:
    def __init__(self, pose_path, las_path, video_path, video_timestamp_path, T_c2b, save_path):
        self._pose_file = open(pose_path, 'rb')  # Assuming binary read mode for pose file

        self._cap = cv2.VideoCapture(video_path)
        self._las = laspy.read(las_path)

        self._p_idx = 0
        self._imgIdx = 0
        self._lastVideoTime = 0
        self._index = 0
        self.save_path = save_path
        self.image_cache = queue.Queue()
        self._videoTimeList = []
        self._T_c2b = T_c2b
        self._camera_time_error = 18

        with open(video_timestamp_path, 'r') as f:
            for line in f:
                self._videoTimeList.append(float(line.strip()) + self._camera_time_error)

        # Ensure directories exist
        os.makedirs(os.path.join(save_path, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "depth"), exist_ok=True)

        self.pose_txt = os.path.join(self.save_path, "groundtruth.txt")
        self.rgb_txt = os.path.join(self.save_path, "rgb.txt")
        self.points_txt = os.path.join(self.save_path, "depth.txt")

        if os.path.exists(self.points_txt):
            os.remove(self.points_txt)
        if os.path.exists(self.rgb_txt):
            os.remove(self.rgb_txt)
        if os.path.exists(self.pose_txt):
            os.remove(self.pose_txt)

    def imgNext(self, cur_pose):
        cur_image = ImageFrame()
        closest_timestamp = float('inf')

        while True:
            if not self._cap.isOpened():
                return None
        
            if not self.image_cache.empty():
                cur_image = self.image_cache.get()
            else:
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    cur_image.img = frame
                    if self._imgIdx < len(self._videoTimeList):
                        cur_image.timestamp = self._videoTimeList[self._imgIdx]
                        self._imgIdx += 1
                        self._lastVideoTime = cur_image.timestamp
                    else:
                        print(f"\n\nend of video time list {self._lastVideoTime}\n")
                else:
                    print(f"\n\nend of video {self._lastVideoTime}\n")

            if abs(closest_timestamp - cur_pose.timestamp) > abs(cur_image.timestamp - cur_pose.timestamp):
                closest_timestamp = cur_image.timestamp

            if cur_pose.timestamp < cur_image.timestamp:
                if abs(closest_timestamp - cur_pose.timestamp) > 0.5:
                    return None
                
                if abs(closest_timestamp - cur_image.timestamp) > 1e-5:
                    self.image_cache.put(cur_image)
                break
            
        return cur_image

    def pointsNext(self, cur_pose):
        points = []

        while self._p_idx < self._las.header.point_count:
            if self._las.gps_time[self._p_idx] < cur_pose.timestamp:
                points.append([self._las.x[self._p_idx], self._las.y[self._p_idx], self._las.z[self._p_idx]])
                self._p_idx += 1
            else:
                break
        points = np.array(points)
        return points
    
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
        # 保存为PLY文件
        o3d.io.write_point_cloud(ply_file, pcd)

    def filter_point_cloud(self, points):
        tmp = np.array(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tmp[:,:3])
        down_sampled_pcd = pcd.voxel_down_sample(0.1)
        return down_sampled_pcd.points

    def transform_points_world_to_body(self, points, cur_pose):
        points = np.array(points).transpose()
        q = Rotation.from_quat(np.array([cur_pose.qX, cur_pose.qY, cur_pose.qZ, cur_pose.qW]))
        r = q.as_matrix().transpose()
        t = np.array([[cur_pose.posX, cur_pose.posY, cur_pose.posZ]]).transpose()
        t = (-1 * r @ t).transpose()

        t = np.array(([t[0] for i in range(points.shape[1])])).transpose()

        points = (r@points + t).transpose()
        return points
    
    def transform_pose_world_to_camera(self, cur_pose):
        pose_q = Rotation.from_quat(np.array([cur_pose.qX, cur_pose.qY, cur_pose.qZ, cur_pose.qW]))
        pose_r = pose_q.as_matrix()
        pose_t = np.array([cur_pose.posX, cur_pose.posY, cur_pose.posZ]).transpose()
        pose_T = np.ones((4,4))
        pose_T[:3,:3] = pose_r
        pose_T[:3,3] = pose_t

        camera_pose_in_world = pose_T @ self._T_c2b
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
        t = T_b2c[:3,3]

        t = np.array([t.transpose() for i in range(points.shape[1])]).transpose()

        points = (r@points + t).transpose()
        return points

    def read_ply(self, path):
        pcd = o3d.io.read_point_cloud(path)
        return pcd.points

    def run(self):
        whole_points = np.array([])

        while True:
            cur_pose = self.poseNext()
            if cur_pose is None:
                break

            cur_points = self.pointsNext(cur_pose)

            if len(cur_points) == 0:
                continue

            if len(whole_points) == 0:
                whole_points = cur_points
            else:
                whole_points = np.vstack((whole_points, cur_points))
            
            whole_points = self.filter_point_cloud(whole_points)

            cur_image = self.imgNext(cur_pose)
            
            if cur_image == None:
                continue
            
            # 保存图像和点云
            cv2.imwrite(os.path.join(self.save_path, f"rgb/{cur_pose.timestamp}.png"), cur_image.img)

            self.save_point_cloud_to_ply(os.path.join(self.save_path, f"depth/{cur_pose.timestamp}.ply"), self.transform_points_body_to_camera(self.transform_points_world_to_body(cur_points, cur_pose)))

            with open( self.pose_txt, 'a') as ofs_camera, \
                open( self.rgb_txt, 'a') as ofs_rgb, \
                open( self.points_txt, 'a') as ofs_points:

                if self._index == 0:  # 只在第一次写入时添加标题
                    ofs_camera.write("# timestamp tx ty tz qx qy qz qw\n")
                    ofs_rgb.write("#  timestamp filename\n")
                    ofs_points.write("#  timestamp filename\n")

                ofs_rgb.write(f"{cur_pose.timestamp} rgb/{cur_pose.timestamp}.png\n")
                ofs_points.write(f"{cur_pose.timestamp} depth/{cur_pose.timestamp}.ply\n")
                
                camera_pose = self.transform_pose_world_to_camera(cur_pose)
                ofs_camera.write(f"{camera_pose.timestamp} {camera_pose.posX} {camera_pose.posY} {camera_pose.posZ} "
                        f"{camera_pose.qX} {camera_pose.qY} {camera_pose.qZ} {camera_pose.qW}\n")
                
            self._index += 1

            print(self._index)
            if self._index == 200:
                break
        
        whole_points = self.transform_points_body_to_camera(whole_points)
        self.save_point_cloud_to_ply(os.path.join(self.save_path, "points3D.ply"), whole_points)

        return True

 
if __name__ == "__main__":
    pose_path = "/home/rick/Datasets/SN_00250/SLAM_PRJ_001/2024-04-23_13-46-50_570/optimised_2024-04-23_14-18-25_662.bin"
    las_path = "/home/rick/Datasets/SN_00250/SLAM_PRJ_001/2024-04-23_13-46-50_570/optimised_2024-04-23_14-18-25_662.las"

    video_path = "/home/rick/Datasets/SN_00250/SLAM_PRJ_001/OPTICAL_CAM/optcam_1.h265"
    video_timestamp_path = "/home/rick/Datasets/SN_00250/SLAM_PRJ_001/OPTICAL_CAM/optcam_1.ts"
    save_path = "/home/rick/Datasets/Custom"
    cameras_params = "0 PINHOLE 4000 3000 2071.184147 2071.184147 2051.995468 1589.171711"

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
    
    pp = PreProcess(pose_path, las_path, video_path, video_timestamp_path, camera_pose, save_path)
    pp.run()