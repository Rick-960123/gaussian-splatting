import cv2
import numpy as np
import os
import open3d as o3d
import laspy
import torch
import struct
import yaml
from scipy.spatial.transform import Rotation, Slerp
from rawDataReader import *

class PreProcess:
    def __init__(self, pose_path, las_path, imu_pose_path, video_path, video_timestamp_path, yaml_path, save_path, duration=200):
        self._raw_data_reader = RawDataReader(pose_path, las_path, imu_pose_path, video_path, video_timestamp_path, yaml_path)
        self._duration = duration
        self._save_path = save_path
        self._whole_points = self._raw_data_reader._las.xyz

        # Ensure directories exist
        self.rgb_path = os.path.join(self._save_path, "rgb")
        self.depth_path = os.path.join(self._save_path, "depth")
        self.point_path = os.path.join(self._save_path, "point")

        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.depth_path, exist_ok=True)
        os.makedirs(self.point_path, exist_ok=True)

        self.pose_img_txt = os.path.join(self._save_path, "groundtruth.txt")
        self.pose_lidar_txt = os.path.join(self._save_path, "groundtruth_lidar.txt")
        self.rgb_txt = os.path.join(self._save_path, "rgb.txt")
        self.depth_txt = os.path.join(self._save_path, "depth.txt")
        self.point_txt = os.path.join(self._save_path, "point.txt")
        self.camera_txt =  os.path.join(self._save_path, "cameras.txt")
    
    def save_whole_points(self):
        CommonTools.savePointCloudToPly(os.path.join(self._save_path, "points3D_density.ply"), self._whole_points)
        points = CommonTools.filterPointCloud(self._whole_points)
        CommonTools.savePointCloudToPly(os.path.join(self._save_path, "points3D.ply"), points)

    def save_lidar_frame(self):
        if os.path.exists(self.point_txt):
            os.remove(self.point_txt)
        if os.path.exists(self.pose_lidar_txt):
            os.remove(self.pose_lidar_txt)

        index = 0
        first_timestamp = 0

        while True:
            cur_lidar_frame = self._raw_data_reader.lidarNext()
            if cur_lidar_frame is None:
                break
            if first_timestamp == 0:
                first_timestamp = cur_lidar_frame.timestamp
            if cur_lidar_frame.timestamp > first_timestamp + self._duration:
                break
            if len(cur_lidar_frame.points) == 0:
                continue
            
            CommonTools.savePointCloudToPly(os.path.join(self.point_path, f"{cur_lidar_frame.timestamp}.ply"), CommonTools.getPointsInBody(cur_lidar_frame.points, cur_lidar_frame.pose))

            with open( self.pose_lidar_txt, 'a') as ofs_pose, \
                open( self.point_txt, 'a') as ofs_point:
                if index == 0:  # 只在第一次写入时添加标题
                    ofs_pose.write("# timestamp tx ty tz qx qy qz qw\n")
                    ofs_point.write("# timestamp filename\n")
                camera_pose = cur_lidar_frame.pose
                ofs_pose.write(f"{camera_pose.timestamp} {camera_pose.posX} {camera_pose.posY} {camera_pose.posZ} "
                        f"{camera_pose.qX} {camera_pose.qY} {camera_pose.qZ} {camera_pose.qW}\n")
                ofs_point.write(f"{cur_lidar_frame.timestamp} point/{cur_lidar_frame.timestamp}.ply\n")
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
        first_timestamp = 0
        while True:
            cur_image = self._raw_data_reader.imgNext()
            
            if cur_image == None:
                break

            if first_timestamp == 0:
                first_timestamp = cur_image.timestamp

            if cur_image.timestamp > first_timestamp + self._duration:
                break

            # 保存图像和点云
            cv2.imwrite(os.path.join(self.rgb_path, f"{cur_image.timestamp}.png"), cur_image.img)

            camera_pose = self._raw_data_reader._camera.getCameraPose(cur_image.pose)
            if len(self._whole_points) > 0: 
                depth_img = CommonTools.getDepthO3d(camera_pose, self._whole_points, self._raw_data_reader._camera)
                cv2.imwrite(os.path.join(self.depth_path, f"{cur_image.timestamp}.png"), depth_img)

            with open( self.pose_img_txt, 'a') as ofs_pose, \
                open( self.rgb_txt, 'a') as ofs_rgb, \
                open( self.depth_txt, 'a') as ofs_depth:

                if index == 0:  # 只在第一次写入时添加标题
                    ofs_pose.write("# timestamp tx ty tz qx qy qz qw\n")
                    ofs_rgb.write("# timestamp filename\n")
                    ofs_depth.write("# timestamp filename\n")
                    
                ofs_rgb.write(f"{cur_image.pose.timestamp} rgb/{cur_image.pose.timestamp}.png\n")
                ofs_depth.write(f"{cur_image.timestamp} depth/{cur_image.pose.timestamp}.png\n")
                
                ofs_pose.write(f"{camera_pose.timestamp} {camera_pose.posX} {camera_pose.posY} {camera_pose.posZ} "
                        f"{camera_pose.qX} {camera_pose.qY} {camera_pose.qZ} {camera_pose.qW}\n")
            
            index += 1

            print(f"camera frame id: {index}")
        
    def save_camera_info(self):
        if os.path.exists(self.camera_txt):
            os.remove(self.camera_txt)
        cameraInfo = " ".join([str(num) for num in self._raw_data_reader._camera.getList()])
        with open(self.camera_txt, 'a') as ofs_camera:
            ofs_camera.write(f"{cameraInfo}\n")

    def run(self):
        self.save_lidar_frame()
        self.save_camera_info()
        self.save_camera_frame()
        self.save_whole_points()
        return True

 
if __name__ == "__main__":

    base_path = "/home/rick/Datasets/slam2000-雪乡情-正走"
    save_path = os.path.join(base_path, "tum")

    imu_pose_path = os.path.join(base_path, "V5-2024-11-05_15-28-23_808/IMUPOS.bin")
    pose_path = os.path.join(base_path, "V5-2024-11-05_15-28-23_808/optimised_2024-11-05_15-30-42_602.bin")
    las_path = os.path.join(base_path, "V5-2024-11-05_15-28-23_808/optimised_2024-11-05_15-30-42_602.las")

    imu_path = os.path.join(base_path, "SLAM_PRJ_001/20240312-030641_Lp_Imu.fmimr")
    video_path = os.path.join(base_path, "SLAM_PRJ_001/OPTICAL_CAM/optcam_1.h265")
    video_timestamp_path = os.path.join(base_path, "SLAM_PRJ_001/OPTICAL_CAM/optcam_1.ts")
    
    yaml_path = os.path.join(base_path, "SLAM_PRJ_001/slam_calib.yaml")
    
    pp = PreProcess(pose_path, las_path, imu_pose_path, video_path, video_timestamp_path, yaml_path, save_path, 20)
    pp.run()