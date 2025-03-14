import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation, Slerp
from rawDataReader import *


class PreProcess:
    def __init__(self, raw_data_reader, save_path, save_depth=True, duration=200):
        self._raw_data_reader = raw_data_reader
        self._duration = duration
        self._whole_points = self._raw_data_reader.las.xyz

        self._colmap_dir = save_path
        self._sparse_dir = os.path.join(self._colmap_dir, "sparse/0")
        self._images_dir = os.path.join(self._colmap_dir, "images")
        self._masks_dir = os.path.join(self._colmap_dir, "masks")
        self._depths_dir = os.path.join(self._colmap_dir, "depths")
        
        os.makedirs(self._sparse_dir, exist_ok=True)
        os.makedirs(self._images_dir, exist_ok=True)
        os.makedirs(self._masks_dir, exist_ok=True)
        os.makedirs(self._depths_dir, exist_ok=True)

        # 定义COLMAP文件路径
        self._colmap_cameras = os.path.join(self._sparse_dir, "cameras.txt")
        self._colmap_images = os.path.join(self._sparse_dir, "images.txt")
        self._colmap_points3D = os.path.join(self._sparse_dir, "points3D.txt")
        self._colmap_points3D_density = os.path.join(self._sparse_dir, "points3D_density.ply")

        if not self._raw_data_reader.video_parms.crop_image:
            self.mask = np.ones((self._raw_data_reader.camera.height, self._raw_data_reader.camera.width), dtype=np.uint8) * 255
            self.mask[:, -150:] = 0
            self.mask[:150, :] = 0
        else:
            self.mask = None

        self.save_depth = save_depth
    def save_whole_points(self):
        CommonTools.savePointCloudToPly(self._colmap_points3D_density, self._whole_points)
        points = CommonTools.filterPointCloud(self._whole_points)
        
        with open(self._colmap_points3D, "w") as fid:
            # 写入文件头注释
            fid.write("# 3D point list with one line of data per point:\n")
            fid.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            
            # 写入每个3D点
            for i, point in enumerate(points):
                point_id = i + 1
                # point_id, x, y, z, r, g, b, error, 后面是track信息(这里为空)
                fid.write(f"{point_id} {point[0]} {point[1]} {point[2]} 255 255 255 0.0\n")

    def save_camera_info(self):
        with open(self._colmap_cameras, "w") as fid:
            # 写入文件头注释
            fid.write("# Camera list with one line of data per camera:\n")
            fid.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            
            # PINHOLE相机模型
            params = [
                self._raw_data_reader.camera.fx, 
                self._raw_data_reader.camera.fy,
                self._raw_data_reader.camera.cx, 
                self._raw_data_reader.camera.cy
            ]
            
            # 写入相机参数
            fid.write(f"1 PINHOLE {self._raw_data_reader.camera.width} {self._raw_data_reader.camera.height} "
                     f"{params[0]} {params[1]} {params[2]} {params[3]}\n")

    def save_camera_frame(self):
        with open(self._colmap_images, "w") as fid:
            # 写入文件头注释
            fid.write("# Image list with two lines of data per image:\n")
            fid.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            fid.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            image_count = 0
            first_timestamp = 0

            while True:
                cur_image = self._raw_data_reader.imgNext()
                
                if cur_image is None:
                    break
                
                if first_timestamp == 0:
                    first_timestamp = cur_image.timestamp
                
                if cur_image.timestamp > first_timestamp + self._duration:
                    break
                
                # 保存图像
                image_name = f"{cur_image.timestamp}.png"
                cv2.imwrite(os.path.join(self._images_dir, image_name), cur_image.img)

                if self.mask is not None:
                    cv2.imwrite(os.path.join(self._masks_dir, image_name), self.mask)

                camera_pose = self._raw_data_reader.camera.getCameraPose(cur_image.pose)

                if self.save_depth:
                    depth_image = CommonTools.getDepthO3d(camera_pose, self._whole_points, self._raw_data_reader.camera)
                    cv2.imwrite(os.path.join(self._depths_dir, image_name), depth_image)

                camera_extrinsic_quat = Rotation.from_matrix(camera_pose.T_inv[:3,:3]).as_quat()
                camera_extrinsic_t = camera_pose.T_inv[:3,3]

                # 写入图像参数 - 第一行
                image_count += 1
                fid.write(f"{image_count} {camera_extrinsic_quat[3]} {camera_extrinsic_quat[0]} {camera_extrinsic_quat[1]} {camera_extrinsic_quat[2]} "
                         f"{camera_extrinsic_t[0]} {camera_extrinsic_t[1]} {camera_extrinsic_t[2]} 1 {image_name}\n")
                
                # 写入第二行 - 由于没有特征点匹配信息,写入空行
                fid.write("\n")
                
                print(f"Saved camera frame: {image_count}")

    def run(self):
        self.save_camera_frame()
        self.save_camera_info()
        self.save_whole_points()
        return True


if __name__ == "__main__":

    base_path = "/home/rick/Datasets/slam2000-雪乡情-正走"
    save_path = os.path.join(base_path, "colmap")
    
    pose_path = os.path.join(base_path, "V5-2024-11-05_15-28-23_808/optimised_2024-11-05_15-30-42_602.bin")
    las_path = os.path.join(base_path, "V5-2024-11-05_15-28-23_808/optimised_2024-11-05_15-30-42_602.las")
    lidar_parms = LidarParams(pose_path, las_path)

    imu_pose_path = os.path.join(base_path, "V5-2024-11-05_15-28-23_808/IMUPOS.bin")
    video_path = os.path.join(base_path, "SLAM_PRJ_001/OPTICAL_CAM/optcam_1.h265")
    video_timestamp_path = os.path.join(base_path, "SLAM_PRJ_001/OPTICAL_CAM/optcam_1.ts")
    yaml_path = os.path.join(base_path, "SLAM_PRJ_001/slam_calib.yaml")
    video_parms = VideoParams(video_path, video_timestamp_path, yaml_path, imu_pose_path, True, True)

    imu_path = os.path.join(base_path, "SLAM_PRJ_001/20241105-144253_Imu_Data.bin")
    imu_parms = ImuParams(imu_path)

    reader = RawDataReader(lidar_parms, video_parms, None)
    
    pp = PreProcess(reader, save_path, 20)
    pp.run()