import cv2
import numpy as np
import queue
import os
import open3d as o3d
import laspy
import struct

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
    def __init__(self, pose_path, las_path, video_path, video_timestamp_path, save_path):
        self._pose_file = open(pose_path, 'rb')  # Assuming binary read mode for pose file

        self._cap = cv2.VideoCapture(video_path)
        self._las = laspy.read(las_path)

        self._p_idx = 0
        self._imgIdx = 0
        self._lastVideoTime = 0
        self.save_path = save_path
        self.image_cache = queue.Queue()
        self._videoTimeList = []

        with open(video_timestamp_path, 'r') as f:
            for line in f:
                self._videoTimeList.append(float(line.strip()))

        # Ensure directories exist
        os.makedirs(os.path.join(save_path, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "points"), exist_ok=True)

        self.pose_txt = os.path.join(self.save_path, "groudtruth.txt")
        self.rgb_txt = os.path.join(self.save_path, "rgb.txt")
        self.points_txt = os.path.join(self.save_path, "points.txt")

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
                points.append(self._las.xyz[self._p_idx])
                self._p_idx += 1
            else:
                break
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
        pcd.colors = o3d.utility.Vector3dVector(tmp[:,3:6])
        pcd.normals  = o3d.utility.Vector3dVector(tmp[:,6:])
        # 保存为PLY文件
        o3d.io.write_point_cloud(ply_file, pcd)
    def run(self):
        while True:
            
            cur_pose = self.poseNext()
            if cur_pose is None:
                break

            cur_points = self.pointsNext(cur_pose)

            if len(cur_points) == 0:
                continue

            cur_image = self.imgNext(cur_pose)
            
            # 保存图像和点云
            cv2.imwrite(os.path.join(self.save_path, f"rgb/{cur_pose.timestamp}.png"), cur_image.img)

            self.save_point_cloud_to_ply(os.path.join(self.save_path, f"points/{cur_pose.timestamp}.ply"), cur_points)  # Assuming laspy.write can save as PLY

            # 保存地面真实值
            with open(os.path.join(self.save_path, "groudtruth.txt"), 'a') as ofs, \
                open(os.path.join(self.save_path, "rgb.txt"), 'a') as ofs_rgb, \
                open(os.path.join(self.save_path, "points.txt"), 'a') as ofs_points:

                if self._imgIdx == 1:  # 只在第一次写入时添加标题
                    ofs.write("# timestamp tx ty tz qx qy qz qw\n")
                    ofs_rgb.write("#  timestamp filename\n")
                    ofs_points.write("#  timestamp filename\n")

                ofs_rgb.write(f"{cur_pose.timestamp} rgb/{cur_pose.timestamp}.png\n")
                ofs_points.write(f"{cur_pose.timestamp} points/{cur_pose.timestamp}.ply\n")

                ofs.write(f"{cur_pose.timestamp} {cur_pose.t[0]} {cur_pose.t[1]} {cur_pose.t[2]} "
                        f"{cur_pose.q[0]} {cur_pose.q[1]} {cur_pose.q[2]} {cur_pose.q[3]}\n")

        return True

if __name__ == "__main__":
    pose_path = "/home/rick/Datasets/S181-办公室实时点云-20#/SN_00250/SLAM_PRJ_001/2024-04-23_13-46-50_570/optimised_2024-04-23_14-18-25_662.bin"
    las_path = "/home/rick/Datasets/S181-办公室实时点云-20#/SN_00250/SLAM_PRJ_001/2024-04-23_13-46-50_570/optimised_2024-04-23_14-18-25_662.las"

    video_path = "/home/rick/Datasets/S181-办公室实时点云-20#/SN_00250/SLAM_PRJ_001/OPTICAL_CAM/optcam_1.h265"
    video_timestamp_path = "/home/rick/Datasets/S181-办公室实时点云-20#/SN_00250/SLAM_PRJ_001/OPTICAL_CAM/optcam_1.ts"
    save_path = "/home/rick/Datasets/Custom_tum"
    
    pp = PreProcess(pose_path, las_path, video_path, video_timestamp_path, save_path)
    pp.run()
