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
    def __init__(self, pose_path, las_path, video_path, video_timestamp_path, T_c2b, save_path, camera):
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
        self._camera = camera

        with open(video_timestamp_path, 'r') as f:
            for line in f:
                self._videoTimeList.append(float(line.strip()) + self._camera_time_error)

        # Ensure directories exist
        os.makedirs(os.path.join(save_path, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "depth"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "point"), exist_ok=True)

        self.pose_txt = os.path.join(self.save_path, "groundtruth.txt")
        self.rgb_txt = os.path.join(self.save_path, "rgb.txt")
        self.depth_txt = os.path.join(self.save_path, "depth.txt")
        self.point_txt = os.path.join(self.save_path, "point.txt")
        self.camera_txt =  os.path.join(self.save_path, "cameras.txt")

        if os.path.exists(self.depth_txt):
            os.remove(self.depth_txt)
        if os.path.exists(self.rgb_txt):
            os.remove(self.rgb_txt)
        if os.path.exists(self.point_txt):
            os.remove(self.point_txt)
        if os.path.exists(self.pose_txt):
            os.remove(self.pose_txt)
        if os.path.exists(self.camera_txt):
            os.remove(self.camera_txt)

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
    
    def getAllPoints(self):
        points = self._las.xyz
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
        height = 400
        width = 300

        Fx = self._camera.fx 
        Fy = self._camera.fy
        Cx = self._camera.cx 
        Cy = self._camera.cy

        image = torch.zeros((height, width, 1), device='cuda')  # 使用GPU
        depth_buffer = torch.full((height, width), float('inf'), device='cuda') 
        pose_T_inv = torch.tensor(pose.T_inv, device='cuda')
        points_tensor = torch.tensor(points, device='cuda')

        # 执行点的变换
        p_hom = pose_T_inv[:3, :3] @ points_tensor.T + pose_T_inv[:3, 3].unsqueeze(1)

        # 过滤掉 z 坐标小于等于0的点
        valid_mask = p_hom[2, :] > 1e-9
        p_hom = p_hom[:, valid_mask]

        # 投影到图像平面
        p_proj_x = (p_hom[0] * Fx / p_hom[2] + Cx).long()
        p_proj_y = (p_hom[1] * Fy / p_hom[2] + Cy).long()

        # 过滤掉超出图像边界的点
        valid_mask = (0 <= p_proj_x) & (p_proj_x < width) & (0 <= p_proj_y) & (p_proj_y < height)
        p_proj_x = p_proj_x[valid_mask]
        p_proj_y = p_proj_y[valid_mask]
        p_hom_z = p_hom[2, valid_mask]

        # 将有效点的投影坐标和深度信息用于更新深度缓冲区和图像
        for x, y, z in zip(p_proj_x.cpu().numpy(), p_proj_y.cpu().numpy(), p_hom_z.cpu().numpy()):
            if z < depth_buffer[y, x]:
                depth_buffer[y, x] = z
                image[y, x] = (z * 1000.0)

        # 将图像转换回CPU进行进一步处理
        depth = image.permute(2, 0, 1).cpu().squeeze(0)
        depth_img = depth.numpy()
        depth_map_cm_uint16 = depth_img.astype(np.uint16)
        return depth_map_cm_uint16

    def get_depth(self, pose, points):
        pcd = o3d.geometry.PointCloud(np.array(points)[:,:3])
        intrinsic = o3d.core.Tensor([[self._camera.fx , 0, self._camera.cx ], [0, self._camera.fy , self._camera.cy],
                                 [0, 0, 1]])
        extrinsic = o3d.core.Tensor(pose.T_inv)
        depth = o3d.geometry.Image(
                np.asarray(pcd.project_to_depth_image(
                                                intrinsics=intrinsic,
                                                extrinsic = extrinsic,
                                                depth_scale=1000.0,
                                                depth_max=100.0)))
        depth_img = np.asarray(depth)
        depth_map_cm_uint16 = depth_img.astype(np.uint16)
        return depth_map_cm_uint16
        
    def run(self):
        while True:
            cur_pose = self.poseNext()
            if cur_pose is None:
                break

            cur_points = self.pointsNext(cur_pose)

            if len(cur_points) == 0:
                continue
            
            cur_image = self.imgNext(cur_pose)
            
            if cur_image == None:
                continue
            
            # 保存图像和点云
            cv2.imwrite(os.path.join(self.save_path, f"rgb/{cur_pose.timestamp}.png"), cur_image.img)

            # depth_img = self.get_depth_image(cur_pose, self._las.xyz)
            # cv2.imwrite(os.path.join(self.save_path, f"depth/{cur_pose.timestamp}.png"), depth_img)

            self.save_point_cloud_to_ply(os.path.join(self.save_path, f"point/{cur_pose.timestamp}.ply"), self.transform_points_body_to_camera(self.transform_points_world_to_body(cur_points, cur_pose)))

            with open( self.pose_txt, 'a') as ofs_pose, \
                open( self.rgb_txt, 'a') as ofs_rgb, \
                open( self.depth_txt, 'a') as ofs_depth, \
                open( self.point_txt, 'a') as ofs_point:

                if self._index == 0:  # 只在第一次写入时添加标题
                    ofs_pose.write("# timestamp tx ty tz qx qy qz qw\n")
                    ofs_rgb.write("#  timestamp filename\n")
                    ofs_point.write("#  timestamp filename\n")
                    ofs_depth.write("#  timestamp filename\n")
                    
                ofs_rgb.write(f"{cur_pose.timestamp} rgb/{cur_pose.timestamp}.png\n")
                ofs_point.write(f"{cur_pose.timestamp} point/{cur_pose.timestamp}.ply\n")
                ofs_depth.write(f"{cur_pose.timestamp} depth/{cur_pose.timestamp}.png\n")

                camera_pose = self.transform_pose_world_to_camera(cur_pose)
                ofs_pose.write(f"{camera_pose.timestamp} {camera_pose.posX} {camera_pose.posY} {camera_pose.posZ} "
                        f"{camera_pose.qX} {camera_pose.qY} {camera_pose.qZ} {camera_pose.qW}\n")
                
            self._index += 1

            print(self._index)

            if self._index == 50:
                break

        cameraInfo = " ".join([str(num) for num in self._camera.getlist()])
        with open(self.camera_txt, 'a') as ofs_camera:
            ofs_camera.write(f"{cameraInfo}\n")

        # whole_points = self.transform_points_body_to_camera(self._las.xyz)
        # self.save_point_cloud_to_ply(os.path.join(self.save_path, "points3D_density.ply"), whole_points)
        # whole_points = self.filter_point_cloud(whole_points)
        # self.save_point_cloud_to_ply(os.path.join(self.save_path, "points3D.ply"), whole_points)

        return True

 
if __name__ == "__main__":
    pose_path = "/home/rick/Datasets/SN_00250/SLAM_PRJ_001/2024-04-23_13-46-50_570/optimised_2024-04-23_14-18-25_662.bin"
    las_path = "/home/rick/Datasets/SN_00250/SLAM_PRJ_001/2024-04-23_13-46-50_570/optimised_2024-04-23_14-18-25_662.las"

    video_path = "/home/rick/Datasets/SN_00250/SLAM_PRJ_001/OPTICAL_CAM/optcam_1.h265"
    video_timestamp_path = "/home/rick/Datasets/SN_00250/SLAM_PRJ_001/OPTICAL_CAM/optcam_1.ts"
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
    
    camera_pose =  T @ tmp_T
    
    camera = Camera(0, "PINHOLE", 4000, 3000, 2071.184147, 2071.184147, 2051.995468, 1589.171711)
    pp = PreProcess(pose_path, las_path, video_path, video_timestamp_path, camera_pose, save_path, camera)
    pp.run()