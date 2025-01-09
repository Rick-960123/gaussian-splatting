#!/usr/bin/env python
import rospy
import tf
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation, Slerp
import matplotlib.pyplot as plt
import threading
import os

class ColmapPoseVisualizer:
    def __init__(self, node_name, path):
        # 创建发布器
        self.pose_pub = rospy.Publisher(f'/{node_name}/camera_poses', PoseStamped, queue_size=10)
        self.path_pub = rospy.Publisher(f'/{node_name}/camera_trajectory', Path, queue_size=10)
        self.marker_pub = rospy.Publisher(f'{node_name}/camera_markers', MarkerArray, queue_size=10)
        
        self.path = Path()
        self.path.header.frame_id = "map"
        self.marker_array = MarkerArray()
        self.marker_id = 0

        self.scale = 1.0
        if node_name == "colmap2":
            self.scale = 0.741

        poses = self.read_images_text(path)
        # self.publish_poses(poses)
        new_thread = threading.Thread(target=self.publish_poses, args=(poses,))
        new_thread.start()

    def read_images_text(self, path):
        """读取COLMAP的images.txt文件"""
        poses = {}
        with open(path, "r") as f:
            lines = f.readlines()
            
        line_index = 0
        first_pose_inv = None
        image_name = None

        while line_index < len(lines):
            line = lines[line_index].strip()
            
            # 跳过注释和空行
            if len(line) == 0 or line.startswith("#"):
                line_index += 1
                continue
                
            # 读取图像数据
            data = line.split()
            image_id = int(data[0])
            qw, qx, qy, qz = map(float, data[1:5])  # 四元数
            tx, ty, tz = map(float, data[5:8])      # 平移
            image_name = str(data[9])

            # 构建位姿矩阵
            pose = np.eye(4)
            rotation = tf.transformations.quaternion_matrix([qx, qy, qz, qw])
            pose[:3, :3] = rotation[:3, :3]
            pose[:3, 3] = np.array([tx, ty, tz]) * self.scale

            tmp_T = np.array([1,0,0,0,
                0,1,0,0,
                0,0,-1,0,
                0,0,0,1]).reshape((4,4)) 
            
            if self.scale == 1.0:
                pose = tmp_T @ np.linalg.inv(pose)

            if first_pose_inv is None:
                first_pose_inv = np.linalg.inv(pose)
                print(f"First Image Name: {image_name}")
                print("First Pose")
                poses[image_id] = first_pose_inv @ pose
                print(poses[image_id])
            
            poses[image_id] = first_pose_inv @ pose
        
            # 跳过下一行（包含特征点信息）
            line_index += 2
        
        print(f"End Pose Image Name: {image_name}")

        return poses

    def publish_poses(self, poses):
        x_data, y_data, z_data, imageids = [], [], [], []
        
        """发布所有位姿"""
        for image_id, pose in sorted(poses.items()):
            # 创建PoseStamped消息
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "map"
            pose_msg.header.stamp = rospy.Time.now()
            
            # 获取位置和方向
            position = pose[:3, 3]

            quaternion = tf.transformations.quaternion_from_matrix(pose)
            euler_angles = tf.transformations.euler_from_matrix(pose)
            euler_x, euler_y, euler_z = euler_angles
            
            x_data.append(euler_x * 180/3.14159)
            y_data.append(euler_y* 180/3.14159)
            z_data.append(euler_z* 180/3.14159)
            imageids.append(image_id)

            pose_msg.pose.position.x = position[0]
            pose_msg.pose.position.y = position[1]
            pose_msg.pose.position.z = position[2]
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            pose_msg.pose.orientation.w = quaternion[3]
            
            # 添加到路径
            self.path.poses.append(pose_msg)
            
            # 发布消息
            self.pose_pub.publish(pose_msg)
            self.path_pub.publish(self.path)
            
            rospy.sleep(0.01)  # 添加小延迟使可视化更流畅

        # fig, ax = plt.subplots()
        # ax.plot(imageids, x_data, label="Euler X", color='r')
        # ax.plot(imageids, y_data, label="Euler Y", color='g')
        # ax.plot(imageids, z_data, label="Euler Z", color='b')

        # ax.set_xlabel('imageId')
        # ax.set_ylabel('Euler Angles')
        # ax.set_title('XYZ Euler Angles Over Time')
        # ax.legend()
        # plt.show()

        while not rospy.is_shutdown():
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "map"
            pose_msg.header.stamp = rospy.Time.now()
            
            # 获取位置和方向
            position = np.array([0.0, 0.0, 0.0])
            quaternion = np.array([0.0, 0.0, 0.0, 1.0])
            
            pose_msg.pose.position.x = position[0]
            pose_msg.pose.position.y = position[1]
            pose_msg.pose.position.z = position[2]
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            pose_msg.pose.orientation.w = quaternion[3]
            self.pose_pub.publish(pose_msg)
            rospy.sleep(0.01)

def main():
    rospy.init_node("VisColMapPose", anonymous=True)

    colmap_path = "/home/rick/Datasets/slam2000-雪乡情-正走/colmap/sparse/0/images.txt"  # 修改为实际路径
    colmap_path2 = "/home/rick/Downloads/1/images.txt"

    ColmapPoseVisualizer("colmap", colmap_path)
    # ColmapPoseVisualizer("colmap2", colmap_path2)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass