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

class ColmapPoseVisualizer:
    def __init__(self, node_name:str, paths:list, scales:list):
        # 创建发布器
        self.publishers = []
        self.poses = []
        for idx, path in enumerate(paths):
            pose_pub = rospy.Publisher(f'/{node_name}_{idx}/camera_poses', PoseStamped, queue_size=10)
            path_pub = rospy.Publisher(f'/{node_name}_{idx}/camera_trajectory', Path, queue_size=10)
            self.publishers.append({pose_pub:pose_pub, path_pub:path_pub})
            poses = self.read_images_text(path, scales[idx])
            self.poses.append(poses)
            threading.Thread(target=self.publish_poses, args=(self.poses[idx], self.publishers[idx][pose_pub], self.publishers[idx][path_pub])).start()

        self.plot_poses(self.poses)

    def read_images_text(self, path, scale):
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
            pose_inv = np.eye(4)
            rotation = tf.transformations.quaternion_matrix([qx, qy, qz, qw])
            pose_inv[:3, :3] = rotation[:3, :3]
            pose_inv[:3, 3] = np.array([tx, ty, tz]) * scale
            
            if first_pose_inv is None:
                first_pose_inv = pose_inv
                print(f"First Image Name: {image_name}")
                print("First Pose")
                print(pose_inv)
            
            poses[image_id] = first_pose_inv @ np.linalg.inv(pose_inv)

            # 跳过下一行（包含特征点信息）
            line_index += 2

        print(f"End Pose Image Name: {image_name}")

        return poses
    
    def plot_poses(self, poseslist):
        plot_data = []

        for poses in poseslist:
            x_data, y_data, z_data, qx_data, qy_data, qz_data, qw_data, x_o, y_o, z_o, imageids = [], [], [], [], [], [], [], [], [], [], []
            for timestamp, pose in sorted(poses.items()):
                # 创建PoseStamped消息
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = "map"
                pose_msg.header.stamp = rospy.Time.now()
                
                # 获取位置和方向
                position = pose[:3, 3]
                quaternion = tf.transformations.quaternion_from_matrix(pose)
                
                x_data.append(position[0])
                y_data.append(position[1])
                z_data.append(position[2])
                qx_data.append(quaternion[0])
                qy_data.append(quaternion[1])
                qz_data.append(quaternion[2])
                qw_data.append(quaternion[3])
                imageids.append(timestamp)

                euler_angles = tf.transformations.euler_from_matrix(pose)
                euler_x, euler_y, euler_z = euler_angles
                
                x_o.append(euler_x * 180/3.14159)
                y_o.append(euler_y* 180/3.14159)
                z_o.append(euler_z* 180/3.14159)

            plot_data.append({"x_data":x_data, "y_data":y_data, "z_data":z_data, "qx_data":qx_data, "qy_data":qy_data, "qz_data":qz_data, "qw_data":qw_data, "x_o":x_o, "y_o":y_o, "z_o":z_o, "imageids":imageids})
        
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        for idx, data in enumerate(plot_data):
            ax.plot(data['imageids'], data['x_data'], label=f"Position X {idx}", color='r')
            ax.plot(data['imageids'], data['y_data'], label=f"Position Y {idx}", color='g')
            ax.plot(data['imageids'], data['z_data'], label=f"Position Z {idx}", color='b')
        
            ax1.plot(data['imageids'], data['qx_data'], label=f"Quaternion X {idx}", color='r')
            ax1.plot(data['imageids'], data['qy_data'], label=f"Quaternion Y {idx}", color='g')
            ax1.plot(data['imageids'], data['qz_data'], label=f"Quaternion Z {idx}", color='b')
            ax1.plot(data['imageids'], data['qw_data'], label=f"Quaternion W {idx}", color='y')
            
            ax2.plot(data['imageids'], data['x_o'], label=f"Euler X {idx}", color='r')
            ax2.plot(data['imageids'], data['y_o'], label=f"Euler Y {idx}", color='g')
            ax2.plot(data['imageids'], data['z_o'], label=f"Euler Z {idx}", color='b')

        ax.set_xlabel('imageId')
        ax.set_ylabel('Position')
        ax1.set_ylabel('Quaternion')
        ax2.set_ylabel('Euler')
        ax.set_title('Position Over Time')
        ax1.set_title('Quaternion Over Time')
        ax2.set_title('Euler Over Time')

        ax.legend()
        ax1.legend()
        ax2.legend()
        plt.show()

    def publish_poses(self, poses, pose_publisher, path_publisher):
        path = Path()
        path.header.frame_id = "map"

        for timestamp, pose in sorted(poses.items()):
            # 获取位置和方向
            position = pose[:3, 3]
            quaternion = tf.transformations.quaternion_from_matrix(pose)

            # 创建PoseStamped消息
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "map"
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.pose.position.x = position[0]
            pose_msg.pose.position.y = position[1]
            pose_msg.pose.position.z = position[2]
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            pose_msg.pose.orientation.w = quaternion[3]
            pose_publisher.publish(pose_msg)

            # 添加到路径
            path.header.stamp = rospy.Time.now()
            path.poses.append(pose_msg)
            path_publisher.publish(path)
        
            rospy.sleep(0.2)  # 添加小延迟使可视化更流畅

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
            pose_publisher.publish(pose_msg)
            rospy.sleep(0.2)

def main():
    rospy.init_node("VisColMapPose", anonymous=True)
    colmap_path = ["/home/rick/Datasets/slam2000-雪乡情-正走/colmap/sparse/0/images.txt", "/home/rick/Datasets/slam2000-雪乡情-正走/colmap1/sparse/0/images.txt"]
    scales = [1.0, 0.738]
    ColmapPoseVisualizer("colmap", colmap_path, scales)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass