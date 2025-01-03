#!/usr/bin/env python
import rospy
import tf
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import os

class ColmapPoseVisualizer:
    def __init__(self):
        rospy.init_node('colmap_pose_visualizer', anonymous=True)
        
        # 创建发布器
        self.pose_pub = rospy.Publisher('/camera_poses', PoseStamped, queue_size=10)
        self.path_pub = rospy.Publisher('/camera_trajectory', Path, queue_size=10)
        self.marker_pub = rospy.Publisher('/camera_markers', MarkerArray, queue_size=10)
        
        self.path = Path()
        self.path.header.frame_id = "map"
        self.marker_array = MarkerArray()
        self.marker_id = 0

    def read_images_text(self, path):
        """读取COLMAP的images.txt文件"""
        poses = {}
        with open(path, "r") as f:
            lines = f.readlines()
                
        for line in lines:
            line = line.strip()
            
            # 跳过注释和空行
            if len(line) == 0 or line.startswith("#"):
                continue
                
            # 读取数据
            data = line.split()
            timestamp = float(data[0])
            tx, ty, tz = map(float, data[1:4])      # 平移
            qx, qy, qz, qw = map(float, data[4:8])  # 四元数
            
            # 构建位姿矩阵
            pose = np.eye(4)
            # 从四元数获取旋转矩阵
            rotation = tf.transformations.quaternion_matrix([qx, qy, qz, qw])
            pose[:3, :3] = rotation[:3, :3]
            pose[:3, 3] = [tx, ty, tz]
            
            poses[timestamp] = pose
            
        return poses

    def publish_poses(self, poses):
        """发布所有位姿"""
        for timestamp, pose in sorted(poses.items()):
            # 创建PoseStamped消息
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "map"
            pose_msg.header.stamp = rospy.Time.now()
            
            # 获取位置和方向
            position = pose[:3, 3]
            quaternion = tf.transformations.quaternion_from_matrix(pose)
            
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

        while True:
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
    visualizer = ColmapPoseVisualizer()
    
    # 读取COLMAP输出文件
    colmap_path = "/home/rick/Datasets/slam2000-雪乡情-正走/tum/groundtruth.txt"  # 修改为实际路径
    poses = visualizer.read_images_text(colmap_path)
    
    # 发布位姿
    visualizer.publish_poses(poses)
    
    # 保持节点运行
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass