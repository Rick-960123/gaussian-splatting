import numpy as np

# 读取文件内容
def read_poses(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    timestamps = []
    positions = []
    orientations = []

    for line in lines:
        parts = line.strip().split()
        if parts[0].startswith("#"):
            continue
        if len(parts) == 8:
            timestamp = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:])
            timestamps.append(timestamp)
            positions.append([tx, ty, tz])
            orientations.append([qx, qy, qz, qw])
        if len(timestamps) > 10:
            break
    return np.array(timestamps), np.array(positions), np.array(orientations)

# 示例文件路径
file_path = "/home/rick/Datasets/Custom_tum_test/groundtruth.txt"
timestamps, positions, orientations = read_poses(file_path)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
# 可视化位姿数据，包括朝向
def visualize_poses(timestamps, positions, orientations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取位姿中的位置
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # 绘制位姿轨迹
    ax.plot(x, y, z, label='Trajectory', color='b')

    # 绘制每个位姿点
    ax.scatter(x, y, z, c='r', marker='o')

    # 绘制每个位姿的朝向
    for i in range(len(positions)):
        position = positions[i]
        orientation = orientations[i]
        # 将四元数转换为旋转矩阵
        rotation = R.from_quat(orientation).as_matrix()
        # 绘制坐标轴，表示朝向
        ax.quiver(position[0], position[1], position[2], 
                  rotation[0, 2], rotation[1, 2], rotation[2, 2], color='r', length=0.1)


    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory of Poses with Orientation')

    # 添加图例
    ax.legend()

    plt.show()

visualize_poses(timestamps, positions, orientations)

