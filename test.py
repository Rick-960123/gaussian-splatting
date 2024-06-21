import torch
import numpy as np

# 假设旋转矩阵 R 是单位矩阵
R = np.eye(4, dtype=np.float32)

# 假设平移矩阵 T 是单位矩阵
T = np.eye(4, dtype=np.float32)

# 假设平移向量 trans 和缩放因子 scale
trans = np.array([0, 0, 0], dtype=np.float32)
scale = 1.0

# 定义投影参数
znear = 0.1
zfar = 1000.0
FoVx = 90.0  # 水平视场角
FoVy = 90.0  # 垂直视场角

# 定义一个世界坐标系中的点
world_point = np.array([30, 20, 800, 1], dtype=np.float32)  # 齐次坐标

# 计算世界坐标系到视图坐标系的变换矩阵
def getWorld2View2(R, T, trans, scale):
    # 这里只是一个简单的示例实现，假设没有旋转和平移
    world2view = np.eye(4, dtype=np.float32)
    return world2view

# 计算视图坐标系到投影坐标系的变换矩阵
def getProjectionMatrix(znear, zfar, fovX, fovY):
    aspect_ratio = 1.0  # 假设宽高比为1
    f = 1.0 / np.tan(np.deg2rad(fovX) / 2)
    proj_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (zfar + znear) / (znear - zfar), (2 * zfar * znear) / (znear - zfar)],
        [0, 0, -1, 0]
    ], dtype=np.float32)
    return proj_matrix

# 计算变换矩阵
world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
projection_matrix = torch.tensor(getProjectionMatrix(znear, zfar, FoVx, FoVy)).transpose(0, 1).cuda()

# 计算完整的变换矩阵
full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

# 将世界坐标系中的点转换到投影坐标系
world_point_tensor = torch.tensor(world_point, dtype=torch.float32).cuda()
projected_point = full_proj_transform.mv(world_point_tensor)

print(projected_point.cpu().numpy())
