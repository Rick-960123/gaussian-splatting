import open3d as o3d
import numpy as np
import cv2
def read_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return pcd.points

points = read_ply("/home/rick/Datasets/Custom_tum/points/184087.380093.ply")
points = np.hstack((points, np.array([[0,0,0] for i in range(len(points))])))
img = cv2.imread("/home/rick/Datasets/Custom_tum/rgb/184087.380093.png")
K = np.array([2071.184147  ,0 , 2051.995468, 0, 2071.184147,  1589.171711, 0, 0, 1]).reshape((3,3))

for p in points:
    if (p[2] < 0):
        continue
    p_xyz = np.array([p[:3]]).transpose()
    uv = K @ p_xyz / p_xyz[2][0]
    u = round(uv[0][0])
    v = round(uv[1][0])
    if u > 0 and u < img.shape[1] and v > 0 and v < img.shape[0]:
        p[5] = img[v][u][0]/255.0
        p[4] = img[v][u][1]/255.0
        p[3] = img[v][u][2]/255.0

tmp = np.array(points)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(tmp[:,:3])
pcd.colors = o3d.utility.Vector3dVector(tmp[:,3:])
pcd.normals  = o3d.utility.Vector3dVector(np.zeros((tmp.shape[0], 3)))
# 保存为PLY文件
o3d.io.write_point_cloud("/home/rick/Datasets/Custom_tum/color.ply", pcd)