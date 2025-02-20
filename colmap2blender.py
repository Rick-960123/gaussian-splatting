import os
import numpy as np
import shutil
from PIL import Image
import cv2
import collections

def read_cameras_text(path):
    """读取 COLMAP cameras.txt 文件"""
    cameras = {}
    with open(path, "r") as f:
        lines = f.readlines()
        
    # 跳过注释行
    lines = [line for line in lines if line[0] != "#"]
    
    for line in lines:
        elements = line.split()
        camera_id = int(elements[0])
        model = elements[1]
        width = int(elements[2])
        height = int(elements[3])
        params = np.array(list(map(float, elements[4:])))
        cameras[camera_id] = Camera(model, width, height, params)
    return cameras

def read_images_text(path):
    """读取 COLMAP images.txt 文件"""
    images = {}
    with open(path, "r") as f:
        lines = f.readlines()
    
    # 跳过注释行
    lines = [line for line in lines if line[0] != "#"]
    
    # COLMAP的images.txt每个图像占用两行
    for i in range(0, len(lines), 2):
        # 第一行：图像ID，旋转四元数，平移向量，相机ID，图像名称
        line = lines[i].split()
        image_id = int(line[0])
        qvec = np.array(list(map(float, line[1:5])))
        tvec = np.array(list(map(float, line[5:8])))
        camera_id = int(line[8])
        image_name = line[9]
        
        images[image_id] = Image(image_id, qvec, tvec, camera_id, image_name)
        
    return images

def convert_colmap_to_blendedmvs(colmap_path, image_path, output_path):
    """
    将COLMAP txt格式转换为BlendedMVS格式
    
    Args:
        colmap_path: COLMAP sparse文件夹路径（包含cameras.txt和images.txt）
        image_path: 原始图像文件夹路径
        output_path: 输出BlendedMVS格式的路径
        scene_id: 场景ID
    """
    # 创建输出目录结构
    scene_path = os.path.join(output_path)
    os.makedirs(os.path.join(scene_path, "blended_images"), exist_ok=True)
    os.makedirs(os.path.join(scene_path, "cams"), exist_ok=True)
    
    # 读取COLMAP数据
    cameras = read_cameras_text(os.path.join(colmap_path, "cameras.txt"))
    images = read_images_text(os.path.join(colmap_path, "images.txt"))
    
    # 转换每个图像
    for img_id, img_data in images.items():
        # 获取相机参数
        camera = cameras[img_data.camera_id]
        
        # 构建外参矩阵
        R = qvec2rotmat(img_data.qvec)
        t = img_data.tvec
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        
        # 构建内参矩阵
        intrinsic = np.eye(3)
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
            intrinsic[0,0] = fx
            intrinsic[1,1] = fy
            intrinsic[0,2] = cx
            intrinsic[1,2] = cy
        elif camera.model == "SIMPLE_RADIAL":
            f, cx, cy, k = camera.params
            intrinsic[0,0] = f
            intrinsic[1,1] = f
            intrinsic[0,2] = cx
            intrinsic[1,2] = cy
        else:
            print(f"Warning: Camera model {camera.model} not fully supported")
        
        # 写入相机参数文件
        cam_filename = os.path.join(scene_path, "cams", f"{img_id:08d}_cam.txt")
        with open(cam_filename, "w") as f:
            f.write("extrinsic\n")
            np.savetxt(f, extrinsic, fmt="%.6f")
            f.write("\nintrinsic\n")
            np.savetxt(f, intrinsic, fmt="%.6f")
            f.write("\n%.6f %.6f\n" % (0.1, 100.0))  # 默认深度范围
        
        # 复制图像文件
        src_img = os.path.join(image_path, img_data.name)
        dst_img = os.path.join(scene_path, "blended_images", f"{img_id:08d}.jpg")
        img = cv2.imread(src_img)
        cv2.imwrite(dst_img, img)

def qvec2rotmat(qvec):
    """四元数转旋转矩阵"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

class Camera:
    def __init__(self, model, width, height, params):
        self.model = model
        self.width = width
        self.height = height
        self.params = params

class Image:
    def __init__(self, id, qvec, tvec, camera_id, name):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name

if __name__ == "__main__":
    # 设置路径
    colmap_path = "/home/rick/Datasets/slam2000-雪乡情-正走/colmap/sparse/0"  # 包含cameras.txt和images.txt的目录
    image_path = "/home/rick/Datasets/slam2000-雪乡情-正走/colmap/images"          # 原始图像目录
    output_path = "/home/rick/Datasets/slam2000-雪乡情-正走/blender"         # 输出BlendedMVS格式的目录
    
    # 执行转换
    convert_colmap_to_blendedmvs(colmap_path, image_path, output_path)