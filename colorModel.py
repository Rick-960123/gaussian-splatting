import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class colorModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, points, K):
        self.K = torch.tensor(torch.tensor(np.asarray(K)).float().cuda())
        self._xyz = torch.tensor(np.asarray(points)).float().cuda()
        self._rgb = nn.Parameter(torch.zeros(len(points), 3).float().cuda().requires_grad_(True))
        self.R = nn.Parameter(torch.zeros(3, 3).float().cuda().requires_grad_(True))
        self.t = nn.Parameter(torch.zeros(3, 1).float().cuda().requires_grad_(True))

    def setNext(self, img, pose):
        self._img = torch.tensor(torch.tensor(np.asarray(img)).float().cuda())
        self.R = nn.Parameter(torch.tensor(np.asarray(np.linalg.inv(pose)[:3, :3])).float().cuda().requires_grad_(False))
        self.t = nn.Parameter(torch.tensor(np.asarray(np.linalg.inv(pose)[:3, 3])).float().cuda().requires_grad_(False))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


def projection_loss(clrmodel):
    point_2D = (clrmodel.K @ (clrmodel.R @ clrmodel._xyz.transpose(0, 1) + clrmodel.t.unsqueeze(-1))).transpose(0, 1)

    depth = point_2D[:, 2]
    u_v = point_2D[:, :2] / point_2D[:, 2].unsqueeze(-1).long()

    # 先进行深度检查
    valid_depth_mask = (depth > 0.1) & (depth < float('inf'))

    filtered_depth = depth[valid_depth_mask]
    filtered_u_v = u_v[valid_depth_mask]
    filtered_rgb = clrmodel._rgb[valid_depth_mask]
    filtered_xyz = clrmodel._xyz[valid_depth_mask]

    # 获取唯一的 UV 坐标及其索引
    unique_u_v, inverse_indices = torch.unique(filtered_u_v, dim=0, return_inverse=True)

    positive_indices = torch.arange(unique_u_v.size(0)).cuda()[inverse_indices]
    unique_rgb = filtered_rgb[positive_indices]
    unique_xyz = filtered_xyz[positive_indices]

    # 初始化最小深度
    min_depths = torch.full((unique_u_v.shape[0],), float('inf')).cuda()

    # 更新最小深度
    min_depths[inverse_indices] = torch.min(
        min_depths[inverse_indices],
        filtered_depth
    )

    # 创建有效掩码
    valid_mask = (min_depths < float('inf'))
    final_uv = unique_u_v[valid_mask]


    # 检查有效性
    valid = (final_uv[:, 0] >= 0) & (final_uv[:, 0] < clrmodel._img.shape[1]) & (final_uv[:, 1] >= 0) & (final_uv[:, 1] < clrmodel._img.shape[0])

    valid_uv_indices = valid.nonzero(as_tuple=True)[0] 
    valid_uv = final_uv[valid_uv_indices].long()
    img_colors = clrmodel._img[valid_uv[:, 0], valid_uv[:, 1]]

    render_color = final_rgb[valid_uv_indices]

    color_diff = render_color - (img_colors / 255)
    loss = (color_diff ** 2).mean()

    # 转换为 NumPy 数组
    img_colors_np = img_colors.cpu().numpy()
    xyz_np = final_xyz[valid_uv_indices].cpu().numpy()

    # 合并数据
    combined_data = np.hstack((xyz_np, img_colors_np))

    # 保存为文本文件
    np.savetxt("./combined_data.txt", combined_data.reshape(-1, combined_data.shape[-1]),
            fmt='%.6f', header='X Y Z R G B', comments='')

    
    return loss