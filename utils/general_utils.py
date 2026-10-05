#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

# 另一种根据法线计算旋转的方式
# def compute_rotation_from_normal(normal):
#     normal = normal / torch.norm(normal)  # 确保法向量是单位向量
#     z_axis = torch.tensor([0.0, 0.0, 1.0], device=normal.device)
    
#     # 计算旋转轴和旋转角度
#     rot_axis = torch.cross(z_axis, normal)
#     rot_angle = torch.acos(torch.dot(z_axis, normal))

#     if rot_axis.norm() == 0:
#         # 如果法向量已经对齐到z轴，则旋转轴为零
#         rot_axis = torch.tensor([1.0, 0.0, 0.0], device=normal.device)
#         rot_angle = torch.tensor(0.0, device=normal.device)
#     else:
#         rot_axis = rot_axis / rot_axis.norm()
    
#     # 计算四元数
#     sin_half_angle = torch.sin(rot_angle / 2)
#     cos_half_angle = torch.cos(rot_angle / 2)

#     # 创建四元数 [w, x, y, z]，其中 w = cos(θ/2)，x = sin(θ/2) * rot_axis[0]，等等
#     quat = torch.cat([
#         cos_half_angle.unsqueeze(0),  # [1]
#         (sin_half_angle * rot_axis[0]).unsqueeze(0),
#         (sin_half_angle * rot_axis[1]).unsqueeze(0),
#         (sin_half_angle * rot_axis[2]).unsqueeze(0)
#     ], dim=0)  # [4]

#     quat = quat.unsqueeze(0)  # [1, 4]，添加批次维度
    
#     return quat

def compute_rotation_from_normal(normals):
    # 确保法向量是单位向量
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=normals.device).unsqueeze(0)
    
    # 计算旋转轴和旋转角度
    rot_axis = torch.cross(z_axis, normals, dim=1)
    rot_angle = torch.acos(torch.clamp(torch.sum(z_axis * normals, dim=1), -1.0, 1.0))
    
    # 处理 rot_axis 为零的情况
    rot_axis_norm = torch.norm(rot_axis, dim=1)
    zero_axis = (rot_axis_norm == 0)
    rot_axis[zero_axis] = torch.tensor([1.0, 0.0, 0.0], device=normals.device)
    rot_angle[zero_axis] = 0.0

    rot_axis = rot_axis / torch.clamp(rot_axis_norm.unsqueeze(1), min=1e-8)

    # 使用四元数表示旋转
    sin_half_angle = torch.sin(rot_angle / 2)
    cos_half_angle = torch.cos(rot_angle / 2)
    quat = torch.cat([cos_half_angle.unsqueeze(1),
                    (sin_half_angle.unsqueeze(1) * rot_axis)
                    ], dim=1)
    return quat