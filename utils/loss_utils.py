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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.general_utils import build_rotation

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def compute_rotation_loss(rot_quats, normals):
    print("Rotation Quats:\n", rot_quats)
    rot_quats = torch.nn.functional.normalize(rot_quats, p=2, dim=1)
    # 将四元数转换为旋转矩阵
    rot_mats = build_rotation(rot_quats)
    
    # 高斯椭球的z轴方向
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=rot_quats.device).unsqueeze(0)
    z_axis_rot = torch.matmul(rot_mats, z_axis.unsqueeze(2)).squeeze(2) # 其实也是旋转矩阵第三列
    
    print("Rotation Matrices:\n", rot_mats)
    print("Rotated Z Axes:\n", z_axis_rot)
    # 单位法向量
    normals = normals / torch.norm(normals, dim=1, keepdim=True)

    print("Normalized Normals:\n", normals)
    
    # 计算夹角
    dot_product = torch.sum(z_axis_rot * normals, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # 防止数值问题
    angles = torch.acos(dot_product)    # [0,pi]
    
    # 打印点积和夹角用于调试
    print("Dot Product:\n", dot_product)
    print("Angles:\n", angles)
    # 计算旋转一致性损失
    rotation_loss = torch.mean(angles)
    return rotation_loss
