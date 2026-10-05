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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixShift

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, mask, gt_alpha_mask, # add mask
                 image_name, uid, resolution_scale, intrinsic, b_pp_is_centered, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.raw_mask = mask
        self.is_masked = None
        if mask is not None:    # mask当中值为 0 的区域转化为True
            self.is_masked = (mask == 0).expand(*image.shape)  # True represent masked pixel

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # for calculating the projection_matrix when principle point is not lay on the center of image
        
        # print(f"the image width{self.image_width}")
        # print(f"the image height{self.image_height}")
        # print(f"the intrinsic: {intrinsic}")
        new_intrinsic = tuple(x / resolution_scale for x in intrinsic)
        focal_x, focal_y, cx, cy = new_intrinsic
        # print(f"the intrinsic after rescale: {new_intrinsic}")
        self.b_pp_is_centered = b_pp_is_centered
        
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # choose different method to calculating projection_matrix according whether the principle point is on the center of image  
        if self.b_pp_is_centered:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        else:
            # print(f"the principle point is not on the center of picture, caculating projection_matrix with new method!!!")
            self.projection_matrix = getProjectionMatrixShift(znear=self.znear, zfar=self.zfar, focal_x=focal_x, focal_y=focal_y, cx=cx, cy=cy, width=self.image_width, height=self.image_height, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

