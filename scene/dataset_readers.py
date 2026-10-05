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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    mask: np.array  # for mask
    image_path: str
    mask_path: str  # for mask
    image_name: str
    width: int
    height: int
    intrinsic: tuple
    b_pp_is_centered: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

# def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, masks_folder):     # for mask
    cam_infos = []

    mask_count = 0

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        # sys.stdout.write("Reading camera {}/{}\n".format(idx+1, len(cam_extrinsics)))
        # sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            # for pp_center
            cx = intr.params[2]
            cy = intr.params[3]
            intrinsic = focal_length_x, focal_length_y, cx, cy
            thres = 1e-1  # min 
            if abs(cx - width/2) < thres and abs(cy - height/2) < thres:
                b_pp_is_centered = True
            else:
                b_pp_is_centered = False
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, extr.name)
        # image_name = os.path.basename(image_path).split(".")[0] # source code only support image_name which just has one "." 
        # image_name = os.path.basename(image_path).rsplit('.', 1)[0]
        image_name = os.path.splitext(extr.name)[0]   # for mask         1664177515.874827
        image = Image.open(image_path)
        
        # for mask
        mask = None
        mask_path = None
        if masks_folder is not None and masks_folder != "":
            # print(f"extr.name: {extr.name}")
            possible_mask_path = os.path.join(masks_folder, "{}.mask.png".format(os.path.splitext(extr.name)[0]))
            if os.path.exists(possible_mask_path):
                # print("finded mask..............")
                mask = Image.open(possible_mask_path)
                assert mask.size == image.size, "image({}) dimension {} doesn't match to the mask({}) {}".format(
                    image_name,
                    image.size,
                    possible_mask_path,
                    mask.size,
                )
                mask_path = possible_mask_path
                mask_count += 1
            else:
                print(f"masks path {possible_mask_path} is not exist!")

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, mask=mask, mask_path=mask_path, # add for mask
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              intrinsic=intrinsic, b_pp_is_centered=b_pp_is_centered)
        cam_infos.append(cam_info)

    if masks_folder != "":  # for mask
        sys.stdout.write('\n')
        sys.stdout.write("Read {} masks".format(mask_count))

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    is_ground = np.asarray(vertices['is_ground'], dtype=np.uint8)
    return BasicPointCloud(points=positions, colors=colors, normals=normals, is_ground=is_ground)

def randomPointCloudFromCameras(cam_infos, num_pts):
    centers = []
    for cam in cam_infos:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        centers.append(C2W[:3, 3])

    if centers:
        centers = np.asarray(centers, dtype=np.float32)
        xyz_min = centers.min(axis=0) - 1.3
        xyz_max = centers.max(axis=0) + 1.3
        xyz = np.random.random((num_pts, 3)).astype(np.float32) * (xyz_max - xyz_min) + xyz_min
    else:
        xyz = np.random.random((num_pts, 3)).astype(np.float32) * 2.6 - 1.3

    shs = np.random.random((num_pts, 3)).astype(np.float32) / 255.0
    colors = SH2RGB(shs)
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (num_pts, 1))
    is_ground = np.zeros((num_pts, 1), dtype=np.uint8)
    return BasicPointCloud(points=xyz, colors=colors, normals=normals, is_ground=is_ground)

def storePly(path, xyz, rgb, normals, is_ground):
    # Check if all inputs are numpy arrays
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    normals = np.asarray(normals, dtype=np.float32)
    is_ground = np.asarray(is_ground, dtype=np.uint8)

    # Verify shapes
    if not (xyz.shape[0] == rgb.shape[0] == normals.shape[0] == is_ground.shape[0]):
        raise ValueError("The number of elements in xyz, rgb, normals, and is_ground must match.")

    # Define the dtype for the structured array, including the new 'is_ground' field
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
             ('is_ground', 'u1')]  # f4: float32; u1: uint8

    # Create an empty structured array with the specified dtype
    elements = np.empty(xyz.shape[0], dtype=dtype)

    # Concatenate xyz, normals, rgb, and is_ground into a single array
    attributes = np.concatenate((xyz, normals, rgb, is_ground), axis=1)

    # Convert each row to a tuple and assign to the structured array
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=5, masks=None, init_point_cloud="lidar", random_points=100000):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        masks_folder=masks,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    init_point_cloud = init_point_cloud.lower()
    if init_point_cloud not in ("lidar", "random"):
        raise ValueError("--init_point_cloud must be either 'lidar' or 'random'")

    if init_point_cloud == "lidar":
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        is_ground_path = os.path.join(path, "sparse/0/is_ground.txt")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, normal, is_ground = read_points3D_binary(bin_path, is_ground_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
                normal = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (xyz.shape[0], 1))
                is_ground = np.zeros((xyz.shape[0], 1), dtype=np.uint8)
            storePly(ply_path, xyz, rgb, normal, is_ground)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    else:
        ply_path = os.path.join(path, "sparse/0/random_points3D.ply")
        pcd = randomPointCloudFromCameras(train_cam_infos, random_points)
        storePly(ply_path, pcd.points, pcd.colors * 255, pcd.normals, pcd.is_ground)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        normals = np.zeros((num_pts, 3))
        is_ground = np.zeros((num_pts, 1))
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=normals, is_ground=is_ground)

        storePly(ply_path, xyz, SH2RGB(shs) * 255, normals, is_ground)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
