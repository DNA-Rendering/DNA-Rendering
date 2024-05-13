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

######### for DNA-Rendering
import torch 
import cv2 
from smplx.body_models import SMPLX
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from SMCReader import SMCReader

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

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

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

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
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
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
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
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
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
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

def readCamerasDNARendering(path, info_dict, white_background, image_scaling=0.5, return_smplx=False):
    output_view = info_dict["views"]
    frame_idx = info_dict["frame_idx"]
    ratio = image_scaling
    
    cam_infos = []
    main_file = path 
    annot_file = path.replace('main', 'annotations').split('.')[0] + '_annots.smc'
    main_reader = SMCReader(main_file)
    annot_reader = SMCReader(annot_file)

    smplx_vertices = None 
    if return_smplx:
        gender = main_reader.actor_info['gender']
        model = SMPLX(
            'assets/body_models/smplx/', smpl_type='smplx',
            gender=gender, use_face_contour=True, flat_hand_mean=False, use_pca=False, 
            num_betas=10, num_expression_coeffs=10, ext='npz'
        )
        smplx_dict = annot_reader.get_SMPLx(Frame_id=frame_idx)
        betas = torch.from_numpy(smplx_dict["betas"]).unsqueeze(0).float()
        expression = torch.from_numpy(smplx_dict["expression"]).unsqueeze(0).float()
        fullpose = torch.from_numpy(smplx_dict["fullpose"]).unsqueeze(0).float()
        translation = torch.from_numpy(smplx_dict['transl']).unsqueeze(0).float()
        output = model(
            betas=betas, 
            expression=expression,
            global_orient = fullpose[:, 0].clone(),
            body_pose = fullpose[:, 1:22].clone(),
            jaw_pose = fullpose[:, 22].clone(),
            leye_pose = fullpose[:, 23].clone(),
            reye_pose = fullpose[:, 24].clone(),
            left_hand_pose = fullpose[:, 25:40].clone(), 
            right_hand_pose = fullpose[:, 40:55].clone(),
            transl = translation,
            return_verts=True)
        smplx_vertices = output.vertices.detach().cpu().numpy().squeeze()
        
    parent_dir = os.path.dirname(os.path.dirname(path))
    out_img_dir = os.path.join(parent_dir, "images")
    # os.makedirs(out_img_dir, exist_ok=True) 
    bg = np.array([255, 255, 255]) if white_background else np.array([0, 0, 0])
    idx = 0   
    for view_index in output_view:
        # Load K, R, T
        cam_params = annot_reader.get_Calibration(view_index)
        K = cam_params['K']
        D = cam_params['D'] # k1, k2, p1, p2, k3
        RT = cam_params['RT']

        # Load image, mask
        image = main_reader.get_img('Camera_5mp', view_index, Image_type='color', Frame_id=frame_idx)
        image = cv2.undistort(image, K, D)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        mask = annot_reader.get_mask(view_index, Frame_id=frame_idx)
        mask = cv2.undistort(mask, K, D)
        mask = mask[..., np.newaxis].astype(np.float32) / 255.0
        image = image * mask + bg * (1.0 - mask)
        
        c2w = np.array(RT, dtype=np.float32)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3].copy()
        if ratio != 1.0:
            H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            K[:2] = K[:2] * ratio

        H, W, _ = image.shape
        focalX = K[0,0]
        focalY = K[1,1]
        FovX = focal2fov(focalX, W)
        FovY = focal2fov(focalY, H)

        image = Image.fromarray(np.array(image, dtype=np.byte), "RGB")
        image_name = "%04d" % view_index
        image_path = os.path.join(out_img_dir, "%s.png" % image_name)

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=W, height=H))
            
        idx += 1
    
    return cam_infos, smplx_vertices

def readDNARenderingInfo(path, white_background, eval):

    test_view_arr = [3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 27, 29, 31, 33, 35, 39, 43, 45]
    train_view_arr = [x for x in range(48) if x not in test_view_arr]
    train_info_dict = {
        "views": train_view_arr,
        "frame_idx": 1,
    }
    test_info_dict = {
        "views": test_view_arr,
        "frame_idx": 1,
    }
    print("Reading Training Transforms", flush=True)
    train_cam_infos, smplx_vertices = readCamerasDNARendering(path, train_info_dict, white_background, return_smplx=True)
    print("Reading Test Transforms", flush=True)
    test_cam_infos, _ = readCamerasDNARendering(path, test_info_dict, white_background, return_smplx=False)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    parent_dir = os.path.dirname(os.path.dirname(path))
    ply_path = os.path.join(parent_dir, "points3d.ply")
    print("Using SMPLX vertices to initiate", flush=True)
    num_pts, _ = smplx_vertices.shape 
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=smplx_vertices, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    storePly(ply_path, smplx_vertices, SH2RGB(shs) * 255)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "DNA-Rendeing": readDNARenderingInfo,
}