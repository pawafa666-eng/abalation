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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_rotation_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.epoch_metrics import EpochMetrics
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import time
from utils.visualize_utils import save_tensor_3NN_as_image, save_points_projected_view
from lpipsPyTorch.modules.lpips import LPIPS
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    train_camera_count = len(scene.getTrainCameras())
    epoch_filename = (f"epoch_metrics_from_{first_iter}.csv" if first_iter else "epoch_metrics.csv")
    epoch_metrics = EpochMetrics(os.path.join(dataset.model_path, epoch_filename), train_camera_count, first_iter)
    print(f"Training cameras: {train_camera_count}; target iterations: {opt.iterations}; "
          f"target epochs: {opt.iterations / train_camera_count:.4f}")
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    save_image_bool = False
    save_pcd_image_bool = False

    # Initialize LPIPS model
    lpips_model = LPIPS(net_type='alex').to("cuda")

    # loss, psnr, ssim, lpips result files
    loss_file_path = os.path.join(dataset.model_path, "loss_results.txt")
    psnr_file_path = os.path.join(dataset.model_path, "psnr_results.txt")
    ssim_file_path = os.path.join(dataset.model_path, "ssim_results.txt")
    lpips_file_path = os.path.join(dataset.model_path, "lpips_results.txt")

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        # for mask
        mask = viewpoint_cam.is_masked
        if mask is not None:
            mask = mask.cuda()
            gt_image[mask] = image.detach()[mask]
        
        # for debug, 鎵撳嵃淇濆瓨鍘熷鐐逛簯鎶曞奖鍒扮浉鏈虹敾闈㈢殑鍥惧儚銆佹覆鏌撳浘鍍忋€丟T鍥惧儚
        # if 0 < iteration <= 300 and iteration % 25 == 0:
        #     print(f"save picture and pointcloud view when iteration {iteration}..... ")
        #     init_pcd = scene.get_init_scene().point_cloud
        #     save_points_projected_view(init_pcd=init_pcd, view_matrix=viewpoint_cam.world_view_transform, proj_matrix=viewpoint_cam.full_proj_transform, image_width=int(viewpoint_cam.image_width),
        #                                 image_height=int(viewpoint_cam.image_height), output_image_path=f"data/debug/{viewpoint_cam.image_name}_point.jpg")
        #     save_tensor_3NN_as_image(gt_image, f"data/debug/{viewpoint_cam.image_name}_gt.jpg")
        #     save_tensor_3NN_as_image(image, f"data/debug/{viewpoint_cam.image_name}_rendered.jpg")
        #     save_pcd_image_bool = True

        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # 淇濇寔涓?d楂樻柉鍜岀偣浜戞硶绾垮厛楠岀殑loss锛屽緟璋冭瘯鏉冮噸; 鐩墠鏄€氳繃娓呴浂姊害瀹炵幇锛岃167琛?        # print("鍘焞oss: ", loss)
        # rotation_loss = compute_rotation_loss(gaussians._rotation, gaussians._normals)
        # print("鏃嬭浆loss: ", rotation_loss)
        # lambda_rotation = 0.1
        # print("鎬籰oss: ", loss)



        # 涓嶉€忔槑搴oss锛岄紦鍔盙S涓嶉€忔槑搴︽帴杩戜簬1
        opac_ = gaussians.get_opacity
        loss_opac = (torch.exp(-(opac_)**2 * 5)).mean()
        lambda_opac = 0.01
        loss += lambda_opac * loss_opac

        #鏂板L2姝ｅ垯鍖栨崯澶?

        loss.backward()

        iter_end.record()
        iter_end.synchronize()

        with torch.no_grad():
            psnr_value = psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
            epoch_row = epoch_metrics.add(iteration, loss.item(), psnr_value)
            if epoch_row is not None:
                tqdm.write(f"[Epoch {epoch_row['epoch_progress']:.4f}] "
                           f"mean loss={epoch_row['mean_loss']:.7f}, "
                           f"mean PSNR={epoch_row['mean_psnr_db']:.4f}")
            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}", "Epoch": f"{iteration/train_camera_count:.2f}"})
                progress_bar.update(10)

                image_name = viewpoint_cam.image_name
                ssim_value = ssim(image, gt_image).item()
                lpips_value = lpips_model(image.unsqueeze(0) * 2 - 1, gt_image.unsqueeze(0) * 2 - 1).item()
                save_metrics(iteration, loss, Ll1, loss_opac, psnr_value, ssim_value, lpips_value, image_name, loss_file_path, psnr_file_path, ssim_file_path, lpips_file_path)

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if (not opt.disable_densification) and iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                if opt.freeze_gaussian_positions and gaussians._xyz.grad is not None:
                    gaussians._xyz.grad *= 0.0

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    epoch_metrics.flush()  # Preserve a final incomplete camera pass with its actual sample count.

def save_metrics(iteration, loss, Ll1, loss_opac, psnr_value, ssim_value, lpips_value, image_name, loss_file_path, psnr_file_path, ssim_file_path, lpips_file_path):
    """
    淇濆瓨鎹熷け銆丳SNR銆丼SIM鍜孡PIPS鍒版枃浠?    """
    # 淇濆瓨鎹熷け缁撴灉
    with open(loss_file_path, "a") as loss_file:
        loss_file.write(f"{iteration} {loss.item()} {Ll1.item()} {loss_opac.item()} {image_name}\n")

    # 淇濆瓨PSNR缁撴灉
    with open(psnr_file_path, "a") as psnr_file:
        psnr_file.write(f"{iteration} {psnr_value} {image_name}\n")

    # 淇濆瓨SSIM缁撴灉
    with open(ssim_file_path, "a") as ssim_file:
        ssim_file.write(f"{iteration} {ssim_value} {image_name}\n")

    # 淇濆瓨LPIPS缁撴灉
    with open(lpips_file_path, "a") as lpips_file:
        lpips_file.write(f"{iteration} {lpips_value} {image_name}\n")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_value = l1_loss(image, gt_image).item()
                    psnr_value = psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
                    print("\n[ITER {}] Evaluating {}/{}: L1 {} PSNR {}".format(
                        iteration, config['name'], viewpoint.image_name, l1_value, psnr_value
                    ))
                    if tb_writer:
                        view_tag = config['name'] + '/view_{}'.format(viewpoint.image_name)
                        tb_writer.add_scalar(view_tag + '/loss_viewpoint - l1_loss', l1_value, iteration)
                        tb_writer.add_scalar(view_tag + '/loss_viewpoint - psnr', psnr_value, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[150_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
