# general
import argparse, pathlib, datetime, tqdm, sys, os, random, pickle, yaml
import numpy as np
from ipdb import set_trace as st
from scipy.spatial.transform import Rotation as R

import matplotlib
from skimage import color, io

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors

import torch
import torch.nn.functional as F
from lpips import LPIPS
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__)))
from nerfacc.nerfacc.estimators.occ_grid import OccGridEstimator

from models.datasets.utils import Rays
from models.utils import (
    render_image_with_occgrid,
    render_image_with_occgrid_test,
    render_image_with_occgrid_with_depth_guide,
    render_probablistic_image_with_occgrid_test,
)
from radiance_fields.ngp import NGPRadianceField

from data_proc.habitat_to_data import Dataset
from data_proc.depth_to_grid import generate_ray_casting_grid_map, Bresenham3D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat-config-file",
        type=str,
        default=str(
            pathlib.Path.cwd()
            / "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"
        ),
        help="scene_dataset_self.config_file",
    )
    return parser.parse_args()


class ActiveNeRFMapper:
    def __init__(self, args) -> None:
        print("Parameters Loading")
        # initialize radiance field, estimator, optimzer, and datasetWWW

        with open(
            f"perception/nerf_scripts/configs/config_" + args["map_name"] + ".yaml", "r"
        ) as f:
            self.config_file = yaml.safe_load(f)

        self.save_path = (
            self.config_file["save_path"]
            + "/"
            + "s%d"%(args["seed"])+"_"
            + args["target_mode"] +"_"
            + "nerf_"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        self.learning_rate_lst = []

        # scene parameters
        self.aabb = torch.tensor(
            self.config_file["aabb"], device=self.config_file["cuda"]
        )

        # model parameters
        self.main_grid_resolution = (
            (
                (self.aabb.cpu().numpy()[3:] - self.aabb.cpu().numpy()[:3])
                / self.config_file["main_grid_size"]
            )
            .astype(int)
            .tolist()
        )

        self.cost_map = np.full(
            (self.main_grid_resolution[0], self.main_grid_resolution[2]), 0.5
        )
        self.visiting_map = np.zeros(self.cost_map.shape)

        self.minor_grid_resolution = (
            (
                (self.aabb.cpu().numpy()[3:] - self.aabb.cpu().numpy()[:3])
                / self.config_file["minor_grid_size"]
            )
            .astype(int)
            .tolist()
        )

        self.trajector_uncertainty_list = [
            [] for _ in range(self.config_file["planning_step"])
        ]

        self.policy_type = "uncertainty"  # "uncertainty", "random", "spatial"

        if self.policy_type == "random":
            self.config_file["num_traj"] = 1

        self.estimators = []
        self.radiance_fields = []
        self.optimizers = []
        self.grad_scalers = []
        self.schedulers = []
        self.binary_grid = None
        self.train_dataset = None
        self.test_dataset = None
        self.errors_hist = []

        self.sem_ce_ls = []

        self.sim_step = 0
        self.viz_save_path = self.save_path + "/viz/"

        for i in range(self.config_file["n_ensembles"]):
            estimator = OccGridEstimator(
                roi_aabb=self.aabb,
                resolution=self.main_grid_resolution,
                levels=self.config_file["main_grid_nlvl"],
            ).to(self.config_file["cuda"])

            radiance_field = NGPRadianceField(
                aabb=estimator.aabbs[-1],
                neurons=self.config_file["main_neurons"],
                layers=self.config_file["main_layer"],
                num_semantic_classes=0,
            ).to(self.config_file["cuda"])
            optimizer = torch.optim.Adam(
                radiance_field.parameters(),
                lr=1e-3,
                eps=1e-15,
                weight_decay=self.config_file["weight_decay"],
            )
            self.estimators.append(estimator)
            self.grad_scalers.append(torch.cuda.amp.GradScaler(2**10))
            self.radiance_fields.append(radiance_field)
            self.optimizers.append(optimizer)
            self.schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.CyclicLR(
                            optimizer,
                            base_lr=1e-4,
                            max_lr=1e-3,
                            step_size_up=int(self.config_file["training_steps"] / 4),
                            mode="exp_range",
                            gamma=1.0,  # 0.9999,
                            cycle_momentum=False,
                        )
                    ]
                )
            )

        self.lpips_net = LPIPS(net="vgg").to(self.config_file["cuda"])
        self.lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1

        self.focal = (
            0.5 * self.config_file["img_w"] / np.tan(self.config_file["hfov"] / 2)
        )

        cmap = plt.cm.tab20
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap1 = plt.cm.tab20b
        cmaplist1 = [cmap1(i) for i in range(cmap1.N)]

        cmaplist = (
            cmaplist
            + [cmaplist1[0]]
            + [cmaplist1[1]]
            + [cmaplist1[4]]
            + [cmaplist1[5]]
            + [cmaplist1[8]]
            + [cmaplist1[9]]
            + [cmaplist1[12]]
            + [cmaplist1[13]]
            + [cmaplist1[16]]
            + [cmaplist1[17]]
        )
        self.custom_cmap = matplotlib.colors.ListedColormap(cmaplist)

        # right and left rays angles to sample from ? must be aligned with depth dim
        r = np.arctan(np.linspace(0.5, 319.5, 160) / 160).tolist()
        r.reverse()
        l = np.arctan(-np.linspace(0.5, 319.5, 160) / 160).tolist()
        self.align_angles = np.array(r + l)

        self.global_origin = np.array(self.config_file["global_origin"])

        self.current_pose = np.array(self.config_file["global_origin"])

        print("Parameters Loaded")

    def initialization(self, data, test_data):
        print("initialization Started")

        sampled_images = np.array(data["images"])
        sampled_depth_images = np.array(data["depths"])
        sampled_poses_mat = np.array(data["poses"])

        test_sampled_images = np.array(test_data["images"])
        test_sampled_depth_images = np.array(test_data["depths"])
        test_sampled_poses_mat = np.array(test_data["poses"])

        self.train_dataset = Dataset(
            training=True,
            save_fp=self.save_path + "/train/",
            num_rays=self.config_file["init_batch_size"],
            num_models=self.config_file["n_ensembles"],
            # device=self.config_file["cuda"],
            device="cpu",
        )

        self.train_dataset.update_data(
            sampled_images,
            sampled_depth_images,
            sampled_poses_mat,
        )

        self.test_dataset = Dataset(
            training=False,
            save_fp=self.save_path + "/test/",
            num_models=self.config_file["n_ensembles"],
            # device=self.config_file["cuda"],
            device="cpu",
        )

        self.test_dataset.update_data(
            test_sampled_images,
            test_sampled_depth_images,
            test_sampled_poses_mat,
        )

        print("Initialization Finished")

    def nerf_training(
        self, steps, final_train=False, initial_train=False, planning_step=-1
    ):
        print("Nerf Training Started")

        if final_train:
            self.schedulers = []
            for i in range(self.config_file["n_ensembles"]):
                optimizer = self.optimizers[i]
                self.schedulers.append(
                    torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=[
                            int(steps * 0.3),
                            int(steps * 0.5),
                            int(steps * 0.8),
                        ],
                        gamma=0.1,
                    )
                )

        # num_test_images = self.test_dataset.size
        num_test_images = 20
        test_idx = np.arange(self.test_dataset.size)

        def occ_eval_fn(x):
            density = radiance_field.query_density(x)
            return density * self.config_file["render_step_size"]

        losses = [[], []]

        for step in tqdm.tqdm(range(steps)):
            # train and record the models in the ensemble
            ground_truth_imgs = []
            rendered_imgs = [[] for _ in range(num_test_images)]

            psnrs_lst = [[] for _ in range(num_test_images)]
            lpips_lst = [[] for _ in range(num_test_images)]

            ground_truth_depth = []
            depth_imgs = [[] for _ in range(num_test_images)]
            mse_dep_lst = [[] for _ in range(num_test_images)]

            # training each model
            for model_idx, (
                radiance_field,
                estimator,
                optimizer,
                scheduler,
                grad_scaler,
            ) in enumerate(
                zip(
                    self.radiance_fields,
                    self.estimators,
                    self.optimizers,
                    self.schedulers,
                    self.grad_scalers,
                )
            ):
                curr_device = (
                    self.config_file["cuda"]
                    if model_idx == 0
                    else self.config_file["cuda"]
                )
                radiance_field.train()
                estimator.train()

                c = np.random.random_sample()

                if c < 0.5 and not final_train and not initial_train:
                    # train with most recent batch of data
                    curr_idx = self.train_dataset.bootstrap(model_idx)
                    curr_idx = curr_idx[
                        curr_idx
                        >= self.train_dataset.size - self.config_file["sample_disc"]
                    ]
                    i = np.random.choice(curr_idx, 1).item()
                else:
                    curr_idx = self.train_dataset.bootstrap(model_idx)
                    i = np.random.choice(curr_idx, 1).item()

                data = self.train_dataset[i]
                render_bkgd = data["color_bkgd"].to(curr_device)
                ry = data["rays"]
                rays = Rays(
                    origins=ry.origins.to(curr_device),
                    viewdirs=ry.viewdirs.to(curr_device),
                )
                pixels = data["pixels"].to(curr_device)
                dep = data["dep"].to(curr_device)

                # update occupancy grid
                if planning_step == -1:
                    estimator.update_every_n_steps(
                        step=step,
                        occ_eval_fn=occ_eval_fn,
                        occ_thre=1e-3,
                    )
                elif planning_step == -10:
                    estimator.update_every_n_steps(
                        step=step,
                        occ_eval_fn=occ_eval_fn,
                        occ_thre=1e-2,
                    )
                elif planning_step < 5:
                    estimator.update_every_n_steps(
                        step=step,
                        occ_eval_fn=occ_eval_fn,
                        occ_thre=1e-3,
                    )
                else:
                    estimator.update_every_n_steps(
                        step=step,
                        occ_eval_fn=occ_eval_fn,
                        occ_thre=3e-3,
                    )

                (
                    rgb,
                    acc,
                    depth,
                    n_rendering_samples,
                ) = render_image_with_occgrid_with_depth_guide(
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=self.config_file["near_plane"],
                    render_step_size=self.config_file["render_step_size"],
                    render_bkgd=render_bkgd,
                    cone_angle=self.config_file["cone_angle"],
                    alpha_thre=self.config_file["alpha_thre"],
                    depth=dep,
                )

                if n_rendering_samples == 0:
                    continue

                if self.config_file["target_sample_batch_size"] > 0:
                    # dynamic batch size for rays to keep sample batch size constant.
                    num_rays = len(pixels)
                    num_rays = int(
                        num_rays
                        * (
                            self.config_file["target_sample_batch_size"]
                            / float(n_rendering_samples)
                        )
                    )
                    self.train_dataset.update_num_rays(min(2000, num_rays))

                # compute loss
                loss_rgb = F.smooth_l1_loss(rgb, pixels)
                loss_dep = F.smooth_l1_loss(depth, dep.unsqueeze(1))
                # loss_sem = F.cross_entropy(semantic, sem)

                # loss = loss_rgb * 10 + loss_dep / 5 #+ loss_sem / 2
                # loss = loss_rgb * 10 + loss_dep /100 # f 0.1 and 100
                loss = loss_rgb * 100 + loss_dep / 100  # f 0.5 and dep~scale of z 300

                losses[0].append(loss_rgb.detach().cpu().item() * 100)
                losses[1].append(loss_dep.detach().cpu().item() / 100)

                optimizer.zero_grad()
                loss.backward()

                flag = False
                for name, param in radiance_field.named_parameters():
                    if torch.sum(torch.isnan(param.grad)) > 0:
                        flag = True
                        break

                if flag:
                    optimizer.zero_grad()
                    print("step jumped")
                    continue
                else:
                    optimizer.step()
                    scheduler.step()

                if model_idx == 0 and step % 500:
                    self.learning_rate_lst.append(scheduler._last_lr)

            ## Save checkpoit for video
            if (step + 1) % 2000 == 0:
                print("start eval")
                radiance_field.eval()
                estimator.eval()

                psnrs = []
                lpips = []
                with torch.no_grad():
                    test_idx = test_idx[-num_test_images:]
                    for i in tqdm.tqdm(range(num_test_images)):
                        data = self.test_dataset[test_idx[i]]
                        render_bkgd = data["color_bkgd"].to(curr_device)
                        ry = data["rays"]
                        rays = Rays(
                            origins=ry.origins.to(curr_device),
                            viewdirs=ry.viewdirs.to(curr_device),
                        )
                        pixels = data["pixels"].to(curr_device)
                        dep = data["dep"].to(curr_device)

                        # rendering
                        (
                            rgb,
                            acc,
                            depth,
                            _,
                        ) = render_image_with_occgrid_test(
                            1024,
                            # scene
                            radiance_field,
                            estimator,
                            rays,
                            # rendering options
                            near_plane=self.config_file["near_plane"],
                            render_step_size=self.config_file["render_step_size"],
                            render_bkgd=render_bkgd,
                            cone_angle=self.config_file["cone_angle"],
                            alpha_thre=self.config_file["alpha_thre"],
                        )

                        lpips_fn = lambda x, y: self.lpips_net.to(curr_device)(
                            self.lpips_norm_fn(x), self.lpips_norm_fn(y)
                        ).mean()

                        mse = F.mse_loss(rgb, pixels)
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())
                        lpips.append(lpips_fn(rgb, pixels).item())

                        mse_dep = F.mse_loss(depth, dep.unsqueeze(2))
                        mse_dep_lst[i].append(mse_dep.item())
                        ground_truth_imgs.append(pixels.cpu().numpy())
                        rendered_imgs[i].append(rgb.cpu().numpy())

                        ground_truth_depth.append(dep.cpu().numpy())
                        depth_imgs[i].append(depth.cpu().numpy())
                        psnrs_lst[i].append(psnr.item())
                        lpips_lst[i].append(lpips_fn(rgb, pixels).item())

                # self.render(np.array([self.current_pose]))
                if not os.path.exists(self.save_path + "/checkpoints/"):
                    os.makedirs(self.save_path + "/checkpoints/")

                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

                checkpoint_path = (
                    self.save_path
                    + "/checkpoints/"
                    + "model_"
                    + str(current_time)
                    + ".pth"
                )
                save_dict = {
                    "occ_grid": self.estimators[0].binaries,
                    "model": self.radiance_fields[0].state_dict(),
                    "optimizer_state_dict": self.optimizers[0].state_dict(),
                }
                torch.save(save_dict, checkpoint_path)
                print("Saved checkpoints at", checkpoint_path)

                print("loss: ", np.mean(np.array(losses), axis=1))

                psnr_test = np.array(psnrs_lst)[:, 0]
                lpips_test = np.array(lpips_lst)[:, 0]
                depth_mse_test = np.array(mse_dep_lst)[:, 0]

                print("Mean PSNR: " + str(np.mean(psnr_test)))
                print("Mean LPIPS: " + str(np.mean(lpips_test)))
                print("Mean Depth MSE: " + str(np.mean(depth_mse_test)))
                self.errors_hist.append(
                    [
                        planning_step,
                        np.mean(psnr_test),
                        np.mean(lpips_test),
                        np.mean(depth_mse_test),
                    ]
                )
                # save psnr and stuff
                np.save(self.save_path + "/errors.npy", np.array(self.errors_hist))

    def update_cost_map(self, data):

        def update_cost_map_func(
            cost_map, depth, angle, g_loc, w_loc, aabb, resolution
        ):
            ox = np.sin(-angle) * depth + w_loc[0]
            oy = -np.cos(-angle) * depth + w_loc[2]
            (
                occupancy_map,
                min_x,
                max_x,
                min_y,
                max_y,
                xy_resolution,
            ) = generate_ray_casting_grid_map(
                ox,
                oy,
                cost_map.shape[0],
                cost_map.shape[1],
                g_loc[0],
                g_loc[2],
                aabb,
                resolution,
            )

            cost_map[occupancy_map > 0.9] = 1
            cost_map[occupancy_map < 0.1] = 0

            visiting_map = np.zeros(cost_map.shape)
            visiting_map[occupancy_map < 0.1] = 1

            return cost_map, visiting_map

        # sampled_images = np.array(data['images'])
        sampled_depth_images = np.array(data["depths"])
        sampled_poses_mat = np.array(data["poses"])

        for i, d_img in enumerate(sampled_depth_images):
            d_points = d_img[int(d_img.shape[0] / 2)]
            R_m = sampled_poses_mat[i][:3, :3]
            euler = R.from_matrix(R_m).as_euler("yzx")
            d_angles = (self.align_angles + euler[0]) % (2 * np.pi)
            w_loc = sampled_poses_mat[i][:3, 3]
            grid_loc = np.array(
                (w_loc - self.aabb.cpu().numpy()[:3])
                // self.config_file["main_grid_size"],
                dtype=int,
            )

            self.cost_map, visiting_map = update_cost_map_func(
                cost_map=self.cost_map,
                depth=d_points,
                angle=d_angles,
                g_loc=grid_loc,
                w_loc=w_loc,
                aabb=self.aabb.cpu().numpy(),
                resolution=self.config_file["main_grid_size"],
            )
            self.visiting_map += visiting_map

        return self.cost_map, self.visiting_map

    def probablistic_uncertainty(self, trajectory):  # , step):
        """uncertainty of each trajectory"""
        rendered_imgs = [[] for _ in range(self.config_file["n_ensembles"])]
        rendered_imgs_var = [[] for _ in range(self.config_file["n_ensembles"])]
        depth_imgs = [[] for _ in range(self.config_file["n_ensembles"])]
        depth_imgs_var = [[] for _ in range(self.config_file["n_ensembles"])]
        acc_imgs = [[] for _ in range(self.config_file["n_ensembles"])]
        num_sample = 40  # self.config_file["sample_disc"] + 5
        for model_idx, (radiance_field, estimator) in enumerate(
            zip(self.radiance_fields, self.estimators)
        ):
            curr_device = (
                self.config_file["cuda"] if model_idx == 0 else self.config_file["cuda"]
            )

            radiance_field.eval()
            estimator.eval()

            with torch.no_grad():
                scale = 0.5
                # scale = 1.
                # a = np.linspace(0, len(trajectory) - 20, 20)
                # b = np.linspace(len(trajectory) - 20, len(trajectory) - 1, 20)
                # unc_idx = np.hstack((a, b)).astype(int)
                (
                    rgb,
                    rgb_var,
                    depth,
                    depth_var,
                    acc,
                    # sem,
                ) = Dataset.render_probablistic_image_from_pose(
                    radiance_field,
                    estimator,
                    # trajectory[unc_idx],
                    trajectory,
                    self.config_file["img_w"],
                    self.config_file["img_h"],
                    self.focal,
                    self.config_file["near_plane"],
                    self.config_file["render_step_size"],
                    scale,
                    self.config_file["cone_angle"],
                    self.config_file["alpha_thre"],
                    4,
                    curr_device,
                )

                rendered_imgs[model_idx].append(rgb[-num_sample:])
                rendered_imgs_var[model_idx].append(rgb_var[-num_sample:])
                depth_imgs[model_idx].append(depth[-num_sample:])
                depth_imgs_var[model_idx].append(depth_var[-num_sample:])
                acc_imgs[model_idx].append(acc[-num_sample:])

        rendered_imgs = np.array(rendered_imgs)
        rendered_imgs_var = np.array(rendered_imgs_var)
        depth_imgs = np.array(depth_imgs)
        depth_imgs_var = np.array(depth_imgs_var)
        acc_imgs = np.array(acc_imgs)

        # rgb predictive information
        rgb_conditional_entropy = (
            np.log(2 * np.pi * np.e * rendered_imgs_var + 1e-4) / 2
        )
        rgb_mean_conditional_entropy = np.mean(rgb_conditional_entropy, axis=0)

        rgb_ensemble_variance = np.sum(rendered_imgs_var, axis=0) / 2
        rgb_entropy = np.log(2 * np.pi * np.e * rgb_ensemble_variance + 1e-4) / 2

        rgb_predictive_information = np.mean(rgb_entropy - rgb_mean_conditional_entropy)

        # depth predictive information
        depth_conditional_entropy = np.log(2 * np.pi * np.e * depth_imgs_var + 1e-4) / 2
        depth_mean_conditional_entropy = np.mean(depth_conditional_entropy, axis=0)

        depth_ensemble_variance = np.sum(depth_imgs_var, axis=0) / 2
        depth_entropy = np.log(2 * np.pi * np.e * depth_ensemble_variance + 1e-4) / 2

        depth_predictive_information = np.mean(
            depth_entropy - depth_mean_conditional_entropy
        )

        # occupancy entropy
        occ_conditional_entropy = -(acc_imgs + 1e-4) * np.log(acc_imgs + 1e-4) - (
            1 - acc_imgs + 1e-4
        ) * np.log(1 - acc_imgs + 1e-4)
        occ_mean_conditional_entropy = np.mean(occ_conditional_entropy, axis=0)

        occ_ensemble_p = np.mean(acc_imgs, axis=0)
        occ_entropy = -(occ_ensemble_p + 1e-4) * np.log(occ_ensemble_p + 1e-4) - (
            1 - occ_ensemble_p + 1e-4
        ) * np.log(1 - occ_ensemble_p + 1e-4)

        occ_predictive_information = np.mean(occ_entropy - occ_mean_conditional_entropy)

        # predictive_information = (
        #     rgb_predictive_information
        #     + depth_predictive_information
        #     # + sem_predictive_information * 3
        #     + occ_predictive_information * 2
        # )
        predictive_information = [
            rgb_predictive_information,
            depth_predictive_information,
            occ_predictive_information,
        ]

        # self.trajector_uncertainty_list[step - 1].append(
        #     [
        #         rgb_predictive_information,
        #         depth_predictive_information,
        #         # sem_predictive_information * 3,
        #         occ_predictive_information * 2,
        #     ]
        # )
        # print(
        #     rgb_predictive_information,
        #     depth_predictive_information,
        #     sem_predictive_information * 3,
        #     occ_predictive_information * 2,
        # )
        # print(predictive_information)
        return predictive_information

    def particles_seen_from_pose(self, points, pos):
        """
        points in the world (x,y,z) that are in the image frame
        pos of the agent (x,y,z)
        we convert voxel grid (xzy) -> (xyz)
        """
        # convert world points to grid points
        aabb_xyz = np.array(
            [
                self.aabb.cpu().numpy()[0],
                self.aabb.cpu().numpy()[2],
                self.aabb.cpu().numpy()[1],
            ]
        )

        v_idx = np.array(
            (pos - aabb_xyz) // self.config_file["main_grid_size"],
            dtype=int,
        )
        # p_loc = np.array([points[:,0], points[:,2], points[:,1]]).T
        p_grid_loc = np.array(
            (points - aabb_xyz)
            // self.config_file["main_grid_size"],
            dtype=int,
        )

        # get voxel grid
        voxel_grid = self.estimators[0].binaries
        voxel_grid = voxel_grid.cpu().numpy()
        vg = np.swapaxes(voxel_grid, 2, 3)

        voxel_grid1 = self.estimators[1].binaries
        voxel_grid1 = voxel_grid1.cpu().numpy()
        vg1 = np.swapaxes(voxel_grid1, 2, 3)

        v_merge = np.squeeze(vg.astype(np.int32) + vg1.astype(np.int32))
        occ_grid = (v_merge > 1e-4).astype(np.int32)
        # we'll 0 out cube around the agent to clear it up
        occ_grid[
            v_idx[0] - 1 : v_idx[0] + 2,
            v_idx[1] - 1 : v_idx[1] + 2,
            v_idx[2] - 1 : v_idx[2] + 2,
        ] = False

        # get points in grid that lie between current pos and points
        # and check if they collide, if they dont save the point
        valid_pts = []
        for p_idx in p_grid_loc:
            pts_list = Bresenham3D(v_idx, p_idx)
            for pts in pts_list:
                try:
                    if occ_grid[pts] == 1:
                        # valid_pts.append(p_idx)
                        break
                except:
                    break
                if pts == pts_list[-1]:
                    valid_pts.append(p_idx)

        try:
            w_valid_points = np.array(valid_pts) * self.config_file["main_grid_size"] + aabb_xyz
        except:
            w_valid_points = np.array([])

        return w_valid_points

    def render(self, data, mode="fpv/"):
        sampled_images = np.array(data["images"])
        sampled_depth_images = np.array(data["depths"])
        sampled_poses_mat = np.array(data["poses"])
        # traj1 = np.copy(traj)
        # traj2 = np.copy(traj)
        step = self.sim_step

        # render_images = np.array(self.sim.render_tpv(traj))
        if not os.path.exists(self.viz_save_path):
            os.makedirs(self.viz_save_path)
        # for img in render_images:
        #     cv2.imwrite(self.viz_save_path + str(self.sim_step) + ".png", img)
        #     self.sim_step += 1

        # render_images = np.array(self.sim.render_top_tpv(traj))
        if not os.path.exists(self.viz_save_path):
            os.makedirs(self.viz_save_path)
        if not os.path.exists(self.viz_save_path + "top/"):
            os.makedirs(self.viz_save_path + "top/")
        # for s, img in enumerate(render_images):
        # cv2.imwrite(self.viz_save_path + "top/" + str(step + s) + ".png", img)

        # fpv_path = self.viz_save_path + "fpv/"
        fpv_path = self.viz_save_path + mode
        if not os.path.exists(fpv_path):
            os.makedirs(fpv_path)
            os.makedirs(fpv_path + "gt_rgb/")
            os.makedirs(fpv_path + "gt_dep/")
            os.makedirs(fpv_path + "pd_rgb/")
            os.makedirs(fpv_path + "pd_dep/")
            os.makedirs(fpv_path + "pd_occ/")

        # (
        #     sampled_images,
        #     sampled_depth_images,
        #     # sampled_sem_images,
        # ) = self.sim.sample_images_from_poses(traj1)

        (
            rgb_predictions,
            depth_predictions,
            acc_predictions,
            # sem_predictions,
        ) = Dataset.render_image_from_pose(
            self.radiance_fields[0],
            self.estimators[0],
            sampled_poses_mat,
            self.config_file["img_w"],
            self.config_file["img_h"],
            self.focal,
            self.config_file["near_plane"],
            self.config_file["render_step_size"],
            1,
            self.config_file["cone_angle"],
            self.config_file["alpha_thre"],
            1,
            self.config_file["cuda"],
        )
        #
        for idx, (rgb, dep, rgb_pd, dep_pd, acc_pd) in enumerate(
            zip(
                sampled_images,
                sampled_depth_images,
                rgb_predictions,
                depth_predictions,
                acc_predictions,
            )
        ):

            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(
                fpv_path + "gt_rgb/" + str(step + idx) + ".png",
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                fpv_path + "pd_rgb/" + str(step + idx) + ".png",
                cv2.cvtColor(np.float32(rgb_pd * 255), cv2.COLOR_RGB2BGR),
            )

            cv2.imwrite(
                fpv_path + "gt_dep/" + str(step + idx) + ".png",
                # np.clip(dep, 0, 255),
                dep,
            )
            cv2.imwrite(
                fpv_path + "pd_dep/" + str(step + idx) + ".png",
                # np.clip(dep_pd, 0, 255),
                dep_pd,
            )

            cv2.imwrite(
                fpv_path + "pd_occ/" + str(step + idx) + ".png",
                np.clip(acc_pd * 255, 0, 255),
            )
            self.sim_step += 1

    def planning(self, steps, training_steps_per_step):
        print("Planning Thread Started")

        current_state = self.global_origin[:3]

        def occ_eval_fn(x):
            density = self.radiance_field.query_density(x)
            return density * self.config_file["render_step_size"]

        sim_step = 0

        step = 0
        flag = True
        while flag and step < self.config_file["planning_step"]:
            print("planning step: " + str(step))
            step += 1

            # get voxel grid
            voxel_grid = self.estimators[0].binaries
            voxel_grid = voxel_grid.cpu().numpy()
            vg = np.swapaxes(voxel_grid, 2, 3)

            voxel_grid1 = self.estimators[1].binaries
            voxel_grid1 = voxel_grid1.cpu().numpy()
            vg1 = np.swapaxes(voxel_grid1, 2, 3)

            print("sampling trajectory from: " + str(current_state))

            xyz_state = np.copy(current_state)
            xyz_state[1] = current_state[2]
            xyz_state[2] = current_state[1]

            aabb = np.copy(self.aabb.cpu().numpy())
            aabb[1] = self.aabb[2]
            aabb[2] = self.aabb[1]
            aabb[4] = self.aabb[5]
            aabb[5] = self.aabb[4]

            N_sample_traj_pose = sample_traj(
                voxel_grid=np.array([vg, vg1]),
                current_state=xyz_state,
                N_traj=self.config_file["num_traj"],
                aabb=aabb,
                sim=self.sim,
                cost_map=self.cost_map,
                N_sample_disc=self.config_file["sample_disc"],
                voxel_grid_size=self.config_file["main_grid_size"],
                visiting_map=self.visiting_map,
                save_path=self.save_path,
            )
            copy_traj = N_sample_traj_pose.copy()

            if self.policy_type == "uncertainty":
                uncertainties = []
                for i in tqdm.tqdm(range(self.config_file["num_traj"])):
                    uncertainty = self.probablistic_uncertainty(
                        N_sample_traj_pose[i], step
                    )
                    uncertainties.append(uncertainty)

                best_index = np.argmax(np.array(uncertainties))

                a = np.linspace(0, len(N_sample_traj_pose[best_index]) - 20, 20)
                b = np.linspace(
                    len(N_sample_traj_pose[best_index]) - 20,
                    len(N_sample_traj_pose[best_index]) - 1,
                    20,
                )
                unc_idx = np.hstack((a, b)).astype(int)

                (
                    sampled_images,
                    sampled_depth_images,
                    sampled_sem_images,
                ) = self.sim.sample_images_from_poses(
                    N_sample_traj_pose[best_index][unc_idx]
                )

                sampled_images = sampled_images[:, :, :, :3]

                self.render(copy_traj[best_index])
                self.current_pose = copy_traj[best_index][-1]

                sampled_poses_mat = []
                for pose in N_sample_traj_pose[best_index][unc_idx]:
                    T = np.eye(4)
                    T[:3, :3] = R.from_quat(pose[3:]).as_matrix()
                    T[:3, 3] = pose[:3]
                    sampled_poses_mat.append(T)

                for i, (mat, d_img) in enumerate(
                    zip(sampled_poses_mat[-6:], sampled_depth_images[-6:])
                ):
                    d_points = d_img[int(d_img.shape[0] / 2)]
                    R_m = mat[:3, :3]
                    euler = R.from_matrix(R_m).as_euler("yzx")
                    d_angles = (self.align_angles + euler[0]) % (2 * np.pi)
                    w_loc = mat[:3, 3]
                    grid_loc = np.array(
                        (w_loc - self.aabb.cpu().numpy()[:3])
                        // self.config_file["main_grid_size"],
                        dtype=int,
                    )

                    self.cost_map, visiting_map = update_cost_map(
                        cost_map=self.cost_map,
                        depth=d_points,
                        angle=d_angles,
                        g_loc=grid_loc,
                        w_loc=w_loc,
                        aabb=self.aabb.cpu().numpy(),
                        resolution=self.config_file["main_grid_size"],
                    )
                    self.visiting_map += visiting_map

                self.train_dataset.update_data(
                    sampled_images,
                    sampled_depth_images,
                    sampled_sem_images,
                    sampled_poses_mat,
                )

                current_state = N_sample_traj_pose[best_index][unc_idx][-1, :3]
            elif self.policy_type == "random":
                uncertainty, mid = self.trajector_uncertainty(
                    N_sample_traj_pose[0], step
                )

                (
                    sampled_images,
                    sampled_depth_images,
                    sampled_sem_images,
                ) = self.sim.sample_images_from_poses(N_sample_traj_pose[0])
                best_index = 0

                sampled_images = sampled_images[:, :, :, :3]

                self.render(N_sample_traj_pose[best_index][1:])

                sampled_poses_mat = []
                for pose in N_sample_traj_pose[best_index]:
                    T = np.eye(4)
                    T[:3, :3] = R.from_quat(pose[3:]).as_matrix()
                    T[:3, 3] = pose[:3]
                    sampled_poses_mat.append(T)

                for i, d_img in enumerate(sampled_depth_images):
                    d_points = d_img[int(d_img.shape[0] / 2)]
                    R_m = sampled_poses_mat[i][:3, :3]
                    euler = R.from_matrix(R_m).as_euler("yzx")
                    d_angles = (self.align_angles + euler[0]) % (2 * np.pi)
                    w_loc = sampled_poses_mat[i][:3, 3]
                    grid_loc = np.array(
                        (w_loc - self.aabb.cpu().numpy()[:3])
                        // self.config_file["main_grid_size"],
                        dtype=int,
                    )
                    self.cost_map = update_cost_map(
                        self.cost_map,
                        d_points,
                        d_angles,
                        grid_loc,
                        self.aabb.cpu().numpy(),
                        self.config_file["main_grid_size"],
                    )

                self.train_dataset.update_data(
                    sampled_images,
                    sampled_depth_images,
                    sampled_sem_images,
                    sampled_poses_mat,
                )

                current_state = N_sample_traj_pose[best_index][-1, :3]

                self.current_pose = N_sample_traj_pose[best_index][-1]

            elif self.policy_type == "spatial":
                (
                    sampled_images,
                    sampled_depth_images,
                    sampled_sem_images,
                ) = None

            print("plan finished at: " + str(current_state))

            self.nerf_training(training_steps_per_step, planning_step=step)

            past_unc = np.array(self.trajector_uncertainty_list[:step]).astype(float)

            unc = np.max(np.mean(past_unc, axis=2), axis=1)
            if step >= 5:
                if (
                    unc[step - 1] > 0.05
                    and unc[step - 2] > 0.05
                    and unc[step - 3] > 0.05
                    and unc[step - 4] > 0.05
                    and unc[step - 5] > 0.05
                ):
                    flag = False

    def pipeline(self):
        self.initialization()

        self.nerf_training(self.config_file["training_steps"])

        # self.planning(
        #     self.config_file["planning_step"], int(self.config_file["training_steps"])
        # )

        self.nerf_training(
            self.config_file["training_steps"] * 2, final_train=True, planning_step=-10
        )

        plt.plot(np.arange(len(self.learning_rate_lst)), self.learning_rate_lst)
        plt.savefig(self.save_path + "/learning_rate.png")

        plt.yscale("log")
        plt.plot(np.arange(len(self.learning_rate_lst)), self.learning_rate_lst)
        plt.savefig(self.save_path + "/learning_rate_log.png")

        # save radiance field, estimator, and optimzer
        print("Saving Models")
        # save_model(radiance_field, estimator, "test")

        self.train_dataset.save()
        self.test_dataset.save()

        if not os.path.exists(self.save_path + "/checkpoints/"):
            os.makedirs(self.save_path + "/checkpoints/")

        # self.trajector_uncertainty_list = np.array(self.trajector_uncertainty_list)
        # np.save(self.save_path + "/uncertainty.npy", self.trajector_uncertainty_list)

        self.errors_hist = np.array(self.errors_hist)
        np.save(self.save_path + "/errors.npy", self.errors_hist)

        for i, (radiance_field, estimator, optimizer, scheduler) in enumerate(
            zip(self.radiance_fields, self.estimators, self.optimizers, self.schedulers)
        ):
            checkpoint_path = (
                self.save_path + "/checkpoints/" + "model_" + str(i) + ".pth"
            )
            save_dict = {
                "occ_grid": estimator.binaries,
                "model": radiance_field.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(save_dict, checkpoint_path)
            print("Saved checkpoints at", checkpoint_path)


if __name__ == "__main__":
    torch.cuda.empty_cache
    args = parse_args()

    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)

    mapper = ActiveNeRFMapper(args)
    mapper.pipeline()
