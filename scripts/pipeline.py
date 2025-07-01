import numpy as np
from copy import deepcopy
from ipdb import set_trace as st
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random, click
import sys, pickle, os, datetime, copy
import seaborn as sns
from scipy.signal import argrelextrema
import cv2

sns.set_theme()
fsz = 12
plt.rc("font", size=fsz)
plt.rc("axes", titlesize=fsz)
plt.rc("axes", labelsize=fsz)
plt.rc("xtick", labelsize=fsz)
plt.rc("ytick", labelsize=fsz)
plt.rc("legend", fontsize=0.7 * fsz)
plt.rc("figure", titlesize=fsz)
plt.rc("pdf", fonttype=42)
sns.set_style("ticks", rc={"axes.grid": True})
# plt.rcParams["figure.figsize"] = [6, 4]

from agents import scout
from target import target
from pf import particle_filter
from utils import init_env, create_graph

sys.path.append("simulator")
from mgl_imgui_simOSM import mgl_imgui_simOSM as sim


class pipeline:
    def __init__(sf, seed, cfg, mapDim):
        sf.setup(seed)
        sf.cfg = cfg
        sf.mapDim = mapDim
        sf.num_agents = sf.cfg["num_agents"]
        sf.num_targets = sf.cfg["num_targets"]
        sf.setup_agents()
        sf.setup_targets()
        sf.save = sf.cfg["save"]

        # sf.fig = [plt.figure() for i in range(sf.num_agents)]
        sf.metric = {"targ-%d" % i: [] for i in range(sf.num_targets)}
        sf.metric["nerf_I"] = []
        sf.metric["PI"] = []
        sf.metric["chosen_idx"] = []
        sf.metric["%part_seen"] = []
        sf.metric["traj_len"] = []
        sf.fig_metric = plt.figure(2)
        sf.fig = plt.figure(1)
        path = "/data/activescout/data_gtmap/"+sf.cfg['map_name']+ "/"
        sf.save_path = path+"s%d_"%(seed)+sf.cfg["target_mode"]+"_"+sf.cfg["method"]+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(sf.save_path):
            os.makedirs(sf.save_path)
            os.makedirs(sf.save_path + "/gt_rgb/")

    def setup_agents(sf):
        sf.agents = [
            scout(
                sf.cfg,
                pf=particle_filter(sf.cfg, sf.mapDim, sf.num_targets),
                sim=sim(sf.cfg, headless=False),
            )
            for i in range(sf.num_agents)
        ]
        sf.a_traj = {
            "agent-%d" % i: sf.agents[i].x0.reshape(1, 5) for i in range(sf.num_agents)
        }
        sf.a_traj['obs'] = []
        # used for image saving counter, could update the name
        sf.step_count = -1

    def setup_targets(sf):
        try:
            G = pickle.load(open(sf.cfg["graph_file"], "rb"))
        except:
            G = create_graph(
                sf.cfg,
                pf=particle_filter(sf.cfg, sf.mapDim, sf.num_targets),
                save=True,
                render=True,
            )
        sf.targets = [target(sf.cfg, sf.mapDim, G=G) for i in range(sf.num_targets)]
        sf.t_traj = {
            "targ-%d" % i: sf.targets[i].x0[:3].reshape(1, 3)
            for i in range(sf.num_targets)
        }
        sf.t_traj["weights"] = []

    def reset(sf):
        for ii in range(sf.num_agents):
            sf.agents[ii].reset()

        for jj in range(sf.num_targets):
            sf.targets[jj].reset()

        sf.warmup_nerf()

    def warmup_nerf(sf):
        """
        First we will do some data collection for test locations
        Then we rollout scout survey trajectoy, this is for train dataset
        """
        for ii in range(sf.num_agents):
            traj = sf.agents[ii].survey_traj()
            data = sf.rollout(ii, traj)

    # @profile
    def step(sf):
        sf.step_count += 1

        # lets move the agents. traj output should be 10, i.e. N_sample_disc= 9+1
        for ii in range(sf.num_agents):
            # hallucinate and pick your trajectory
            traj, I, chosen_idx = sf.agents[ii].step()
            data = sf.rollout(ii, traj)
            # particles seen over a trajectory reset otherwise it is over the entire episode
            if sf.cfg['target_mode'] == 'active':
                if sf.step_count%10 == 0:
                    sf.agents[ii].part_seen = []

            # sf.metric["nerf_I"].append(unc)
            sf.metric["PI"].append(I)
            sf.metric["chosen_idx"].append(chosen_idx)
            sf.metric["traj_len"].append(np.linalg.norm(traj[0, :3] - traj[-1, :3]))

    def rollout(sf, ii, traj):
        """
        ii is the agent idx
        traj is the agent trajectory to follow
        targ_pos is a list of target trajectories to follow
        """
        # target finding trajectory based on what agent sampled and can't see
        targ_pos = []
        for jj in range(sf.num_targets):
            # move to places the agent cannot see
            not_seen = np.squeeze(
                np.array([sf.agents[i].pf.p_targ for i in range(sf.num_agents)])
            )
            tpos = sf.targets[jj].step(not_seen)

            # this number should agree with minsnap N_sample_disc
            # if traj is shorter than we just stay in place of the last pos
            if len(tpos) < len(traj):
                targ_pos.append(
                    np.vstack((tpos, np.tile(tpos[-1, :], (len(traj) - len(tpos), 1))))
                )
            else:
                targ_pos.append(tpos)
        targ_pos = np.array(targ_pos)

        data = {"images": [], "depths": [], "poses": []}
        # loop over the trajectory
        for jj, x in enumerate(traj):
            # if jj % 2 == 0:
            # set the agent and target state to the current state for rendering
            # sf.agents[ii].x = np.array([x[0], x[1], x[2], x[3], x[4]])
            sf.agents[ii].x = x
            # save agent trajectory
            sf.a_traj["agent-%d" % ii] = np.concatenate(
                (sf.a_traj["agent-%d" % ii], sf.agents[ii].x.reshape(1, 5))
            )
            for kk in range(sf.num_targets):
                sf.targets[kk].x = targ_pos[kk, jj, :]
                # save targets trajectory
                sf.t_traj["targ-%d" % kk] = np.concatenate(
                    (
                        sf.t_traj["targ-%d" % kk],
                        sf.targets[kk].x[:3].reshape(1, 3),
                    )
                )
            # perform observation over current true pose and true target state
            # z = particles in the world frame
            # idx = indexes of particles that observed
            part_z, sf.part_idx, targ_dict, real_data = sf.agents[
                ii
            ].sim.run_view2particles(
                x[:3],
                yaw=x[3],
                pitch=x[4],
                targ_pos=targ_pos[:, jj, :],
                in_situ=True,
                save_img=-1,
            )
            ## union of target particles seen with building footprints pruned
            sf.agents[ii].part_seen = np.hstack(
                (
                    sf.agents[ii].part_seen,
                    sf.agents[ii].world2particleidx(part_z, sf.agents[ii].pf.p_targ_og),
                )
            )

            # execute motion model if targets are not stationary
            if sf.cfg["target_mode"] != "stationary":
                sf.agents[ii].pf.motion_model()
            # update weights based on particles observed
            sf.agents[ii].pf.update_weights(sf.part_idx, targ_dict)

            # prints only the first target seen
            # if targ_z.size != 0:
            #     targ_loc = targ_z.mean(axis=0)
            #     print("target found!!! at (%f, %f)" % (targ_loc[0], targ_loc[1]))

            ## clean up copies of particles seen
            sf.agents[ii].part_seen = np.unique(sf.agents[ii].part_seen)
            ## output p_targ of particles that were not seen
            mask = np.full(len(sf.agents[ii].pf.p_targ_og), True, dtype=bool)
            mask[sf.agents[ii].part_seen.astype(int)] = False
            sf.agents[ii].pf.p_targ = sf.agents[ii].pf.p_targ_og[mask]

            # sf.agents[ii].pf.clear_footprints()
            data["images"].append(real_data["image"])
            data["depths"].append(real_data["depth"])
            data["poses"].append(real_data["pose"])

            # cv2.imwrite(
            #     sf.save_path + "/gt_rgb/" + str(sf.step_count)+str(jj) + ".png",
            #     cv2.cvtColor(real_data["image"], cv2.COLOR_RGB2BGR),
            # )

            # sf.cost_map, sf.visiting_map = sf.nerf.update_cost_map(data)
            # sf.metric["cost_map"] = sf.cost_map
            # sf.metric["visiting_map"] = sf.visiting_map

            sf.reward()
            sf.render()

        return data

    def reward(sf):
        # calculate mmse and %of particles seen
        for ii in range(sf.num_agents):
            mmse = sf.agents[ii].pf.mmse_estimate()
            sf.metric["%part_seen"].append(
                len(sf.agents[ii].part_seen) / len(sf.agents[ii].pf.p_targ_og)
            )
            sf.metric["part_seen"] = sf.agents[ii].part_seen

        # calculate distance between mmse and true target
        for jj in range(sf.num_targets):
            sf.metric["targ-%d" % jj].append(
                np.linalg.norm(sf.targets[jj].x[:2] - mmse["targ-%d" % jj][:2])
            )

        pickle.dump(sf.metric, open(sf.save_path + "/metrics.pkl", "wb"))

    # @profile
    def render(sf):
        for ii in range(sf.num_agents):
            sf.fig.clf()
            ax = sf.fig.add_subplot()

            # plot buildings footprints
            try:
                sf.agents[ii].pf.buildingsXY.plot(ax=ax, color="black", alpha=0.3)
            except:
                # theres an issue when gdf has multipolygons
                sf.agents[ii].pf.buildingsXY.plot(
                    ax=ax, color="black", alpha=0.3, aspect=1
                )

            # plot observations but just in case there are none
            try:
                ax.scatter(
                    sf.agents[ii].pf.p[:, 0][sf.part_idx],
                    sf.agents[ii].pf.p[:, 1][sf.part_idx],
                    marker=".",
                    label="obs",
                    alpha=0.8,
                )
            except:
                print("huh?")
                pass

            # plot pf particles
            # alpha = sf.agents[ii].pf.w["targ-4"] * 1000
            # alpha[alpha > 1] = 1.0
            # alpha[alpha < 0] = 0.0
            # ax.scatter(
            #     sf.agents[ii].pf.p[:, 0],
            #     sf.agents[ii].pf.p[:, 1],
            #     marker=".",
            #     alpha=alpha,
            #     label="pf",
            # )

            # plot particles that have not been seen, target wants to go
            # ax.scatter(
            #     sf.agents[ii].pf.p_targ[:, 0],
            #     sf.agents[ii].pf.p_targ[:, 1],
            #     marker=".",
            #     alpha=0.1,
            #     label="not seen",
            # )

            # localmax = argrelextrema(sf.agents[ii].pf.w, np.greater)[0]
            agent_color = "blue"
            colors = ["red", "green", "magenta", "cyan"]

            for jj in range(sf.num_targets):
                top = 100
                localmax = np.argpartition(sf.agents[ii].pf.w["targ-%d" % jj], -top)[
                    -top:
                ]
                # alpha = sf.agents[ii].pf.w['targ-%d'%jj] * 1000
                # alpha[alpha > 1] = 1.0
                # alpha[alpha < 0] = 0.0
                ax.scatter(
                    sf.agents[ii].pf.p[localmax, 0],
                    sf.agents[ii].pf.p[localmax, 1],
                    marker=".",
                    # label="pf max %d" % jj,
                    # color=colors[jj],
                    alpha=0.3,
                )

                ax.scatter(
                    sf.agents[ii].pf.mmse["targ-%d" % jj][0],
                    sf.agents[ii].pf.mmse["targ-%d" % jj][1],
                    marker="x",
                    # color=colors[jj],
                    # label="mmse %d" % jj,
                )

            # plot agents and targets
            ax.scatter(
                sf.agents[ii].x[0], sf.agents[ii].x[1], label="scout", color=agent_color
            )
            for jj in range(sf.num_targets):
                ax.scatter(
                    sf.targets[jj].x[0],
                    sf.targets[jj].x[1],
                    label="targ-%d" % (jj),
                    # color=colors[jj],
                )

            # plot trajectories
            ax.plot(
                sf.a_traj["agent-%d" % ii][:, 0],
                sf.a_traj["agent-%d" % ii][:, 1],
                color=agent_color,
                alpha=0.4,
            )
            for jj in range(sf.num_targets):
                ax.plot(
                    sf.t_traj["targ-%d" % jj][:, 0],
                    sf.t_traj["targ-%d" % jj][:, 1],
                    # color=colors[jj],
                    alpha=0.4,
                )

            ax.legend()
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            sf.fig.tight_layout()
            sf.fig.canvas.draw()

        sf.fig_metric.clf()
        ax1 = sf.fig_metric.add_subplot()
        for jj in range(sf.num_targets):
            ax1.plot(sf.metric["targ-%d" % jj], label="targ-%d" % jj)#, color=colors[jj])
        ax1.legend()
        sf.fig_metric.tight_layout()
        sf.fig_metric.canvas.draw()

        # plt.pause(0.001)
        sf.a_traj['obs'].append(np.copy(sf.part_idx))
        weights = copy.deepcopy(sf.agents[0].pf.w)
        sf.t_traj["weights"].append(weights)

        sf.fig.savefig(sf.save_path + "/2dplot.pdf", bbox_inches="tight")
        sf.fig_metric.savefig(sf.save_path + "/mse.pdf", bbox_inches="tight")
        pickle.dump(sf.a_traj, open(sf.save_path + "/a_traj.pkl", "wb"))
        pickle.dump(sf.t_traj, open(sf.save_path + "/t_traj.pkl", "wb"))

    def setup(sf, seed):
        random.seed(seed)
        np.random.seed(seed)


@click.command()
@click.option("--seed", default=88, type=int)
@click.option("--map_name", default="philly_cc", type=str)
@click.option("--num_steps", default=40, type=int)
@click.option("--method", default="MI", type=str)
@click.option("--target_mode", default="active", type=str)
def main(seed, map_name, num_steps, method, target_mode):
    cfg, mapDim = init_env(map_name)
    cfg["target_mode"] = target_mode
    cfg["method"] = method
    app = pipeline(seed, cfg, mapDim)
    app.reset()
    for ii in range(num_steps):
        app.step()


if __name__ == "__main__":
    main()
