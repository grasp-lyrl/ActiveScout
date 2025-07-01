import numpy as np
from copy import deepcopy
from ipdb import set_trace as st
import matplotlib.pyplot as plt
import random, sys, pickle
import networkx as nx
from pf import particle_filter
from utils import init_env

sys.path.append("planning/rotorpy")
sys.path.append("planning")
from planning_funcs import get_minsnap_traj

sys.path.append("simulator")
from mgl_imgui_simOSM import mgl_imgui_simOSM as sim


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)


class scout:
    def __init__(sf, cfg, pf, sim):
        sf.cfg = cfg
        sf.dim = 4
        sf.uDim = 1
        sf.Q = 1e-1 * np.eye(3)
        sf.dt = 1.0
        sf.pf = pf
        sf.sim = sim
        sf.sim.init_objects(particles=sf.pf.p)
        # sf.planner = planning_opt(sf.dim, sf.uDim)
        # (x,y,z,ϕ,ψ,θ)
        sf.x0 = np.array(sf.cfg["agent"]["x0"])
        sf.create_graph()
        sf.reset()

    def reset(sf, x=None):
        # [x, y, z, ϕ, ψ, θ]
        sf.x = x if x is not None else sf.x0
        sf.x_node = sf.closest_node(sf.x[:3])
        # list of particles seen
        sf.part_seen = []

    def survey_traj(sf, traj_len=11, survey_z=None, survey_pitch=None):
        if not survey_z:
            survey_z = sf.cfg["agent"]["survey_z"]
        if not survey_pitch:
            survey_pitch = sf.cfg["agent"]["survey_pitch"]
        # fly up in same position
        xf = np.array([sf.x[0], sf.x[1], survey_z, sf.x[3], survey_pitch])

        # interpolate z
        z = np.linspace(sf.x[2], xf[2], traj_len)

        ## minsnap waypoints from a star
        waypoints = []
        for ii in range(traj_len):
            waypoints.append([sf.x[0], sf.x[1], z[ii]])
        waypoints = np.array(waypoints)

        # yaw is body and cam yaw
        yaw = np.linspace(0, 360, traj_len) + sf.x[3]
        yaw = sf.yaw_clip(yaw, deg=True)

        # (x,y,z,i,j,k,w) or currently -> (x,y,z,yaw,b_pitch,b_roll)
        trajectory = get_minsnap_traj(waypoints, np.deg2rad(yaw))

        # interpolate camera pitch
        θ = np.linspace(sf.x[4], xf[4], len(trajectory))

        # [x,y,z,yaw,c_pitch]
        traj = np.vstack((trajectory[:, :4].T, θ)).T

        # do another 360
        for y_ang in yaw:
            if y_ang == yaw[-1]:
                pitch = xf[4]
            else:
                pitch = xf[4] + np.random.uniform(-10,10)
            traj = np.vstack((traj, [xf[0], xf[1], xf[2], y_ang, pitch]))

        # just for protection, but it will be updated outside in pipeline
        sf.x = traj[-1]

        return traj

    def sample_poses(sf, N):
        """
        ϕ: agent yaw, not used currently. maybe this is yaw in minsap
        ψ: camera yaw wrt to mgl camera frame
        θ: camera pitch wrt to mgl camera frame

        """
        if sf.cfg["agent"]["local_sample"]:
            x = sf.x[0] + np.random.uniform(
                sf.cfg["agent"]["pos_sample"][0], sf.cfg["agent"]["pos_sample"][1], N
            )
            y = sf.x[1] + np.random.uniform(
                sf.cfg["agent"]["pos_sample"][0], sf.cfg["agent"]["pos_sample"][1], N
            )
            x = np.clip(x, sf.pf.mapDim["x_min"], sf.pf.mapDim["x_max"])
            y = np.clip(y, sf.pf.mapDim["y_min"], sf.pf.mapDim["y_max"])
        else:
            x = np.random.uniform(sf.pf.mapDim["x_min"], sf.pf.mapDim["x_max"], N)
            y = np.random.uniform(sf.pf.mapDim["y_min"], sf.pf.mapDim["y_max"], N)

        z = np.random.uniform(
            sf.cfg["agent"]["z_sample"][0], sf.cfg["agent"]["z_sample"][1], N
        )
        # ϕ is heading angle, yaw in body frame
        # ϕ = np.random.uniform(-1, 1, N) * np.pi / 4 + sf.x[3]

        # ψ is camera yaw wrt to mgl camera frame
        ψ = np.random.uniform(
            sf.cfg["agent"]["yaw_sample"][0], sf.cfg["agent"]["yaw_sample"][1], N
        )
        # θ is camera pitch wrt to mgl camera frame
        θ = np.random.uniform(
            sf.cfg["agent"]["pitch_sample"][0], sf.cfg["agent"]["pitch_sample"][1], N
        )
        # singularity at -90
        θ[θ == -90] = -89.99

        return np.vstack((x, y, z, ψ, θ)).T

    # @profile
    def hallucinate(sf, x):
        """
        x is set of states (Nx, stDim)
        Given poses hallucinate to calculate likelihood p(z|θ)
        calculate predictive information, i.e. mutual information
        return observations, weights, mutual information of best pose

        we will sample from Nx poses each with No observations

        θ = location of the target
        p(θ) = our posterior over the locations
        p(z|θ,x) = prob. of receiving an observation of the target
                   at location θ from any viewpoint x

        I(z;θ) = H(z) - H(z|θ)
        where H(z) = entropy of any observation
        where H(z|θ) = entropy of obs given target state distribution
        """
        N = sf.pf.N
        Nx = x.shape[0]
        No = 10  # number of observations
        # mutual information
        I = np.empty((Nx, sf.pf.num_targets + 1))
        # max likelihood estimate
        mle = []
        maP = np.empty((Nx, sf.pf.num_targets+1))
        for ii in range(Nx):
            particles_idx = []
            yaw = x[ii, 3]
            # for each pos, num cam pose, collect idx of particles seen
            for _ in range(4):
                _, part_idx, _, _ = sf.sim.run_view2particles(
                    x[ii, :3],
                    yaw=yaw,
                    pitch=x[ii, 4],
                    targ_pos=[],
                    in_situ=False,
                    save_img=-1,
                )

                yaw += 90
                if yaw > 180:
                    yaw = yaw - 360
                if yaw < -180:
                    yaw = yaw + 360

                particles_idx = np.hstack((particles_idx, part_idx)).astype(int)
            particles_idx = np.unique(particles_idx)
            """
            check if we see any particles first
            no obs particles = p(false obs)
            observed particles = 1-p(false obs)
            """
            ## calculate p(z|θ)
            if len(particles_idx) != 0:
                # prob of false obs
                z = np.random.binomial(n=No, p=0.05, size=N) / No
                # for observed particles flip the prob
                z[particles_idx] = 1 - z[particles_idx]
                # normalize
                p_zgθ = z / z.sum()
            else:
                p_zgθ = np.zeros(N)

            for jj, targ_name in enumerate(sf.pf.w):
                p_θ = sf.pf.w[targ_name]
                p_z = p_θ * p_zgθ
                H_z = sf.pf.entropy(p_θ)

                H_zgθ = sf.pf.entropy(p_zgθ)
                weightedH_zgθ = p_θ * H_zgθ
                avgH_zgθ = np.sum(weightedH_zgθ) / N

                I[ii, jj] = H_z - avgH_zgθ

                maP[ii,jj] = (np.sum(np.log(p_z + 1e-6)) / N)
            # max likelihood estimate
            mle.append(np.sum(np.log(p_zgθ + 1e-6)) / N)

        ## based on samples across targets create multinomial distribution
        samp_idx = np.zeros(I.shape[1]).astype('int')
        sampI = np.zeros(I.shape[1])
        for ii in range(I.shape[1]):
            I_prob = np.exp(I[:,ii])/ np.sum(np.exp(I[:,ii]))
            samp_idx[ii] = np.nonzero(np.random.multinomial(1, I_prob))[0][0]
            sampI[ii] = I[samp_idx[ii],ii]

        ## based on multinomial of targets, select the pose
        poseI = np.exp(sampI)/ np.sum(np.exp(sampI))
        targ_idx = np.nonzero(np.random.multinomial(1, poseI))[0][0]
        chosen_idx = samp_idx[targ_idx]
        # print(chosen_idx)
        
        if sf.cfg["method"] == "map":
            I = np.sum(maP, axis=1)
            chosen_idx = np.argmax(I)  
        if sf.cfg["method"] == "mle":
            I = mle

        return I, chosen_idx

    # @profile
    def step(sf):
        """
        sample poses
        robot needs to get image from scene
        from scene calculate probability of observation of target
        update weights of the pf
        move to location that max MI
        """
        xp = sf.sample_poses(N=sf.cfg["agent"]["hallucinate_n_samples"])

        # nerf_pose_idx = np.argmax(unc)
        # pose_idx = sf.hallucinate(xp)
        I, chosen_idx = sf.hallucinate(xp)

        # stuff for graph and astar path
        xf = xp[chosen_idx, :]
        x_node = sf.closest_node(sf.x[:3])
        xf_node = sf.closest_node(xf[:3])
        try:
            path = nx.astar_path(sf.G, x_node, xf_node)
        except:
            path = [x_node]

        # interpolate z
        z = np.linspace(sf.x[2], xf[2], len(path))
        # ψ = np.linspace(sf.x[4],xf[4],len(path))

        ## minsnap waypoints from a star
        waypoints = []
        for ii, node in enumerate(path):
            # graph is 2d, so we interpolate z and use that
            waypoints.append(sf.node2state(node) + (z[ii],))
        waypoints = np.array(waypoints)

        # yaw is still broken from minsnap traj, also just not used
        yaw = np.linspace(sf.x[3], xf[3] + 360, len(path))
        yaw = sf.yaw_clip(yaw, deg=True)

        # (x,y,z,i,j,k,w) or currently -> (x,y,z,yaw,pitch,roll)
        trajectory = get_minsnap_traj(waypoints, np.deg2rad(yaw))

        # interpolate camera yaw and pitch
        # ψ = np.linspace(sf.x[4], xf[4], len(traj))
        θ = np.linspace(sf.x[4], xf[4], len(trajectory))

        # [x,y,z,yaw,pitch]
        traj = np.vstack((trajectory[:, :4].T, θ)).T

        # do another 360 for 10 steps
        yaw = np.linspace(sf.x[3], xf[3] + 360, 10)
        yaw = sf.yaw_clip(yaw, deg=True)
        for ii, y_ang in enumerate(yaw):
            if ii == len(yaw)-1:
                pitch = xf[4]
            else:
                pitch = np.random.uniform(sf.cfg["agent"]["pitch_sample"][0], sf.cfg["agent"]["pitch_sample"][1])
                if pitch > 85.0:
                    pitch = 85.0
                if pitch < -85.0:
                    pitch = -85.0
            traj = np.vstack((traj, [xf[0], xf[1], xf[2], y_ang, pitch]))

        # just for protection, but it will be updated outside in pipeline
        sf.x = traj[-1]

        return traj, I, chosen_idx

    def world2particleidx(sf, p_w, grid):
        """
        p_w: particles in world frame [x,y,z,1]
        grid: 2d array of world particles [x,y,z,1]
        return the grid index of the matched particle
        """
        if p_w.size == 0:
            return np.array([])
        # only compare x and y and get indices of updated particles
        gridxy = np.stack((grid[:, 0], grid[:, 1]))
        pwxy = np.stack((p_w[:, 0], p_w[:, 1]))

        # use broadcast compute distance between grid (1000,p,2) and pwxy (p,2)
        distances = np.linalg.norm(gridxy[:, np.newaxis, :].T - pwxy.T, axis=2)
        # distances (1000,p)
        part_idx = np.argmin(distances, axis=0)
        return part_idx

    def create_graph(sf):
        try:
            sf.G = pickle.load(open(sf.cfg["graph_file"], "rb"))
        except:
            Nx = sf.pf.Nx
            Ny = sf.pf.Ny
            sf.G = nx.grid_2d_graph(Nx, Ny)
            # Set all weights to 1
            for edge in sf.G.edges:
                sf.G.edges[edge]['weight'] = 1
            sf.G.add_edges_from([
                ((x, y), (x+1, y+1))
                for x in range(Nx-1)
                for y in range(Ny-1)
            ] + [
                ((x+1, y), (x, y+1))
                for x in range(Nx-1)
                for y in range(Ny-1)
            ], weight=1.4)

            x = np.linspace(sf.pf.mapDim["x_min"], sf.pf.mapDim["x_max"], Nx)
            y = np.linspace(sf.pf.mapDim["y_min"], sf.pf.mapDim["y_max"], Ny)
            Y, X = np.meshgrid(y, x)
            z = np.ones(Nx * Ny) * sf.x0[2]
            positions = np.vstack((X.flatten(), Y.flatten(), z)).T
            pos = {(x, y): (positions[ii, :]) for ii, (x, y) in enumerate(sf.G.nodes())}
            nx.set_node_attributes(sf.G, pos, "pos")


        sf.nodes_pos = np.array(
            [
                np.hstack((node, pos))
                for node, pos in nx.get_node_attributes(sf.G, "pos").items()
            ]
        )
        # pos = nx.get_node_attributes(sf.G, "pos")
        # nx.draw(sf.G, pos, node_size=1, node_color="skyblue", edge_color="gray", width=1.0)
        # plt.show()

    def node2state(sf, node):
        x = np.array(nx.get_node_attributes(sf.G, "pos")[node])
        return x[0], x[1]  # , x[2]

    # @profile
    def closest_node(sf, x):
        # sf.nodes_pos = [idx,idx,x,y,z]
        #     distances = np.linalg.norm(sf.nodes_pos[:, 2:] - x, axis=1)
        #     node = tuple(sf.nodes_pos[np.argmin(distances), :2].astype(int))
        
        distances = np.linalg.norm(sf.nodes_pos[:, 1:] - x[:2], axis=1)
        node = sf.nodes_pos[np.argmin(distances), 0].astype(int)
        return node

    def yaw_clip(sf, yaw, deg=True):
        # doing it twice should make sure its within -pi to pi
        if deg:
            yaw[yaw > 180] = yaw[yaw > 180] - 360
            yaw[yaw > 180] = yaw[yaw > 180] - 360
            yaw[yaw < -180] = yaw[yaw < -180] + 360
            yaw[yaw < -180] = yaw[yaw < -180] + 360
        else:
            yaw[yaw > np.pi] = yaw[yaw > np.pi] - 2 * np.pi
            yaw[yaw > np.pi] = yaw[yaw > np.pi] - 2 * np.pi
            yaw[yaw < -np.pi] = yaw[yaw < -np.pi] + 2 * np.pi
            yaw[yaw < -np.pi] = yaw[yaw < -np.pi] + 2 * np.pi

        return yaw

    # def sample_target_from_particles(sf, N=10):
    #     """
    #     sample from pf weights target locations
    #     return particles locations that were sampled, have not sample a 'z'
    #     out: [3,N]
    #     """
    #     sample_counts = np.random.multinomial(N, sf.pf.w)
    #     sample_idx = np.repeat(np.arange(len(sf.pf.w)), sample_counts)
    #     targ_pose = sf.pf.p[sample_idx]
    #     targ_weight = sf.pf.w[sample_idx]
    #     return targ_pose, targ_weight

    # def get_control(sf, xf):
    #     """
    #     Based on information gain, sample trajectories and get a control
    #     planner class is Torch code
    #     """
    #     x0 = th.from_numpy(sf.x.astype("f4"))
    #     xf = th.from_numpy(xf.astype("f4"))
    #     u, τ = sf.planner.plan(x0, xf)
    #     return u.numpy(), τ.numpy()


def main():
    map_name = "philly_cc"
    cfg, mapDim = init_env(map_name)

    setup(88)
    agent = scout(cfg, pf=particle_filter(cfg, mapDim), sim=sim(cfg, headless=False))
    agent.reset()
    from target import target

    target = target(cfg, mapDim)

    fig = plt.figure()
    T = 100
    for ii in range(T):
        agent.step()
        target.step()

        fig.clf()
        ax = fig.add_subplot()
        # plot pf
        alpha = agent.pf.w * 1000
        alpha[alpha > 1] = 1.0
        alpha[alpha < 0] = 0.0

        ax.scatter(
            agent.pf.p[:, 0], agent.pf.p[:, 1], marker=".", alpha=alpha, label="pf"
        )
        try:
            ax.scatter(
                agent.pf.p[:, 0][agent.obs],
                agent.pf.p[:, 1][agent.obs],
                marker=".",
                label="obs",
            )
        except:
            pass
        ax.scatter(agent.x[0], agent.x[1], label="scout")
        ax.scatter(target.x[0], target.x[1], label="target")

        plt.legend()
        plt.draw()
        plt.pause(0.001)


if __name__ == "__main__":
    main()
