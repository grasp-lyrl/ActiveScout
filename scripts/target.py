import numpy as np
import networkx as nx
from ipdb import set_trace as st
import matplotlib.pyplot as plt
import sys, pickle, random
from utils import init_env, create_graph
from pf import particle_filter


class target:
    def __init__(sf, cfg, mapDim, G=None, dt=1.0):
        sf.cfg = cfg
        sf.mapDim = mapDim
        sf.dt = dt
        sf.x0 = np.array(
            [
                np.random.uniform(sf.mapDim["x_min"], sf.mapDim["x_max"]),
                np.random.uniform(sf.mapDim["y_min"], sf.mapDim["y_max"]),
                sf.cfg["target_altitude"],
                0.0,
            ]
        )
        sf.xf = np.array(
            [
                np.random.uniform(sf.mapDim["x_min"], sf.mapDim["x_max"]),
                np.random.uniform(sf.mapDim["y_min"], sf.mapDim["y_max"]),
                sf.cfg["target_altitude"],
                0.0,
            ]
        )
        sf.G = G
        sf.nodes_pos = np.array(
            [
                np.hstack((node, pos))
                for node, pos in nx.get_node_attributes(sf.G, "pos").items()
            ]
        )

        sf.fig = None
        sf.mode = sf.cfg["target_mode"]
        sf.reset()

    def reset(sf):
        """
        given some world location x0, find the closest node in the graph
        and set x to that node position in world coordinates
        """
        x_node = sf.closest_node(sf.x0[:2])
        sf.x = sf.node2state(x_node)

    # @profile
    def step(sf, not_seen=None):
        ## do multiple node steps
        if sf.mode == "stationary":
            x_node = sf.closest_node(sf.x[:2])
            path = [x_node]

        if sf.mode == "brownian":
            path = []
            for ii in range(8):
                x_node = sf.closest_node(sf.x[:2])
                neighbors = list(sf.G.neighbors(x_node))
                if len(neighbors) > 0:
                    x_node = random.choice(neighbors)
                    path.append(x_node)

        if sf.mode == "goal":
            x_node = sf.closest_node(sf.x[:2])
            xf_node = sf.closest_node(sf.xf[:2])
            if x_node == xf_node:
                sf.xf = np.array(
                    [
                        np.random.uniform(sf.mapDim["x_min"], sf.mapDim["x_max"]),
                        np.random.uniform(sf.mapDim["y_min"], sf.mapDim["y_max"]),
                        sf.cfg["target_altitude"],
                        0.0,
                    ]
                )
                xf_node = sf.closest_node(sf.xf[:2])
            try:
                path = nx.astar_path(sf.G, x_node, xf_node)
            except:
                path = [x_node]

        if sf.mode == "active":
            # sample from particles not seen by agents
            # goal_idx = np.random.choice(not_seen.shape[0])

            # select n_min closest not seen particles to target
            try:   
                goals_idx = np.linalg.norm(not_seen[:,:3]-sf.x[:3],axis=1)
                n_min = min(20, len(goals_idx)-1)
                idx = np.random.choice(np.arange(0,n_min))
                goal_idx = np.argpartition(goals_idx, n_min)[idx]
                goal = not_seen[goal_idx]
            except:
                goal = sf.x[:2]

            x_node = sf.closest_node(sf.x[:2])
            xf_node = sf.closest_node(goal[:2])
            try:
                path = nx.astar_path(sf.G, x_node, xf_node)
            except:
                # no path found when goal is inside free space of closed building
                path = [x_node]

        # convert from nodes to world coordinates
        traj = []
        for ii in range(len(path)):
            x_node = path[ii]
            traj.append(sf.node2state(x_node))
            # each node is ~9m apart
            if sf.mode == 'goal':
                end = 10
            elif sf.mode == 'active':
                end = 20
            else:
                end = 0
            if ii == end:
                break
        traj = np.array(traj)
        sf.x = traj[-1, :]
        return traj

    def closest_node(sf, x):
        #sf.nodes_pos = [idx,x,y]        
        distances = np.linalg.norm(sf.nodes_pos[:, 1:] - x, axis=1)
        node = sf.nodes_pos[np.argmin(distances), 0].astype(int)
        return node

    def node2state(sf, node):
        x = np.array(nx.get_node_attributes(sf.G, "pos")[node])
        return np.array([x[0], x[1], sf.x0[2], sf.x0[3]])

    def render(sf):
        if sf.fig == None:
            sf.fig = plt.figure()

        sf.fig.clf()
        ax = sf.fig.add_subplot()

        pos = nx.get_node_attributes(sf.G, "pos")
        nx.draw(
            sf.G, pos, node_size=1, node_color="skyblue", edge_color="gray", width=1.0
        )
        ax.scatter(sf.x[0], sf.x[1])

        plt.tight_layout()
        plt.draw()
        plt.pause(0.0001)


def main():
    map_name = 'philly_cc'
    cfg, mapDim = init_env(map_name)

    try:
        G = pickle.load(open(cfg["graph_file"], "rb"))
    except:
        G = create_graph(pf=particle_filter(cfg, mapDim), render=True, save=False)

    targ = target(cfg, mapDim, G=G)

    for i in range(100):
        targ.step()
        targ.render()


if __name__ == "__main__":
    main()
