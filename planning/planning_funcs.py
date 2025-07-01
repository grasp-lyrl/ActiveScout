"""
Imports
"""

# Vehicles. Currently there is only one.
# There must also be a corresponding parameter file.
from rotorpy.vehicles.crazyflie_params import quad_params

# You will also need a controller (currently there is only one) that works for your vehicle.
from rotorpy.controllers.quadrotor_control import SE3Control

# And a trajectory generator
from rotorpy.trajectories.minsnap import MinSnap

from rotorpy.simulate import (
    time_exit,
    merge_dicts,
    sanitize_trajectory_dic,
    sanitize_control_dic,
)

# Reference the files above for more documentation.

# Other useful imports
import sys, os, copy, tqdm, datetime
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.spatial.transform import (
    Rotation,
)  # For doing conversions between different rotation descriptions, applying rotations, etc.
from ipdb import set_trace as st


def get_minsnap_traj(waypoints, yaw=None):
    controller = SE3Control(quad_params)

    # yaw = np.linspace(2 * np.pi, 0, len(waypoints))
    if yaw is None:
        yaw = np.linspace(0, 0, len(waypoints))

    ## yaw should be in radians?
    trajectory = MinSnap(points=waypoints, yaw_angles=yaw, v_avg=1.0, verbose=False)

    try:
        t_final = np.sum(trajectory.delta_t)
    except:
        t_final = 1.0

    # N_sample_disc = max(int(t_final), 20)
    # number of steps + 1
    N_sample_disc = 20
    t_step = t_final / N_sample_disc
    time = [0]
    flat = [sanitize_trajectory_dic(trajectory.update(time[-1]))]
    control_ref = [sanitize_control_dic(controller.update_ref(time[-1], flat[-1]))]
    exit_status = None
    while True:
        exit_status = exit_status or time_exit(time[-1], t_final)
        if exit_status:
            break

        time.append(time[-1] + t_step)
        flat.append(sanitize_trajectory_dic(trajectory.update(time[-1])))
        control_ref.append(
            sanitize_control_dic(controller.update_ref(time[-1], flat[-1]))
        )

    time = np.array(time, dtype=float)
    flat = merge_dicts(flat)
    control_ref = merge_dicts(control_ref)
    # traj_x_quat = np.hstack((flat['x'], control_ref["cmd_q"]))

    # convert to euler
    euler = np.zeros((control_ref["cmd_q"].shape[0], 3))
    for i in range(control_ref["cmd_q"].shape[0]):
        rot = Rotation.from_quat(control_ref["cmd_q"][i])
        euler[i] = rot.as_euler("zyx", degrees=True)

    # idk why but output seems to be yaw, pitch, roll
    traj_x_quat = np.hstack((flat["x"], euler))

    if traj_x_quat.shape[0] > N_sample_disc:
        traj_x_quat = traj_x_quat[: N_sample_disc + 1]
    return traj_x_quat


def main():
    import networkx as nx

    def create_graph():
        Nx = 20
        Ny = 20
        G = nx.grid_2d_graph(Nx, Ny)
        ## world coords (x,y,z)
        x = np.linspace(-10, 10, Nx)
        y = np.linspace(-10, 10, Ny)
        Y, X = np.meshgrid(y, x)
        z = np.ones(Nx * Ny) * 350.0
        positions = np.vstack((X.flatten(), Y.flatten(), z))
        pos = {(x, y): (positions[:, ii]) for ii, (x, y) in enumerate(G.nodes())}
        nx.set_node_attributes(G, pos, "pos")
        return G

    def node2state(G, node):
        return np.array(nx.get_node_attributes(G, "pos")[node])

    G = create_graph()
    x_node = (5, 5)
    xf_node = (10, 3)
    astar_path = nx.astar_path(G, x_node, xf_node)

    waypoints = []
    for node in astar_path:
        waypoints.append(node2state(G, node))
    waypoints = np.array(waypoints)

    traj_x_quat = get_minsnap_traj(waypoints)
    st()


if __name__ == "__main__":
    main()
