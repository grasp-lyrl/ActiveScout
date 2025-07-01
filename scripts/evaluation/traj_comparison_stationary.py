import numpy as np
from ipdb import set_trace as st
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os, sys
import pickle
import seaborn as sns

sys.path.append("scripts")
from utils import init_env
from pf import particle_filter

sns.set_theme()
fsz = 20
plt.rc("font", size=fsz)
plt.rc("axes", titlesize=fsz)
plt.rc("axes", labelsize=fsz)
plt.rc("xtick", labelsize=fsz)
plt.rc("ytick", labelsize=fsz)
plt.rc("legend", fontsize=0.7 * fsz)
plt.rc("figure", titlesize=fsz)
plt.rc("pdf", fonttype=42)
sns.set_style("ticks", rc={"axes.grid": True})

"""
This script is used to plot the results shown in the paper.

Figure 6 Bottom
"""


s_colors = np.array(sns.color_palette())
ind = np.array([3, 2, 4, 1], dtype=int)
colors = s_colors[ind]

num_targets = 20


mode = "stationary"
map_name = "philly_cc"
path_base = "data_gtmap/" + map_name
# path_base = "data_nerf/"+map_name
method = "_map"
exp_name = "/s88_" + mode + method
# exp_name = "/forfilterplot"
path = path_base + exp_name
metric = pickle.load(open(path + "/metrics.pkl", "rb"))
a_traj = pickle.load(open(path + "/a_traj.pkl", "rb"))
t_traj = pickle.load(open(path + "/t_traj.pkl", "rb"))  

cfg, mapDim = init_env(map_name)
pf = particle_filter(cfg, mapDim, num_targets)

fig, ax = plt.subplots()

# plot buildings footprints
pf.buildingsXY.plot(ax=ax, color="black", alpha=0.3)

targ_num = 3
step = 500

# plot agents and targets
ax.scatter(
    a_traj["agent-0"][step, 0],
    a_traj["agent-0"][step, 1],
    label="scout",
    color=s_colors[0],
    edgecolors="black",
    linewidths=0.7,
)
for jj in range(num_targets):
    ax.scatter(
        t_traj["targ-%d" % jj][step, 0],
        t_traj["targ-%d" % jj][step, 1],
        # label="targ-%d" % (jj),
        # color=colors[jj],
        # edgecolors="black",
        # linewidths=0.7,
    )

#plot the start of the traj as an x
ax.scatter(
    a_traj["agent-0"][0, 0],
    a_traj["agent-0"][0, 1],
    label="start loc",
    marker='x',
    color='black',
    edgecolors="black",
    linewidths=0.7,
)

# plot trajectories
ax.plot(
    a_traj["agent-%d" % 0][:step, 0],
    a_traj["agent-%d" % 0][:step, 1],
    color=s_colors[0],
    # alpha=0.4,
    label='trajectory'
)
for jj in range(num_targets):
    ax.plot(
        t_traj["targ-%d" % jj][:step, 0],
        t_traj["targ-%d" % jj][:step, 1],
        # color=colors[jj],
        # alpha=0.4,
    )
# ax.set_xlabel("x (m)")
# ax.set_ylabel("y (m)")
ax.set_yticklabels([])
ax.set_xticklabels([])
# ax.set_xlim(-600,0)
# ax.set_ylim(0,600)
ax.legend()
fig.tight_layout()
fig.savefig(path_base + "/" + map_name+mode+method + "_traj.pdf", bbox_inches="tight")


mode = "stationary"
map_name = "philly_cc"
# path_base = "data_gtmap/" + map_name
path_base = "data_nerf/"+map_name
method = "_nerf"
exp_name = "/s88_" + mode + method
# exp_name = "/forfilterplot"
path = path_base + exp_name
metric = pickle.load(open(path + "/metrics.pkl", "rb"))
a_traj = pickle.load(open(path + "/a_traj.pkl", "rb"))
try:
    t_traj = pickle.load(open(path + "/t_traj.pkl", "rb"))  
except: 
    pass


cfg, mapDim = init_env(map_name)
pf = particle_filter(cfg, mapDim, num_targets)

fig, ax = plt.subplots()

# plot buildings footprints
pf.buildingsXY.plot(ax=ax, color="black", alpha=0.3)

# plot agents and targets
ax.scatter(
    a_traj["agent-0"][step, 0],
    a_traj["agent-0"][step, 1],
    label="scout",
    color=s_colors[0],
    edgecolors="black",
    linewidths=0.7,
)
for jj in range(num_targets):
    ax.scatter(
        t_traj["targ-%d" % jj][step, 0],
        t_traj["targ-%d" % jj][step, 1],
        # label="targ-%d" % (jj),
        # color=colors[jj],
        # edgecolors="black",
        # linewidths=0.7,
    )

#plot the start of the traj as an x
ax.scatter(
    a_traj["agent-0"][0, 0],
    a_traj["agent-0"][0, 1],
    label="start loc",
    marker='x',
    color='black',
    edgecolors="black",
    linewidths=0.7,
)

# plot trajectories
ax.plot(
    a_traj["agent-%d" % 0][:step, 0],
    a_traj["agent-%d" % 0][:step, 1],
    color=s_colors[0],
    # alpha=0.4,
    label='trajectory'
)
for jj in range(num_targets):
    ax.plot(
        t_traj["targ-%d" % jj][:step, 0],
        t_traj["targ-%d" % jj][:step, 1],
        # color=colors[jj],
        # alpha=0.4,
    )
# ax.set_xlabel("x (m)")
# ax.set_ylabel("y (m)")
# ax.set_xlim(-600,0)
# ax.set_ylim(0,600)
ax.set_yticklabels([])
ax.set_xticklabels([])
# ax.legend()
fig.tight_layout()

fig.savefig(path_base + "/" + map_name+mode+method + "_traj.pdf", bbox_inches="tight")


plt.show()