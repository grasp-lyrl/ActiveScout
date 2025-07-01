import numpy as np
from ipdb import set_trace as st
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns

"""
This script is used to plot the results shown in the paper.

Figure 9
"""

sns.set_theme()
fsz = 32
plt.rc("font", size=fsz)
plt.rc("axes", titlesize=fsz)
plt.rc("axes", labelsize=fsz)
plt.rc("xtick", labelsize=fsz)
plt.rc("ytick", labelsize=fsz)
plt.rc("legend", fontsize=0.5*fsz)
plt.rc("figure", titlesize=fsz)
plt.rc("pdf", fonttype=42)
sns.set_style("ticks", rc={"axes.grid": True})


s_colors = np.array(sns.color_palette())

mode = "active"
map_name = 'philly_cc'
path = "data_nerf/"+map_name+"/s88_"+mode+"_nerf"
err = np.load(path+"/errors.npy")
psnr = err[:-1,1].reshape(43,2)
lpips = err[:-1,2].reshape(43,2)
path = "data_nerf/"+map_name+"/s88_"+mode+"_nerf_2000"
err1 = np.load(path+"/errors.npy")

map_name = 'nyc_stuy'
path = "data_nerf/"+map_name+"/s88_"+mode+"_nerf"
err2 = np.load(path+"/errors.npy")
psnr2 = err2[:-1,1].reshape(43,2)
path = "data_nerf/"+map_name+"/s88_"+mode+"_nerf_2000"
err3 = np.load(path+"/errors.npy")


fig, ax1 = plt.subplots()
end = 42
xaxis = np.arange(0,end)

ax1.bar(
    xaxis,
    psnr[:,1][:end],
    width=(xaxis[1] - xaxis[0]),
    align="edge",
    linewidth=0,
    color = s_colors[0],
    label= "4k training steps"
)
ax1.bar(
    xaxis,
    # err1[:end,1],
    psnr[:,0][:end],
    width=(xaxis[1] - xaxis[0]),
    align="edge",
    linewidth=0,
    color = s_colors[1],
    label = "2k training steps",
    alpha = 0.6
)

ax1.set_xlabel('Planning Step')
ax1.set_ylabel('PSNR')
ax1.legend(loc=3)
fig.tight_layout()
sns.despine(ax=ax1)
fig.savefig(os.path.join('data_nerf/philly_cc/'+'philly_psnr.pdf'),bbox_inches='tight')



fig, ax1 = plt.subplots()
ax1.bar(
    xaxis,
    psnr2[:,1][:end],
    width=(xaxis[1] - xaxis[0]),
    align="edge",
    linewidth=0,
    color = s_colors[0],
    label= "4000 training steps"
)
ax1.bar(
    xaxis,
    # err3[:end,1],
    psnr2[:,0][:end],
    width=(xaxis[1] - xaxis[0]),
    align="edge",
    linewidth=0,
    color = s_colors[1],
    label = "2000 training steps",
    alpha = 0.6
)

ax1.set_xlabel('Planning Step')
ax1.set_ylabel('PSNR')
# ax1.legend(loc=3)
fig.tight_layout()
sns.despine(ax=ax1)
fig.savefig(os.path.join('data_nerf/nyc_stuy/'+'nyc_psnr.pdf'),bbox_inches='tight')

plt.show()