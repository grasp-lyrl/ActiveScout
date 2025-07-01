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

Figure 6 Top
Figure 7 
Figure 10
"""

sns.set_theme()
fsz = 32
plt.rc("font", size=fsz)
plt.rc("axes", titlesize=fsz)
plt.rc("axes", labelsize=fsz)
plt.rc("xtick", labelsize=fsz)
plt.rc("ytick", labelsize=fsz)
plt.rc("legend", fontsize=0.7*fsz)
plt.rc("figure", titlesize=fsz)
plt.rc("pdf", fonttype=42)
sns.set_style("ticks", rc={"axes.grid": True})


s_colors = np.array(sns.color_palette())
ind = np.array([3, 2, 4, 1],dtype=int)
colors = s_colors[ind]

num_targets = 4


def plot_things(metric,path_base,mode,label,legend=False):
    max_targ = []
    max_idx = []
    max_color = []
    min_targ = []
    min_idx = []
    min_color = []
    avg_targ = []

    xaxis = np.arange(np.array(metric["targ-0"]).shape[0])
    
    for ii in xaxis:
        targs = []
        for jj in range(num_targets):
            targs.append(metric['targ-%d'%jj][ii])
        targs =np.array(targs)
        max_targ.append(np.max(targs))
        avg_targ.append(np.mean(targs))
        max_idx.append(np.argmax(targs))
        max_color.append(colors[np.argmax(targs)])

        min_targ.append(np.min(targs))
        min_idx.append(np.argmin(targs))
        min_color.append(colors[np.argmin(targs)])

    fig, ax1 = plt.subplots(figsize=(8,7))

    ax1.bar(
        xaxis,
        max_targ,
        width=(xaxis[1] - xaxis[0]),
        align="edge",
        linewidth=0,
        color=max_color,
        alpha = 0.2
    )
    ax1.bar(
        xaxis,
        min_targ,
        width=(xaxis[1] - xaxis[0]),
        align="edge",
        linewidth=0,
        edgecolor='black',
        color=min_color,
    )

    color_ = {'target-0':colors[0], 'target-1':colors[1], 'target-2':colors[2], 'target-3':colors[3]}         
    labels = list(color_.keys())
    handles = [plt.Rectangle((0,0),1,1, color=color_[label]) for label in labels]
    from matplotlib.lines import Line2D
    handles.append(Line2D([0],[0],color='black', linestyle='dashed'))
    if legend:
        plt.legend(handles, labels)
    

    if mode == "goal":
        ax1.set_ylim(0,800)
    else:
        ax1.set_ylim(0,700)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('RMSE')
    fig.tight_layout()
    sns.despine(ax=ax1)
    fig.savefig(os.path.join(path_base,map_name+"_"+mode+'_'+label+'.pdf'),bbox_inches='tight')


mode = "active"
map_name = 'philly_cc'
path_base = "/data/activescout/data_gtmap/"+map_name
path = "/data/activescout/data_nerf/"+map_name+"/s88_"+mode+"_nerf"
metric = pickle.load(open(path + "/metrics.pkl", "rb"))
plot_things(metric, path_base, mode, "nerfMI")


exp_name = "/s88_"+mode+"_MI"
path = path_base + exp_name
metric1 = pickle.load(open(path + "/metrics.pkl", "rb"))

plot_things(metric1, path_base, mode, "MI")

exp_name = "/s88_"+mode+"_map"
path = path_base + exp_name
metric2 = pickle.load(open(path + "/metrics.pkl", "rb"))
plot_things(metric2, path_base, mode, "map", legend=True)


mode = "active"
map_name = 'nyc_stuy'
path_base = "/data/activescout/data_gtmap/"+map_name
path = "/data/activescout/data_nerf/"+map_name+"/s88_"+mode+"_nerf"
metric = pickle.load(open(path + "/metrics.pkl", "rb"))
plot_things(metric, path_base, mode, "nerfMI")


exp_name = "/s88_"+mode+"_MI"
path = path_base + exp_name
metric1 = pickle.load(open(path + "/metrics.pkl", "rb"))

plot_things(metric1, path_base, mode, "MI")

exp_name = "/s88_"+mode+"_map"
path = path_base + exp_name
metric2 = pickle.load(open(path + "/metrics.pkl", "rb"))
plot_things(metric2, path_base, mode, "map", legend=True)


mode = "stationary"
map_name = 'philly_cc'
path_base = "/data/activescout/data_gtmap/"+map_name
path = "/data/activescout/data_nerf/"+map_name+"/s88_"+mode+"_nerf"
metric = pickle.load(open(path + "/metrics.pkl", "rb"))
for ii in range(20):
    metric['targ-%d'%ii] = metric['targ-%d'%ii][:500]
plot_things(metric, path_base, mode, "nerfMI")


exp_name = "/s88_"+mode+"_MI"
path = path_base + exp_name
metric1 = pickle.load(open(path + "/metrics.pkl", "rb"))
for ii in range(20):
    metric1['targ-%d'%ii] = metric1['targ-%d'%ii][:500]
plot_things(metric1, path_base, mode, "MI")

exp_name = "/s88_"+mode+"_map"
path = path_base + exp_name
metric2 = pickle.load(open(path + "/metrics.pkl", "rb"))
for ii in range(20):
    metric2['targ-%d'%ii] = metric2['targ-%d'%ii][:500]
plot_things(metric2, path_base, mode, "map", legend=True)


mode ='active'
map_name = 'philly_cc'
path = "/data/activescout/data_nerf/"+map_name+"/s88_"+mode+"_nerf_2000"
metric1 = pickle.load(open(path + "/metrics.pkl", "rb"))
plot_things(metric1, path_base, mode, "nerf_2000")

map_name = 'nyc_stuy'
path = "/data/activescout/data_nerf/"+map_name+"/s88_"+mode+"_nerf_2000"
metric2 = pickle.load(open(path + "/metrics.pkl", "rb"))
plot_things(metric2, path_base, mode, "nerf_2000", legend=True)


plt.show()
