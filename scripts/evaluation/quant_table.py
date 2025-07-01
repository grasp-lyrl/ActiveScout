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

Table II.
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


def quant(metric):
    targ0 = np.array(metric['targ-0'])
    targ1 = np.array(metric['targ-1'])
    targ2 = np.array(metric['targ-2'])
    targ3 = np.array(metric['targ-3'])

    if targ0.shape[0] > 1500:
        targ0 = targ0[1274:]
        targ1 = targ1[1274:]
        targ2 = targ2[1274:]
        targ3 = targ3[1274:]

    targ0_min = targ0.min()
    targ0_max = targ0.max()
    targ1_min = targ1.min()
    targ1_max = targ1.max()
    targ2_min = targ2.min()
    targ2_max = targ2.max()
    targ3_min = targ3.min()
    targ3_max = targ3.max()
    targ0_avg = targ0.mean()
    targ1_avg = targ1.mean()
    targ2_avg = targ2.mean()
    targ3_avg = targ3.mean()

    min_avg = (targ0_min + targ1_min + targ2_min + targ3_min) / 4
    max_avg = (targ0_max + targ1_max + targ2_max + targ3_max) / 4 
    mean_avg = (targ0_avg + targ1_avg + targ2_avg + targ3_avg) / 4
    # spread = max_avg - min_avg

    return mean_avg, min_avg, max_avg

def over_seeds(path, metric1, metric2, metric3):
    mean_avg1, min_avg1, max_avg1 = quant(metric1)
    mean_avg2, min_avg2, max_avg2 = quant(metric2)
    mean_avg3, min_avg3, max_avg3 = quant(metric3)

    mean_ = np.array([mean_avg1, mean_avg2, mean_avg3])
    min_ = np.array([min_avg1, min_avg2, min_avg3])
    max_ = np.array([max_avg1, max_avg2, max_avg3])

    mean_avg = mean_.mean()
    min_avg = min_.mean()
    max_avg = max_.mean()
    mean_std = mean_.std()
    min_std = min_.std()
    max_std = max_.std()


    print(path, "mean_avg:%f+-%f"%(mean_avg,mean_std), "min_avg:%f+-%f"%(min_avg,min_std), "max_avg:%f+-%f"%(max_avg,max_std))
    return


mode = "active"
map_name = 'philly_cc'
path = "/data/activescout/data_gtmap/"+map_name+"/s72_"+mode+"_map"
metric1 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_gtmap/"+map_name+"/s80_"+mode+"_map"
metric2 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_gtmap/"+map_name+"/s88_"+mode+"_map"
metric3 = pickle.load(open(path + "/metrics.pkl", "rb"))
over_seeds(map_name+"map",metric1, metric2, metric3)

mode = "active"
map_name = 'philly_cc'
path = "/data/activescout/data_gtmap/"+map_name+"/s72_"+mode+"_MI"
metric1 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_gtmap/"+map_name+"/s80_"+mode+"_MI"
metric2 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_gtmap/"+map_name+"/s88_"+mode+"_MI"
metric3 = pickle.load(open(path + "/metrics.pkl", "rb"))
over_seeds(map_name+"MI",metric1, metric2, metric3)

mode = "active"
map_name = 'philly_cc'
path = "/data/activescout/data_nerf/"+map_name+"/s72_"+mode+"_nerf"
metric1 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_nerf/"+map_name+"/s80_"+mode+"_nerf"
metric2 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_nerf/"+map_name+"/s88_"+mode+"_nerf"
metric3 = pickle.load(open(path + "/metrics.pkl", "rb"))
over_seeds(map_name+"nerf",metric1, metric2, metric3)


mode = "active"
map_name = 'nyc_stuy'
path = "/data/activescout/data_gtmap/"+map_name+"/s72_"+mode+"_map"
metric1 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_gtmap/"+map_name+"/s80_"+mode+"_map"
metric2 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_gtmap/"+map_name+"/s88_"+mode+"_map"
metric3 = pickle.load(open(path + "/metrics.pkl", "rb"))
over_seeds(map_name+"map",metric1, metric2, metric3)

mode = "active"
map_name = 'nyc_stuy'
path = "/data/activescout/data_gtmap/"+map_name+"/s72_"+mode+"_MI"
metric1 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_gtmap/"+map_name+"/s80_"+mode+"_MI"
metric2 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_gtmap/"+map_name+"/s88_"+mode+"_MI"
metric3 = pickle.load(open(path + "/metrics.pkl", "rb"))
over_seeds(map_name+"MI",metric1, metric2, metric3)

mode = "active"
map_name = 'nyc_stuy'
path = "/data/activescout/data_nerf/"+map_name+"/s72_"+mode+"_nerf"
metric1 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_nerf/"+map_name+"/s80_"+mode+"_nerf"
metric2 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_nerf/"+map_name+"/s88_"+mode+"_nerf"
metric3 = pickle.load(open(path + "/metrics.pkl", "rb"))
over_seeds(map_name+"nerf",metric1, metric2, metric3)



mode = "active"
map_name = 'philly_cc'
path = "/data/activescout/data_nerf/"+map_name+"/s72_"+mode+"_nerf_2000"
metric1 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_nerf/"+map_name+"/s80_"+mode+"_nerf_2000"
metric2 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_nerf/"+map_name+"/s88_"+mode+"_nerf_2000"
metric3 = pickle.load(open(path + "/metrics.pkl", "rb"))
over_seeds(map_name+"nerf_2k",metric1, metric2, metric3)

mode = "active"
map_name = 'nyc_stuy'
path = "/data/activescout/data_nerf/"+map_name+"/s72_"+mode+"_nerf_2000"
metric1 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_nerf/"+map_name+"/s80_"+mode+"_nerf_2000"
metric2 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_nerf/"+map_name+"/s88_"+mode+"_nerf_2000"
metric3 = pickle.load(open(path + "/metrics.pkl", "rb"))
over_seeds(map_name+"nerf_2k",metric1, metric2, metric3)

mode = "active"
map_name = 'philly_cc'
path = "/data/activescout/data_nerf/"+map_name+"/s72_"+mode+"_nerf_offline"
metric1 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_nerf/"+map_name+"/s80_"+mode+"_nerf_offline"
metric2 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_nerf/"+map_name+"/s88_"+mode+"_nerf_offline"
metric3 = pickle.load(open(path + "/metrics.pkl", "rb"))
over_seeds(map_name+"nerf_offline",metric1, metric2, metric3)

mode = "active"
map_name = 'nyc_stuy'
path = "/data/activescout/data_nerf/"+map_name+"/s72_"+mode+"_nerf_offline"
metric1 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_nerf/"+map_name+"/s80_"+mode+"_nerf_offline"
metric2 = pickle.load(open(path + "/metrics.pkl", "rb"))
path = "/data/activescout/data_nerf/"+map_name+"/s88_"+mode+"_nerf_offline"
metric3 = pickle.load(open(path + "/metrics.pkl", "rb"))
over_seeds(map_name+"nerf_offline",metric1, metric2, metric3)

