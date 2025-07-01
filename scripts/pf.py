import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from ipdb import set_trace as st
import scipy as sp
from shapely.geometry import Polygon, Point
from utils import init_env, building_footprints
from scipy.stats import beta, skewnorm


class particle_filter:
    def __init__(sf, cfg, mapDim, num_targets):
        sf.mapDim = mapDim
        sf.num_targets = num_targets
        sf.Nx = sf.mapDim["Nx"]
        sf.Ny = sf.mapDim["Ny"]
        sf.N = sf.Nx * sf.Ny
        x = np.linspace(sf.mapDim["x_min"], sf.mapDim["x_max"], sf.Nx)
        y = np.linspace(sf.mapDim["y_min"], sf.mapDim["y_max"], sf.Ny)
        sf.Py, sf.Px = np.meshgrid(y, x)
        z = np.ones(int(sf.Nx * sf.Ny)) * sf.mapDim["z_min"]
        ## transpose breaks c-contiguous memory , copy fixes that issue for mgl convention
        # particle position in world frame
        sf.p = np.vstack((sf.Px.flatten(), sf.Py.flatten(), z)).T.copy(order="C")
        sf.p = np.array(sf.p, dtype="f4")

        ## get building footprints
        sf.buildingsXY = building_footprints(cfg["osm_pbf_file"], cfg["coord_origin"])
        # id particles that are inside and outside buildings
        points = [Point(x, y) for x, y in zip(sf.Px.flatten(), sf.Py.flatten())]
        sf.inside = []
        sf.outside = []
        for ii, point in enumerate(points):
            if sf.buildingsXY.contains(point).any() == True:
                sf.inside.append(ii)
            else:
                sf.outside.append(ii)

        ## particles locations that are for the target not seen by the scout
        sf.p_targ = np.copy(sf.p)
        ## delete particles that are inside buildings for target particles
        sf.p_targ = sf.p_targ[sf.outside]
        sf.p_targ_og = np.copy(sf.p_targ)

        ## zero out weights that are inside buildings
        sf.w0 = np.ones(sf.N) / float(sf.N)
        sf.w0[sf.inside] = 0.0
        sf.w0 = sf.w0 / np.sum(sf.w0)
        # weight dict for multiple targets
        sf.w = {"targ-%d" % i: np.copy(sf.w0) for i in range(num_targets+1)}
        # mmse estimate dict for targets
        sf.mmse = {"targ-%d" % i: np.zeros(3) for i in range(num_targets)}

        # motion model kernel, to make symetric sigma*2+1 = size
        sf.T = sf.kernel(size=5,sigma=2)
        # plt.imshow(sf.T, extent=[-2.5,2.5,-2.5,2.5]);
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig("motion_model_kernel.pdf", bbox_inches="tight")
        # plt.show()

    def motion_model(sf):
        for targ_name in sf.w.keys():
            # wave like kernel to simulate motion
            sf.w[targ_name] = sp.signal.convolve2d(
                sf.w[targ_name].reshape(sf.Nx, sf.Ny), sf.T, mode="same"
            ).flatten()
            
            # gaussian conv to smooth out the weights
            # sf.w[targ_name] = sp.ndimage.gaussian_filter(
            #     sf.w[targ_name].reshape(sf.Nx, sf.Ny), sigma=0.5
            # ).flatten()
    
            sf.w[targ_name] /= sf.w[targ_name].sum()
            
    def update_weights(sf, obs_idx, targ_dict):
        """
        update weights based on observations and target observations
        obs_idx is a list of indexes of particles that observed
        targ_idx is a tuple of (pos, idx)
        """
        No = 10 # number of observations
        z = np.random.binomial(n=No, p=0.05, size=sf.N)

        for targ_name, targ_info in targ_dict.items():
            # targ_w = targ_dict["world"]
            targ_idx = targ_info["idx"]

            # update weights observed
            if obs_idx.size != 0:
                # sf.w[targ_name][obs_idx] = 0.0
                success = np.nonzero(z[obs_idx])[0]
                sf.w[targ_name][obs_idx[success]] = 0.0

            # update weights with targ idx
            if targ_idx.size != 0:
                # sf.w[targ_name][targ_idx] = 1.0
                sf.w[targ_name][targ_idx] = 1 - (z/No)[targ_idx]

            # normalize
            sf.w[targ_name] /= sf.w[targ_name].sum()

    def mmse_estimate(sf):
        for targ_name in sf.w.keys():
            top = 500
            localmax = np.argpartition(sf.w[targ_name], -top)[-top:]
            weight = sf.w[targ_name][localmax]
            weight /= weight.sum()
            sf.mmse[targ_name] = weight @ sf.p[localmax,:]
        return sf.mmse

    def clear_footprints(sf,):
        for ii in range(sf.num_targets):
            # zero out weights that are inside buildings
            sf.w["targ-%d" % ii][sf.inside] = 0.0
            sf.w["targ-%d" % ii] /= sf.w["targ-%d" % ii].sum()

    def calc_mutual_info(sf, w):
        return sf.mutual_info(sf.w, w)

    def gaussian_kernel(sf, a, b, sigma=1.0):
        error = a - b
        return np.exp(-(error**2) / (2 * sigma**2))

    def mutual_info(sf, a, b):
        return sf.entropy(a) - sf.entropy(b)

    def entropy(sf, w):
        return -np.sum((w + 1e-6) * np.log(w + 1e-6))

    def kernel(sf, size, sigma):
        mid_m, mid_p = int(size*0.33), int(size*0.66+1)
        x, y = np.meshgrid(np.arange(mid_m,mid_p), np.arange(mid_m,mid_p))
        middle = 1-np.exp(-((x-size//2)**2 + (y-size//2)**2)/(2*2**2))
        
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        kernel = np.exp(-((x-size//2)**2 + (y-size//2)**2)/(2*sigma**2))
        kernel[mid_m:mid_p,mid_m:mid_p] = middle
        kernel /= np.sum(kernel)

        return kernel

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

def main():
    map_name = "philly_cc"
    cfg, mapDim = init_env(map_name)
    pf = particle_filter(cfg, mapDim)
    z = np.array([0, 1])
    pf.update_weights(z)
    plt.imshow(pf.w)
    plt.show()
    st()


if __name__ == "__main__":
    main()
