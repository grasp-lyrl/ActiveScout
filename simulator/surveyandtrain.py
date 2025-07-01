import numpy as np
import moderngl as mgl
import moderngl_window as mglw
from moderngl_window.conf import settings
from moderngl_window.timers.clock import Timer
from moderngl_window import resources
from moderngl_window.scene.camera import KeyboardCamera
from moderngl_window import geometry
from pathlib import Path
from pyrr import Matrix44, Vector3
from PIL import Image
from ipdb import set_trace as st
import matplotlib.pyplot as plt
import imgui
import torch as th
from moderngl_window.integrations.imgui import ModernglWindowRenderer
import sys, click, pickle, os
from scipy.spatial.transform import Rotation as R
from mgl_imgui_simOSM import mgl_imgui_simOSM as sim
sys.path.append("scripts")
from agents import scout
from pf import particle_filter


sys.path.append("perception")
from nerf_scripts.nerf_predictive_info import ActiveNeRFMapper


@click.command()
@click.option("--map_name", default="philly_cc", type=str)
def main(map_name):
    sys.path.append("scene_config")
    if map_name == "philly_cc":
        from philly_cc import cfg, mapDim
    if map_name == "nyc_stuy":
        from nyc_stuy import cfg, mapDim
    if map_name == "nyc_downtown":
        from nyc_downtown import cfg, mapDim



    images = []
    depths = []
    cam_poses = []
    Ts = []
    

    agent = scout(
                cfg,
                pf=particle_filter(cfg, mapDim, num_targets=4),
                sim=sim(cfg, headless=False),
            )
    
    zs = [10,20,30,40,50,60,70,80,90,100,110,130,150,170,190,210]#,230,250,270,290,300]
    # pitches = [-1,-1,-1,-30,-45,-1,-1,-45,-1,-5,-10,-15,-20,-25,-30,-35,-40,-45,-60,-75,-80]
    pitches = [-1,-1,15,-30,-45,30,-1,-45,-1,-5,-45,-50,-89,-89,-75,-80]#,-40,-45,-60,-75,-80]
    for z, pitch in zip(zs,pitches):
        traj = agent.survey_traj(traj_len=20, survey_z=z, survey_pitch=pitch)

        for state in traj:
            _, _, _, real_data = agent.sim.run_view2particles(
                state[:3], state[3], state[4], [], True
            )
            images.append(real_data['image'])
            depths.append(real_data['depth'])
            # cam_poses.append(cam_pose)
            Ts.append(real_data['pose'])

    
    images = np.array(images)
    depths = np.array(depths)
    # cam_poses = np.array(cam_poses)
    Ts = np.array(Ts)
    
    
    data = {"images":images,
            "depths":depths,
            # "cam_poses":cam_poses,
            "poses":Ts}
    
    save_dir = './data/'+map_name+'/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pickle.dump(data, open(save_dir+map_name+".pkl", "wb"))
    cam_poses = []

    nerf = ActiveNeRFMapper(cfg)
    nerf.initialization(data, data)
    nerf.nerf_training(steps=10000, planning_step=-1)
    nerf.render(data)
    nerf.nerf_training(steps=2000, final_train=True, planning_step=-10)
    nerf.render(data)

    # st()





if __name__ == "__main__":
    th.cuda.empty_cache
    main()
