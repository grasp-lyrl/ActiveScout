import numpy as np
import moderngl as mgl
import moderngl_window as mglw
from moderngl_window.conf import settings
from moderngl_window.timers.clock import Timer
from moderngl_window import resources
from moderngl_window.scene.camera import KeyboardCamera, OrbitCamera
from moderngl_window import geometry
from pathlib import Path
from pyrr import Matrix44, Vector3
from PIL import Image
from ipdb import set_trace as st
import matplotlib.pyplot as plt
import imgui
from moderngl_window.integrations.imgui import ModernglWindowRenderer
import sys, click, pickle

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
fsz = 15
plt.rc("font", size=fsz)
plt.rc("axes", titlesize=fsz)
plt.rc("axes", labelsize=fsz)
plt.rc("xtick", labelsize=fsz)
plt.rc("ytick", labelsize=fsz)
plt.rc("legend", fontsize=0.7 * fsz)
plt.rc("figure", titlesize=fsz)
plt.rc("pdf", fonttype=42)
sns.set_style("ticks", rc={"axes.grid": True})

s_colors = np.array(sns.color_palette())
ind = np.array([3, 2, 4, 1], dtype=int)
colors = s_colors[ind]


def render_plot(step, pf, fig, metric, a_traj,t_traj, num_targets=4):
    fig.clf()
    ax = fig.add_subplot()
    # plot buildings footprints
    pf.buildingsXY.plot(ax=ax, color="black", alpha=0.3)

    for jj in range(num_targets):
        top = 8000
        # st()
        weight = t_traj["weights"][step]["targ-%d" % jj]
        # st()
        localmax = np.argpartition(weight, -top)[-top:]
        # st()
        alpha = weight[localmax] * 500
        alpha[alpha > 1] = 1.0
        alpha[alpha < 0] = 0.0
        ax.scatter(
            pf.p[localmax, 0],
            pf.p[localmax, 1],
            marker=".",
            # label="pf max %d" % jj,
            color=colors[jj],
            alpha=alpha,
        )
    try:
        ax.scatter(
            pf.p[:, 0][a_traj["obs"][step]],
            pf.p[:, 1][a_traj["obs"][step]],
            marker=".",
            label="$y_{detect}$",
            alpha=0.8,
        )
    except:
        pass

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
            label="target-%d" % (jj),
            color=colors[jj],
            edgecolors="black",
            linewidths=0.7,
        )

    ax.scatter(
        a_traj["agent-0"][0, 0],
        a_traj["agent-0"][0, 1],
        # label="scout",
        color=s_colors[0],
        marker='x',
        # edgecolors="black",
        # linewidths=0.7,
    )
    for jj in range(num_targets):
        ax.scatter(
            t_traj["targ-%d" % jj][0, 0],
            t_traj["targ-%d" % jj][0, 1],
            # label="target-%d" % (jj),
            color=colors[jj],
            # edgecolors="black",
            # linewidths=0.7,
            marker= 'x',
        )

    # plot trajectories
    ax.plot(
        a_traj["agent-%d" % 0][:step, 0],
        a_traj["agent-%d" % 0][:step, 1],
        color=s_colors[0],
        # alpha=0.4,
    )
    for jj in range(num_targets):
        ax.plot(
            t_traj["targ-%d" % jj][:step, 0],
            t_traj["targ-%d" % jj][:step, 1],
            color=colors[jj],
            # alpha=0.4,
        )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.001)

class mgl_imgui_simOSM(mglw.WindowConfig):
    resources.register_dir(Path(__file__).parent.parent.resolve())
    """
    Custom setup using a class.
    We create the window, main loop and register events.
    """

    def __init__(self, cfg, headless=True):
        # scene configurations
        self.cfg = cfg
        # Configure to use pyglet window
        if headless == True:
            settings.WINDOW["class"] = "moderngl_window.context.headless.Window"
        else:
            settings.WINDOW["class"] = "moderngl_window.context.pyglet.Window"
            settings.WINDOW["title"] = self.cfg["name"]
            # settings.WINDOW["aspect_ratio"] = 1.0

        self.wnd = mglw.create_window_from_settings()
        self.ctx = self.wnd.ctx
        # self.window_size = (1280, 720)
        self.window_size = (640, 360)
        # self.window_size = (320,320)

        # register event methods
        self.wnd.resize_func = self.resize
        self.wnd.iconify_func = self.iconify
        self.wnd.key_event_func = self.key_event
        self.wnd.mouse_position_event_func = self.mouse_position_event
        self.wnd.mouse_drag_event_func = self.mouse_drag_event
        self.wnd.mouse_scroll_event_func = self.mouse_scroll_event
        self.wnd.mouse_press_event_func = self.mouse_press_event
        self.wnd.mouse_release_event_func = self.mouse_release_event
        self.wnd.unicode_char_entered_func = self.unicode_char_entered
        self.wnd.close_func = self.close

        # imgui
        imgui.create_context()
        self.wnd.ctx.error
        self.imgui = ModernglWindowRenderer(self.wnd)

        # load osm obj scene
        self.scene = self.load_scene(self.cfg["obj_file"], cache=True)

        # self.camera = OrbitCamera(radius = 100.0, angles=(98.34, -45), aspect_ratio=self.wnd.aspect_ratio)
        # self.camera_enabled = True
        # self.near = 0.1
        # self.far = 5000.0
        # self.camera_enabled = True
        # self.camera.projection.update(near=self.near, far=self.far)
        # self.camera.velocity = 100.0
        # self.camera.mouse_sensitivity = 0.3
        # free control over camera vs not. see key_event for online switching
        self.free = False
        self.oview = self.cfg['oview']
        self.tpv = False

        # hallucinate frame buffer
        self.hallucinate_depth = self.ctx.depth_texture(self.window_size)
        self.hallucinate_depth.compare_func = ""
        self.hallucinate_depth.repeat_x = False
        self.hallucinate_depth.repeat_y = False
        self.hallucinate_fbo = self.ctx.framebuffer(
            color_attachments=self.ctx.texture(self.window_size, 4),
            depth_attachment=self.hallucinate_depth,
        )
        self.imgui.register_texture(self.hallucinate_fbo.color_attachments[0])

        # in situ frame buffer
        self.insitu_depth = self.ctx.depth_texture(self.window_size)
        self.insitu_depth.compare_func = ""
        self.insitu_depth.repeat_x = False
        self.insitu_depth.repeat_y = False
        self.insitu_fbo = self.ctx.framebuffer(
            color_attachments=self.ctx.texture(self.window_size, 4),
            depth_attachment=self.insitu_depth,
        )
        self.imgui.register_texture(self.insitu_fbo.color_attachments[0])

        # in situ frame buffer
        self.insitu_tpv_depth = self.ctx.depth_texture(self.window_size)
        self.insitu_tpv_depth.compare_func = ""
        self.insitu_tpv_depth.repeat_x = False
        self.insitu_tpv_depth.repeat_y = False
        self.insitu_tpv_fbo = self.ctx.framebuffer(
            color_attachments=self.ctx.texture(self.window_size, 4),
            depth_attachment=self.insitu_tpv_depth,
        )
        self.imgui.register_texture(self.insitu_tpv_fbo.color_attachments[0])

        # depth render display top right
        self.depth_quad = geometry.quad_2d(size=(0.5, 0.5), pos=(0.75, 0.75))
        # self.insitu_quad = geometry.quad_2d(size=(0.5, 0.5), pos=(0.0, 0.75))

        self.floor = geometry.cube(size=(2000.0, 1.0, 1000.0), center=(0.0, -1.0, 0.0))

        # Programs
        self.floor_program = self.load_program("simulator/programs/world2pixel.glsl")
        self.floor_program["color"].value = 0.0, 0.0, 0.0, 1.0

        self.lin_depth_prog = self.load_program(
            "simulator/programs/linearized_depth.glsl"
        )
        self.near = 0.1
        self.far = 5000
        self.lin_depth_prog["near"].value = self.near
        self.lin_depth_prog["far"].value = self.far

        # use must run self.init_objects
        self.init_obj = False

        self.colors = [
            ((1.0, 0.0, 0.0, 1.0), (255, 0, 0)),  # red
            ((0.0, 1.0, 0.0, 1.0), (0, 255, 0)),  # green
            ((1.0, 0.0, 1.0, 1.0), (255, 0, 255)),  # magenta
            ((1.0, 1.0, 0.0, 1.0), (255, 255, 0)),  # yellow
            ((0.0, 1.0, 1.0, 1.0), (0, 255, 255)),  # cyan
            ((0.0, 0.0, 1.0, 1.0), (0, 0, 255)),  # blue
        ]

    def init_objects(self, particles):
        self.init_obj = True
        # particle filter
        # [x,y,z] -> [x,z,y]
        self.grid = np.vstack(
            (particles[:, 0], particles[:, 2], particles[:, 1])
        ).T.copy(order="C")
        self.Ngrid = int(self.grid.shape[0])
        self.particles_program = self.load_program(
            "simulator/programs/world2pixel.glsl"
        )
        # agent model used when in third person view
        self.agent = geometry.cube(size=(8.0, 2.0, 8.0))
        self.agent_program = self.load_program("simulator/programs/world2pixel.glsl")
        # self.agent_program["color"].value = 1.0, 165/255., 0.0, 1.0
        self.agent_program["color"].value = 0.29803922, 0.44705882, 0.69019608, 1.0
        self.agent_poses = []
        self.agent_marker = geometry.cube(size=(20,50,20))
        # self.agent_frustrum = 
        self.trail = geometry.sphere(radius=2.0)
        self.trail_program = self.load_program("simulator/programs/world2pixel.glsl")
        
        # particles 
        self.particles_program["color"].value = 1.0, 1.0, 1.0, 1.0
        self.particles_vbo = self.ctx.buffer(self.grid)
        self.particles_vao = self.ctx.vertex_array(
            self.particles_program, self.particles_vbo, "in_position"
        )

        # target
        self.target_program = self.load_program("simulator/programs/world2pixel.glsl")
        self.target_program["color"].value = 1.0, 0.0, 0.0, 1.0
        self.target_vbo = self.ctx.buffer(np.empty((4)))  # target (x,y,z,yaw)
        self.target_vao = self.ctx.vertex_array(
            self.target_program, self.target_vbo, "in_position"
        )
        self.target = geometry.cube(size=(5,5,5))
        self.target_marker = geometry.cube(size=(20,50,20))
        # particles and target render point size
        self.ctx.point_size = 1.0
        self.step = 0

    def render(self, time: float, frametime: float):
        """
        use for free view with mouse and keyboard control
        """
        if self.wnd.name == "headless":
            raise RuntimeError("This example only works with headless = False")        

        # set background to all black
        # self.ctx.clear(0.0, 0.0, 0.0)
        self.ctx.enable_only(mgl.DEPTH_TEST | mgl.CULL_FACE)

        # # set camera position and orientation
        if self.free == False:
            # (x,z,-y)
            self.yaw = -90.0
            self.pitch = -89.99
            self.camera.set_position(5 * time, 500, 5 * time)
            # (yaw,pitch)
            self.camera.set_rotation(self.yaw, self.pitch)

        if self.oview:
            self.agent_program["position"].value = (self.camera.position)
            self.agent_poses.append(self.camera.position)
            if len(self.agent_poses) > 50:
                    self.agent_poses.pop(0)
            
        self.targ_pos = np.array([[0, 5 * time, 10], [5 * time, 0, 10]])
        ## set camera params, render scene, particles, targets
        self.in_situ = True
        self.render_func(time)
        ## use buffers to get image and compute particles in world
        # self.image2world(render=True)
        # renders depth in top right
        self.depth_quad.render(self.lin_depth_prog)
        # self.insitu_quad.render()
        # render imgui
        self.render_ui()

    def get_view2particleidx(
        self, time, ego_pos, yaw, pitch, targ_pos, in_situ, save_img=-1
    ):
        """
        Given camera pose return particles in view
        """
        self.yaw = yaw
        self.pitch = pitch
        if in_situ and save_img==-10:
            self.oview = True
            self.tpv = True
            # self.camera = OrbitCamera(radius = 100.0, aspect_ratio=self.wnd.aspect_ratio)
            self.camera = OrbitCamera(radius = 100.0, angles=(-90., -45), aspect_ratio=self.wnd.aspect_ratio)
            self.camera_enabled = True
            self.camera.projection.update(near=self.near, far=self.far)
            self.camera.velocity = 100.0
            self.camera.mouse_sensitivity = 0.3
        elif in_situ and save_img!= -10:
            self.oview = True
            self.tpv = False
            # self.camera = OrbitCamera(radius = 100.0, angles=(-90., -15), aspect_ratio=self.wnd.aspect_ratio)
            self.camera = OrbitCamera(radius = 100.0, angles=(-90., -30), aspect_ratio=self.wnd.aspect_ratio)
            self.camera_enabled = True
            self.camera.projection.update(near=self.near, far=self.far)
            self.camera.velocity = 100.0
            self.camera.mouse_sensitivity = 0.3
        else:
            self.oview = False
            self.tpv = False
            self.camera = KeyboardCamera(self.wnd.keys, fov=60.0, aspect_ratio=self.wnd.aspect_ratio)
            self.camera_enabled = True
            self.camera.projection.update(near=self.near, far=self.far)
            self.camera.velocity = 100.0
            self.camera.mouse_sensitivity = 0.3
        # set background to all black
        self.ctx.clear(0.0, 0.0, 0.0)
        self.ctx.enable_only(mgl.DEPTH_TEST | mgl.CULL_FACE)

        # convert from (x,y,z) to (x,z,-y)
        self.ego_pos = ego_pos
        if not self.oview and not self.tpv:
            self.camera.set_position(self.ego_pos[0], self.ego_pos[2], -self.ego_pos[1])
            # # (yaw,pitch)
            if pitch > 85.0:
                pitch = 85.0
            if pitch < -85.0:
                pitch = -85.0

            self.camera.set_rotation(self.yaw, self.pitch)
            self.camera.target = (self.ego_pos[0], self.ego_pos[2], -self.ego_pos[1])
            self.camera.radius = 1.0
            self.camera.angles = (0,0)
        elif self.tpv and self.oview:
            self.camera.set_rotation(-90., 0.0)
            self.camera.target = (self.ego_pos[0], self.ego_pos[2], -self.ego_pos[1])
            self.agent_program["position"].value = (self.ego_pos[0], self.ego_pos[2], -self.ego_pos[1])
            self.agent_poses.append((self.ego_pos[0], self.ego_pos[2], -self.ego_pos[1]))
            if len(self.agent_poses) > 200:
                    self.agent_poses.pop(0)
        else:
            self.camera.set_rotation(-90., 0.0)
            self.camera.target = (0,600,400)
            # self.camera.target = (100,400,200)
            self.agent_program["position"].value = (self.ego_pos[0], self.ego_pos[2], -self.ego_pos[1])
            self.agent_poses.append((self.ego_pos[0], self.ego_pos[2], -self.ego_pos[1]))
            if len(self.agent_poses) > 200:
                    self.agent_poses.pop(0)

        self.targ_pos = targ_pos
        # set camera params, render scene, particles, target
        self.in_situ = in_situ
        self.render_func(time)
        # render imgui
        self.render_ui()


        # if self.step >3000:
        #     image = Image.frombytes("RGB", self.insitu_fbo.size, self.insitu_fbo.read())
        #     rgb_image = image.transpose(
        #         Image.FLIP_TOP_BOTTOM
        #     )  # only do this to save image
        
        #     rgb_image.save(
        #         self.cfg["data_dir"] + "images/step-%d.png" % (self.step), format="png"
        #     )

        self.step += 1

    def render_func(self, time):
        """
        set camera params, render scene, particles, targets
        """
        # render everthing else at camera location
        translation = Matrix44.from_translation((0, 0, 0))
        rotation = Matrix44.from_eulers((0, 0, 0))
        model_matrix = translation * rotation
        # P*V*M
        camera_matrix = self.camera.matrix * model_matrix
        self.depth_mvp = self.camera.projection.matrix * camera_matrix
        self.agent_program["mvp"].write(self.depth_mvp.astype("f4"))
        self.trail_program["mvp"].write(self.depth_mvp.astype("f4"))
        self.particles_program["mvp"].write(self.depth_mvp.astype("f4"))
        self.floor_program["mvp"].write(self.depth_mvp.astype("f4"))

        # target vertex: needed to visualize target in correct location
        # as it is originally put at origin and translated by targ program "position"
        translation = Matrix44.from_translation((0, 0, 0))
        rotation = Matrix44.from_eulers((0, 0, 0))
        targ_model_matrix = translation * rotation
        # P*V*M
        targ_camera_matrix = self.camera.matrix * targ_model_matrix
        self.target_depth_mvp = self.camera.projection.matrix * targ_camera_matrix
        self.target_program["mvp"].write(self.target_depth_mvp.astype("f4"))

        def render_things(self):
            def interpolate_color(value):
                blue = (1 - value)
                red = value
                return (blue, 0, red)
            self.scene.draw(
                projection_matrix=self.camera.projection.matrix,
                camera_matrix=camera_matrix,
                time=time,
            )
            if self.oview:
                self.agent.render(self.agent_program)
                self.agent_program["position"].value = (self.ego_pos[0], 200, -self.ego_pos[1])
                self.agent_marker.render(self.agent_program)
                agent_poses_plot = self.agent_poses[:-1]
                for i, ex_pose in enumerate(reversed(agent_poses_plot)):
                    color_value = i / 200
                    color = interpolate_color(color_value)
                    self.trail_program['color'].value = (color[0], color[1], color[2], 1.0)
                    self.trail_program['position'].value = ex_pose
                    self.trail.render(self.trail_program)
            self.ctx.point_size = 1.0
            if not self.oview:
                self.particles_vao.render(mgl.POINTS, self.Ngrid)
            self.render_target(pos=self.targ_pos)

        # --- PASS 1: Render shadow map
        if self.in_situ and self.tpv:
            # Render to insitu window
            self.insitu_tpv_fbo.clear()
            self.insitu_tpv_fbo.use()
            render_things(self)
            # use depth texture when in-situ
            self.insitu_tpv_depth.use(location=0)
        elif self.in_situ and not self.tpv:
            self.insitu_fbo.clear()
            self.insitu_fbo.use()
            render_things(self)
            # use depth texture when in-situ
            self.insitu_depth.use(location=0)
        else:
            # Render to hallucinate window
            self.hallucinate_fbo.clear()
            self.hallucinate_fbo.use()
            render_things(self)
            # use hallucinate depth texture when not in-situ
            self.hallucinate_depth.use(location=0)

        # --- PASS 2: Render scene to screen
        self.wnd.use()

    def render_target(self, pos):
        """
        Render target at pos, we translate from in_position = (0,0,0) to pos
        pos = (x,y,z) -> (x,z,-y)
        """
        if len(pos) != 0:
            for ii in range(pos.shape[0]):
                self.ctx.point_size = self.cfg["target_size"]
                # get rgba color in opengl form
                if self.cfg['target_mode'] == "stationary":
                    self.target_program["color"].value = (1. - ii/255, 0.,0.,1.)
                else:
                    self.target_program["color"].value = self.colors[ii][0]
                
                try:  # for multiple targets
                    self.target_program["position"].value = (
                        pos[ii, 0],
                        pos[ii, 2],
                        -pos[ii, 1],
                    )
                except:  # for 1 target
                    self.target_program["position"].value = pos[0], pos[2], -pos[1]                    
                
                if self.oview:
                    self.target.render(self.target_program)
                    self.target_program["position"].value = (pos[ii,0], 200, -pos[ii,1])
                    self.target_marker.render(self.target_program)
                else:
                    self.target.render(self.target_program)
                    # self.target_vao.render(mgl.POINTS, 1)

    def image2world(self, render=False, save_img=-1):
        """
        image from current view due to pose
        to get particles in world,
        indices of particles seen, and target in world
        based on color of pixels
        """

        # from framebuffer get rgb data
        # image = Image.frombytes("RGB", self.wnd.fbo.size, self.wnd.fbo.read())
        if self.in_situ:
            # image = Image.frombytes("RGB", self.wnd.fbo.size, self.wnd.fbo.read())
            image = Image.frombytes("RGB", self.insitu_fbo.size, self.insitu_fbo.read())
        else:
            image = Image.frombytes(
                "RGB", self.hallucinate_fbo.size, self.hallucinate_fbo.read()
            )
        if save_img != -1:
            rgb_image = image.transpose(
                Image.FLIP_TOP_BOTTOM
            )  # only do this to save image
            rgb_image.save(
                self.cfg["data_dir"] + "images/step-%d.png" % (save_img), format="png"
            )

        img = np.array(image)
        # find row,col of the image that has white pixels [255,255,255]
        v, u = np.where(
            (img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255)
        )

        p_w = self.pixel2world(u, v, self.depth_mvp)
        update_idx = self.world2particleidx(p_w)

        # find row,col of the image that has red pixels [255,0,0] (target)
        v, u = np.where(
            (img[:, :, 0] == 255) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
        )
        # we use depth_mvp because we want to get the target in world coordinates
        target_w = self.pixel2world(u, v, self.depth_mvp)
        targ_idx = self.world2particleidx(target_w)

        if render:
            # debug
            plt.clf()
            plt.scatter(self.grid[:, 0], self.grid[:, 2])
            plt.scatter(p_w[:, 0], p_w[:, 2])
            plt.scatter(self.grid[:, 0][update_idx], self.grid[:, 2][update_idx])
            try:
                plt.scatter(target_w[:, 0], target_w[:, 2])
                plt.scatter(self.grid[:, 0][targ_idx], self.grid[:, 2][targ_idx])
            except:
                pass
            plt.draw()
            plt.pause(0.0001)

        return p_w, update_idx, target_w, targ_idx

    def pixel2world(self, u: list, v: list, mvp):
        """
        take lists of pixel coordinates [u,v] and output 2d array of world coordinates [x,z,y,1]
        code  is vectorized so the lists need to have a certain dim
        """
        if len(u) == 0:
            return np.array([])

        # revert to clip space, the + 0.5 is for the center of the pixel
        uclip = 2.0 * (u + 0.5) / self.window_size[0] - 1.0
        vclip = 2.0 * (v + 0.5) / self.window_size[1] - 1.0

        # from depth buffer get depth
        if self.in_situ:
            depth = np.frombuffer(self.insitu_depth.read(), dtype="f4")
        else:
            depth = np.frombuffer(self.hallucinate_depth.read(), dtype="f4")
        depth = depth.reshape(self.window_size[1], self.window_size[0])
        depth_clip = 2.0 * depth - 1.0

        # image [u,v,d,1] to world [x,z,y,1]
        img_coord = np.stack((uclip, vclip, depth_clip[v, u], np.ones(len(u)))).T
        inv_mvp = np.linalg.inv(np.array(mvp))
        world_coord = img_coord @ inv_mvp
        # normalize by w, need to do some broadcasting
        p_w = world_coord / world_coord[:, 3][:, None]
        # [x,z,y,1] -> [x,z,-y,1]
        p_w[:, 2] = -p_w[:, 2]

        return p_w

    def world2pixel(self, world):
        """
        debug forward process. from world to image pixels
        """
        # ex1 = np.array([100, 1, 160, 1], dtype="f4")
        # openGL is column major
        cam = np.array(self.depth_mvp).T @ world
        cam /= cam[3]
        xw = int((cam[0] + 1.0) * (self.window_size[0] / 2))
        yw = int((cam[1] + 1.0) * (self.window_size[1] / 2))
        p_w = self.pixel2world([xw], [yw])
        st()

    def world2particleidx(self, p_w):
        """
        given a 2d array of world particles [x,z,y,1]
        return the grid index of the matched particle
        """
        if p_w.size == 0:
            return np.array([])
        # only compare x and y and get indices of updated particles
        gridxy = np.stack((self.grid[:, 0], self.grid[:, 2]))
        pwxy = np.stack((p_w[:, 0], p_w[:, 2]))

        # use broadcast compute distance between grid (1000,p,2) and pwxy (p,2)
        distances = np.linalg.norm(gridxy[:, np.newaxis, :].T - pwxy.T, axis=2)
        # distances (1000,p)
        update_idx = np.argmin(distances, axis=0)
        return update_idx

    def run_view2particles(self, pos, yaw, pitch, targ_pos, in_situ, save_img=-1):
        """
        Given camera pose, return particles viewed
        We will not close or destroy the window or
        you will need to go through init again
        """
        if self.init_obj == False:
            raise RuntimeError("Be sure to use self.init_objects(particles,targets)")
        timer = Timer()
        timer.start()

        self.wnd.clear()
        time, _ = timer.next_frame()
        update_idx = self.get_view2particleidx(
            time, pos, yaw, pitch, targ_pos, in_situ, save_img
        )
        self.wnd.swap_buffers()

        return update_idx

    def run(self):
        """
        Use run if you want free control with mouse and keyboard
        """
        if self.init_obj == False:
            raise RuntimeError("Be sure to use self.init_objects(particles,targets)")
        timer = Timer()
        timer.start()

        while not self.wnd.is_closing:
            self.wnd.clear()
            time, frame_time = timer.next_frame()
            self.render(time, frame_time)
            self.wnd.swap_buffers()

        self.wnd.destroy()

    def render_ui(self):
        """Render the UI"""
        imgui.new_frame()
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", "Cmd+Q", False, True
                )

                if clicked_quit:
                    exit(1)

                imgui.end_menu()
            imgui.end_main_menu_bar()
        imgui.show_test_window()

        # Create window with the framebuffer image
        imgui.begin("First Person View", True)
        # Create an image control by passing in the OpenGL texture ID (glo)
        # and pass in the image size as well.
        # The texture needs to he registered using register_texture for this to work
        # uv0 and uv1 flip the window
        imgui.image(
            self.hallucinate_fbo.color_attachments[0].glo,
            *self.hallucinate_fbo.size,
            uv0=(0, 1),
            uv1=(1, 0)
        )
        imgui.end()

        imgui.begin("Overview", True)
        imgui.image(
            self.insitu_fbo.color_attachments[0].glo,
            *self.insitu_fbo.size,
            uv0=(0, 1),
            uv1=(1, 0)
        )
        imgui.end()

        imgui.begin("Third Person View", True)
        imgui.image(
            self.insitu_tpv_fbo.color_attachments[0].glo,
            *self.insitu_tpv_fbo.size,
            uv0=(0, 1),
            uv1=(1, 0)
        )
        imgui.end()

        # imgui stuff
        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        # if self.camera_enabled:
            # self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.SPACE:
                self.timer.toggle_pause()
            # toggle between following policy vs mouse and key control
            # if key == keys.PERIOD:
            #     self.free = True
            # if key == keys.COMMA:
            #     self.free = False

        self.imgui.key_event(key, action, modifiers)

    def resize(self, width: int, height: int):
        self.imgui.resize(width, height)

    def mouse_drag_event(self, x, y, dx, dy):
        self.imgui.mouse_drag_event(x, y, dx, dy)

    # def mouse_scroll_event(self, x_offset, y_offset):
    #     self.imgui.mouse_scroll_event(x_offset, y_offset)
    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        if self.camera_enabled:
            self.camera.zoom_state(y_offset)

    def mouse_press_event(self, x, y, button):
        self.imgui.mouse_press_event(x, y, button)

    def mouse_release_event(self, x: int, y: int, button: int):
        self.imgui.mouse_release_event(x, y, button)

    def unicode_char_entered(self, char):
        self.imgui.unicode_char_entered(char)

    def mouse_position_event(self, x: int, y: int, dx, dy):
        if self.camera_enabled:
            self.camera.rot_state(-dx, -dy)

        self.imgui.mouse_position_event(x, y, dx, dy)

    def close(self):
        # print("Window was closed")
        pass


@click.command()
@click.option("--map_name", default="philly_cc", type=str)
@click.option("--mode", default="active", type=str)
def main(map_name, mode):
    sys.path.append("scene_config")
    if map_name == "philly_cc":
        from philly_cc import cfg, mapDim
    if map_name == "nyc_stuy":
        from nyc_stuy import cfg, mapDim
    if map_name == "nyc_downtown":
        from nyc_downtown import cfg, mapDim

    cfg['oview'] = True

    num_targets = 4
    pf = particle_filter(cfg, mapDim, num_targets)


    app = mgl_imgui_simOSM(cfg, headless=False)
    app.init_objects(pf.p)


    fig, ax = plt.subplots()
    # needs to headless=False
    # app.run()

    path = "data_nerf/"+map_name+"/s88_"+mode+"_nerf"
    # path = 'data_gtmap/'+map_name+"/s88_"+mode+"_map"
    metric = pickle.load(open(path + "/metrics.pkl", "rb"))
    a_traj = pickle.load(open(path + "/a_traj.pkl", "rb"))
    t_traj = pickle.load(open(path + "/t_traj.pkl", "rb"))


    past_pose = a_traj['agent-0'][0]
    for ii, pose in enumerate(a_traj['agent-0']):
        

        targ_pose = []
        for jj in range(cfg['num_targets']):
            targ_pose.append([t_traj['targ-%d'%jj][ii]])

        targ_pose = np.squeeze(np.array(targ_pose))

        num = 20
        x = np.linspace(past_pose[0], pose[0], num)
        y = np.linspace(past_pose[1], pose[1], num)
        z = np.linspace(past_pose[2], pose[2], num)
        yaw = np.linspace(past_pose[3], pose[3], num)
        pitch = np.linspace(past_pose[4], pose[4], num)
        for kk in range(num):
            update_idx = app.run_view2particles(
                np.array([x[kk],y[kk],z[kk]]), yaw[kk], pitch[kk], targ_pose, in_situ=True
            )
            update_idx = app.run_view2particles(
                np.array([x[kk],y[kk],z[kk]]), yaw[kk], pitch[kk], targ_pose, in_situ=True, save_img=-10
            )
            update_idx = app.run_view2particles(
                np.array([x[kk],y[kk],z[kk]]), yaw[kk], pitch[kk], targ_pose, in_situ=False
            )
        past_pose = pose

        render_plot(ii, pf, fig, metric, a_traj, t_traj)

    st()

if __name__ == "__main__":
    main()