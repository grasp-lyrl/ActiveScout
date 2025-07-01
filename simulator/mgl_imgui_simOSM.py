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
from moderngl_window.integrations.imgui import ModernglWindowRenderer
import sys, click
from scipy.spatial.transform import Rotation as R


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
            settings.WINDOW["aspect_ratio"] = 1.0

        self.wnd = mglw.create_window_from_settings()
        self.ctx = self.wnd.ctx
        # self.window_size = (1280, 720)
        self.window_size = (320, 320)

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

        self.camera = KeyboardCamera(
            self.wnd.keys, fov=90.0, aspect_ratio=self.wnd.aspect_ratio
        )
        self.near = 0.1
        self.far = 1000.0
        self.camera_enabled = True
        self.camera.projection.update(near=self.near, far=self.far)
        self.camera.velocity = 100.0
        self.camera.mouse_sensitivity = 0.3
        # free control over camera vs not. see key_event for online switching
        self.free = False

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
        self.lin_depth_prog["near"].value = self.camera.projection.near
        self.lin_depth_prog["far"].value = self.camera.projection.far

        # use must run self.init_objects
        self.init_obj = False

        # preset colors for targets
        # rgb and opacity. (opengl, rgb int)
        self.colors = [
            ((1.0, 0.0, 0.0, 1.0), (255, 0, 0)),  # red
            ((0.0, 1.0, 0.0, 1.0), (0, 255, 0)),  # green
            ((1.0, 0.0, 1.0, 1.0), (255, 0, 255)),  # magenta
            ((0.0, 1.0, 1.0, 1.0), (0, 255, 255)),  # cyan
            ((0.0, 0.0, 1.0, 1.0), (0, 0, 255)),  # blue
            ((1.0, 1.0, 0.0, 1.0), (255, 255, 0)),  # yellow
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
        # particles and target render point size
        self.ctx.point_size = 1.0

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

        self.targ_pos = np.array([[0, 5 * time, 10], [5 * time, 0, 10]])
        ## set camera params, render scene, particles, targets
        self.in_situ = True
        self.render_func(time)
        ## use buffers to get image and compute particles in world
        self.image2world(render=True)
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
        # set background to all black
        self.ctx.clear(0.0, 0.0, 0.0)
        self.ctx.enable_only(mgl.DEPTH_TEST | mgl.CULL_FACE)

        # convert from (x,y,z) to (x,z,-y)
        self.camera.set_position(ego_pos[0], ego_pos[2], -ego_pos[1])
        # # (yaw,pitch)
        # if pitch == -90:
        #     pitch = -89.99
        if pitch > 85.0:
            pitch = 85.0
        if pitch < -85.0:
            pitch = -85.0
        self.camera.set_rotation(yaw, pitch)

        self.targ_pos = targ_pos
        # set camera params, render scene, particles, target
        self.in_situ = in_situ
        self.render_func(time)
        # renders depth in top right
        # self.depth_quad.render(self.lin_depth_prog)
        # self.insitu_quad.render(self.lin_depth_prog)
        # render imgui
        self.render_ui()

        # collect image, depth and pose data
        image = Image.frombytes("RGB", self.insitu_fbo.size, self.insitu_fbo.read())
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image = np.array(image)

        depth = np.frombuffer(self.insitu_depth.read(), dtype="f4")
        depth = (2 * self.far * self.near) / (
            self.far + self.near - (self.far - self.near) * (2 * depth - 1)
        )
        depth = depth.reshape(self.window_size[1], self.window_size[0])
        depth = np.flipud(depth)

        # pose in rotation matrix form 4x4
        # r = R.from_euler("zyx", [pitch, -yaw, 0], degrees=True).as_matrix()
        r = R.from_euler("xyz", [pitch, -yaw, 0], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = r
        T[0, 3] = ego_pos[0]
        T[1, 3] = ego_pos[2]
        T[2, 3] = ego_pos[1]

        img_data = {"image": image, "depth": depth, "pose": T, "mvp": self.depth_mvp}

        # current view to get particles in world,
        # indices of particles seen, and target in world
        part_w, part_idx, targ_dict = self.image2world(save_img=save_img, render=False)

        return part_w, part_idx, targ_dict, img_data

    # @profile
    def render_func(self, time):
        """
        set camera params, render scene, particles, targets
        """
        # world coordinates
        translation = Matrix44.from_translation((0, 0, 0))
        rotation = Matrix44.from_eulers((0, 0, 0))
        model_matrix = translation * rotation
        # P*V*M
        camera_matrix = self.camera.matrix * model_matrix
        self.depth_mvp = self.camera.projection.matrix * camera_matrix
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
            self.scene.draw(
                projection_matrix=self.camera.projection.matrix,
                camera_matrix=camera_matrix,
                time=time,
            )
            self.floor.render(self.floor_program)
            self.ctx.point_size = 1.0
            self.particles_vao.render(mgl.POINTS, self.Ngrid)
            self.render_target(pos=self.targ_pos)

        # --- PASS 1: Render shadow map
        if self.in_situ:
            # Render to insitu window
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
                # self.target_program["color"].value = self.colors[ii][0]
                self.target_program["color"].value = (1. - ii/255, 0.,0.,1.)
                try:  # for multiple targets
                    self.target_program["position"].value = (
                        pos[ii, 0],
                        pos[ii, 2],
                        -pos[ii, 1],
                    )
                except:  # for 1 target
                    self.target_program["position"].value = pos[0], pos[2], -pos[1]
                self.target_vao.render(mgl.POINTS, 1)

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
        targ_dict = {}
        for jj in range(len(self.targ_pos)):
            targ_dict["targ-%d" % jj] = {"world": np.array([]), "idx": np.array([])}
            # colors in rgb integers
            # color = self.colors[0,0][1] - jj
            color = (255-jj, 0, 0)
            v, u = np.where(
                (img[:, :, 0] == color[0])
                & (img[:, :, 1] == color[1])
                & (img[:, :, 2] == color[2])
            )

            # we use depth_mvp because we want to get the target in world coordinates
            target_w = self.pixel2world(u, v, self.depth_mvp)
            targ_idx = self.world2particleidx(target_w)

            # convert from (x,z,y,1) to (x,y,z,1) and store in dict
            if target_w.size != 0:
                target_w = target_w[:, [0, 2, 1, 3]]
            targ_dict["targ-%d" % jj]["world"] = target_w
            targ_dict["targ-%d" % jj]["idx"] = targ_idx

        # convert from (x,z,y,1) to (x,y,z,1)
        if p_w.size != 0:
            p_w = p_w[:, [0, 2, 1, 3]]

        if render:
            # debug
            # y is 1 cuz i converted above for p_w and target_w
            plt.clf()
            plt.scatter(self.grid[:, 0], self.grid[:, 2])
            plt.scatter(p_w[:, 0], p_w[:, 1])
            plt.scatter(self.grid[:, 0][update_idx], self.grid[:, 2][update_idx])
            try:
                for targ_name in targ_dict.keys():
                    target_w = targ_dict[targ_name]["world"]
                    targ_idx = targ_dict[targ_name]["idx"]
                    plt.scatter(target_w[:, 0], target_w[:, 1])
                    plt.scatter(self.grid[:, 0][targ_idx], self.grid[:, 2][targ_idx])
            except:
                pass
            plt.draw()
            plt.pause(0.0001)

        return p_w, update_idx, targ_dict

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

    def get_mvp(self, pos, yaw, pitch):
        self.camera.set_position(pos[0], pos[2], pos[1])
        if pitch == -90:
            pitch = -89.99
        self.camera.set_rotation(yaw, pitch)
        translation = Matrix44.from_translation((0, 0, 0))
        rotation = Matrix44.from_eulers((0, 0, 0))
        model_matrix = translation * rotation
        # P*V*M
        camera_matrix = self.camera.matrix * model_matrix
        mvp = self.camera.projection.matrix * camera_matrix
        return mvp

    def world2pixel(self, points, pos, yaw, pitch):
        """
        debug forward process. from world to image pixels
        given points in the world find the points that lie within the image
        """
        # ex1 = np.array([100, 1, 160, 1], dtype="f4")
        mvp = self.get_mvp(pos, yaw, pitch)
        world = np.ones((points.shape[0], 4))
        world[:, 0] = points[:, 0]
        world[:, 1] = points[:, 2]
        world[:, 2] = points[:, 1]
        # openGL is column major
        cam = (np.array(mvp).T @ world.T).T
        cam /= cam[:, 3][:, None]

        u = ((cam[:, 0] + 1.0) * (self.window_size[0] / 2)).astype("int")
        v = ((cam[:, 1] + 1.0) * (self.window_size[1] / 2)).astype("int")

        u_in = np.where(((u > 0) & (u < self.window_size[0])))[0]
        v_in = np.where(((v > 0) & (v < self.window_size[1])))[0]
        all_in = np.intersect1d(u_in, v_in)
        points_in = points[all_in]
    
        # p_w = self.pixel2world(u, v, mvp)
        return points_in

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
        imgui.begin("Hallucinate Window", True)
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

        imgui.begin("In Situ Window", True)
        imgui.image(
            self.insitu_fbo.color_attachments[0].glo,
            *self.insitu_fbo.size,
            uv0=(0, 1),
            uv1=(1, 0)
        )
        imgui.end()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.SPACE:
                self.timer.toggle_pause()
            # toggle between following policy vs mouse and key control
            if key == keys.PERIOD:
                self.free = True
            if key == keys.COMMA:
                self.free = False

        self.imgui.key_event(key, action, modifiers)

    def resize(self, width: int, height: int):
        self.imgui.resize(width, height)

    def mouse_drag_event(self, x, y, dx, dy):
        self.imgui.mouse_drag_event(x, y, dx, dy)

    def mouse_scroll_event(self, x_offset, y_offset):
        self.imgui.mouse_scroll_event(x_offset, y_offset)

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
def main(map_name):
    sys.path.append("scene_config")
    if map_name == "philly_cc":
        from philly_cc import cfg, mapDim
    if map_name == "nyc_stuy":
        from nyc_stuy import cfg, mapDim
    if map_name == "nyc_downtown":
        from nyc_downtown import cfg, mapDim

    npx = mapDim["Nx"]
    npy = mapDim["Ny"]
    x = np.linspace(mapDim["x_min"], mapDim["x_max"], npx)
    y = np.linspace(mapDim["y_min"], mapDim["y_max"], npy)
    Py, Px = np.meshgrid(y, x)
    z = np.ones(int(npx * npy))*5

    ## transpose breaks c-contiguous memory , copy fixes that issue
    particles = np.vstack((Px.flatten(), Py.flatten(), z)).T.copy(order="C")
    particles = np.array(particles, dtype="f4")

    app = mgl_imgui_simOSM(cfg, headless=False)
    app.init_objects(particles)

    # app.world2pixel(particles, np.array([100,100,50]), 90, -45)

    # needs to headless=False
    app.run()

    # (x,y,z)
    pos = np.array([10.0, 10.0, 500.0])
    for ii in range(100):
        for jj in range(10):
            update_idx = app.run_view2particles(
                pos, -90 + jj, -89.99, np.array([[0 + jj, 0, 10]]), False
            )
        for jj in range(10):
            update_idx = app.run_view2particles(
                pos, -90 + jj, -89.99, np.array([[0 + ii, 0, 10]]), True
            )

    st()


if __name__ == "__main__":
    main()
