"""
Configuration file for the nyc_stuy scene.
"""
data_dir = "data/osm/nyc_stuy/"
cfg = {
    "map_name": "nyc_stuy",
    "name": "NYC StuyTown",
    "num_agents": 1,
    "num_targets": 4,
    "save": False,
    "data_dir": data_dir,
    "coord_origin": {"lat": 40.73274999999998, "lon": -73.97710000000001},
    "obj_file": data_dir + "nyc_stuy_blend.obj",
    "graph_file": data_dir + "graph.pkl",
    "osm_pbf_file": data_dir + "nyc_stuy.osm.pbf",
    "agent": {
        "x0": [0.0, 0.0, 5.0, 0.0, 0.0],
        "hallucinate_n_samples": 15,
        "local_sample": False,
        "pos_sample": [-200, 200],
        "z_sample": [25.0, 100.0],
        "yaw_sample": [-180, 180],
        "pitch_sample": [0, -89.99],
        "survey_z": 100,
        "survey_pitch":-45,
    },
    "method": "MI",
    "target_mode": "active",
    "target_altitude": 1.0,
    "target_size": 8.0,
    "warmup_train_steps":4000,
    "inbtwn_train_steps":4000,
}

mapDim = {
    "x_min": -550.0,
    "x_max": 400.0,
    "y_min": -550.0,
    "y_max": 550.0,
    "z_min": 1.0,
    "z_max": 300.0,
    "Nx": 80,
    "Ny": 80,
}
