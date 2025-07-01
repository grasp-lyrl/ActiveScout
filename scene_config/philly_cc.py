"""
Configuration file for the philly_cc scene.
"""
data_dir = "data/osm/philly_cc/"
cfg = {
    "map_name": "philly_cc",
    "name": "Philly Center City",
    "num_agents": 1,
    "num_targets": 4,
    "save": False,
    "data_dir": data_dir,
    "coord_origin": {"lat": 39.951749999999976, "lon": -75.16705},
    "obj_file": data_dir + "philly_cc_blend.obj",
    "graph_file": data_dir + "graph.pkl",
    "osm_pbf_file": data_dir + "philly_cc.osm.pbf",
    "agent": {
        "x0": [0.0, 0.0, 100.0, 0.0, 0.0],
        "hallucinate_n_samples": 15,
        "local_sample": False,
        "pos_sample": [-200, 200],
        "z_sample": [50, 150.0],
        "yaw_sample": [-180, 180],
        "pitch_sample": [0, -89.99],
        "survey_z": 150,
        "survey_pitch":-45,
    },
    "method": "MI",
    "target_mode": "active",
    "target_altitude": 5.0,
    "target_size": 8.0,
    "warmup_train_steps":4000,
    "inbtwn_train_steps":4000,
}

mapDim = {
    "x_min": -800.0,
    "x_max": 800.0,
    "y_min": -500.0,
    "y_max": 500.0,
    "z_min": 1.0,
    "z_max": 300.0,
    "Nx": 100,
    "Ny": 80,
}
