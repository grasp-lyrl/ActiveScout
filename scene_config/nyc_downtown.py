"""
Configuration file for the nyc_downtown scene.
"""
data_dir = "data/osm/nyc_downtown/"
cfg = {
    "name": "NYC Downtown",
    "num_agents": 1,
    "num_targets": 4,
    "save": False,
    "data_dir": data_dir,
    "coord_origin": {"lat": 40.70465000000004, "lon": -74.01164999999999},
    "obj_file": data_dir + "nyc_downtown_blend.obj",
    "graph_file": data_dir + "graph.pkl",
    "osm_pbf_file": data_dir + "nyc_downtown.osm.pbf",
    "agent": {
        "x0": [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        "hallucinate_n_samples": 15,
        "pos_sample": [-100, 100],
        "z_sample": [100.0, 400.0],
        "yaw_sample": [-180, 180],
        "pitch_sample": [-25, -89.99],
    },
    "target_mode": "active",
    "target_altitude": 1.0,
    "target_size": 8.0,
}

mapDim = {
    "x_min": -600.0,
    "x_max": 850.0,
    "y_min": -600.0,
    "y_max": 900.0,
    "z_min": 1.0,
    "z_max": 300.0,
    "Nx": 60,
    "Ny": 60,
}
