import pyrosm
import numpy as np
from shapely.geometry import Polygon, Point
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import pickle, sys


def init_env(map_name: str):
    """
    add your env config file name here
    """
    sys.path.append("scene_config")
    if map_name == "philly_cc":
        from philly_cc import cfg, mapDim
    if map_name == "nyc_stuy":
        from nyc_stuy import cfg, mapDim
    if map_name == "nyc_downtown":
        from nyc_downtown import cfg, mapDim
        
    return cfg, mapDim

def building_footprints(osm_file: str, coord_origin: dict):
    """
    the following conversions follow osm2world metric map conversion
    https://github.com/tordanik/OSM2World/blob/master/src/main/java/org/osm2world/core/map_data/creation/MetricMapProjection.java
    In OSM2World coordinates, 1 unit of distance along the X, Y or Z axis corresponds to approximately 1 meter.

    osm_file = "data/osm/philly_cc/philly_cc.osm.pbf"
    coord_origin = {"lat": 39.951749999999976, "lon": -75.16705}
    be sure to include these coord_origin in the confiration files
    """

    def lontoX(lon):
        return (np.array(lon) + 180.0) / 360.0

    def lattoY(lat):
        sinlat = np.sin(np.radians(lat))
        return np.log((1.0 + sinlat) / (1.0 - sinlat)) / 4.0 / np.pi + 0.5

    def scaleFactor(lat):
        earthcircum = 40075016.686
        return earthcircum * np.cos(np.radians(lat))

    def originXY(lat, lon, scaleFactor):
        x = lontoX(lon) * scaleFactor
        y = lattoY(lat) * scaleFactor
        return x, y

    def toXY(lat, lon, scaleFactor, originX, originY):
        x = lontoX(lon) * scaleFactor - originX
        y = lattoY(lat) * scaleFactor - originY
        x = np.round(x * 1000.0) / 1000.0
        y = np.round(y * 1000.0) / 1000.0
        return x, y

    def gdftoXY(geometry):
        # usually osm has polygon or multipolygon, if multipolygon, we need loop through each polygon
        try:
            lat = geometry.exterior.coords.xy[1]
            lon = geometry.exterior.coords.xy[0]
            scale = scaleFactor(coord_origin["lat"])
            ox, oy = originXY(coord_origin["lat"], coord_origin["lon"], scale)
            x, y = toXY(lat, lon, scale, ox, oy)
        except:
            x = []
            y = []
            for ii in range(len(geometry.geoms)):
                lat = geometry.geoms[ii].exterior.coords.xy[1]
                lon = geometry.geoms[ii].exterior.coords.xy[0]
                scale = scaleFactor(coord_origin["lat"])
                ox, oy = originXY(coord_origin["lat"], coord_origin["lon"], scale)
                xi, yi = toXY(lat, lon, scale, ox, oy)
                x = np.hstack((x, xi))
                y = np.hstack((y, yi))

        return Polygon(np.vstack((x, y)).T)

    # Create a Pyrosm object
    osm = pyrosm.OSM(osm_file)

    # Get building footprints as GeoDataFrame in latlon
    buildings = osm.get_buildings()
    buildingsXY = buildings.geometry.apply(gdftoXY)
    return buildingsXY


def create_graph(cfg, pf, render=False, save=True):
    """
    this create graph is made when you need to consider the buildings
    the particle filter is given when pruned particles so a regular grid cannot be used
    it is slow than the regular grid and should be saved
    """
    x_diff = pf.mapDim["x_max"] - pf.mapDim["x_min"]
    y_diff = pf.mapDim["y_max"] - pf.mapDim["y_min"]
    edge_len_x = x_diff / (pf.Nx - 1.0)
    edge_len_y = y_diff / (pf.Ny - 1.0)
    max_edge = np.ceil(np.sqrt(edge_len_x**2 + edge_len_y**2))

    G = nx.Graph()
    for ii, pos in enumerate(pf.p_targ):
        G.add_node(ii, pos=(pos[0], pos[1]))

    # finds combinations of all pairs of particles
    for (ii, p1), (jj, p2) in combinations(enumerate(pf.p_targ), 2):
        dist = np.linalg.norm(p1 - p2)
        if dist < max_edge:
            G.add_edge(ii, jj)

    if save:
        print("saving graph to " + cfg["graph_file"])
        pickle.dump(G, open(cfg["graph_file"], "wb"))

    if render:
        print("rendering graph... close to continue")
        pos = nx.get_node_attributes(G, "pos")
        nx.draw(G, pos, node_size=1, node_color="skyblue", edge_color="gray", width=1.0)
        plt.show()

    return G
