"""

LIDAR to 2D grid map example

author: Erno Horvath, Csaba Hajdu based on Atsushi Sakai's scripts

"""

import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np


def file_read(f):
    """
    Reading LIDAR laser beams (angles and corresponding distance data)
    """
    with open(f) as data:
        measures = [line.split(",") for line in data]
    angles = []
    distances = []
    for measure in measures:
        angles.append(float(measure[0]))
        distances.append(float(measure[1]))
    angles = np.array(angles)
    distances = np.array(distances)
    return angles, distances


def bresenham(start, end):
    """
    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end (original from roguebasin.com)
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points

def Bresenham3D(start, end):
    x1 = start[0]
    y1 = start[1]
    z1 = start[2]
    x2 = end[0]
    y2 = end[1]
    z2 = end[2]
    ListOfPoints = []
    ListOfPoints.append((x1, y1, z1))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (z2 > z1):
        zs = 1
    else:
        zs = -1
 
    # Driving axis is X-axis"
    if (dx >= dy and dx >= dz):        
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))
 
    # Driving axis is Y-axis"
    elif (dy >= dx and dy >= dz):       
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                x1 += xs
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))
 
    # Driving axis is Z-axis"
    else:        
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while (z1 != z2):
            z1 += zs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            ListOfPoints.append((x1, y1, z1))
    return ListOfPoints


def atan_zero_to_twopi(y, x):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0
    return angle


def init_flood_fill(center_point, obstacle_points, xy_points, min_coord, xy_resolution):
    """
    center_point: center point
    obstacle_points: detected obstacles points (x,y)
    xy_points: (x,y) point pairs
    """
    center_x, center_y = center_point
    prev_ix, prev_iy = center_x - 1, center_y
    ox, oy = obstacle_points
    xw, yw = xy_points
    min_x, min_y = min_coord
    occupancy_map = (np.ones((xw, yw))) * 0.5
    for x, y in zip(ox, oy):
        # x coordinate of the the occupied area
        ix = int(round((x - min_x) / xy_resolution))
        # y coordinate of the the occupied area
        iy = int(round((y - min_y) / xy_resolution))
        free_area = bresenham((prev_ix, prev_iy), (ix, iy))
        for fa in free_area:
            occupancy_map[fa[0]][fa[1]] = 0  # free area 0.0
        prev_ix = ix
        prev_iy = iy
    return occupancy_map


def flood_fill(center_point, occupancy_map):
    """
    center_point: starting point (x,y) of fill
    occupancy_map: occupancy map generated from Bresenham ray-tracing
    """
    # Fill empty areas with queue method
    sx, sy = occupancy_map.shape
    fringe = deque()
    fringe.appendleft(center_point)
    while fringe:
        n = fringe.pop()
        nx, ny = n
        # West
        if nx > 0:
            if occupancy_map[nx - 1, ny] == 0.5:
                occupancy_map[nx - 1, ny] = 0.0
                fringe.appendleft((nx - 1, ny))
        # East
        if nx < sx - 1:
            if occupancy_map[nx + 1, ny] == 0.5:
                occupancy_map[nx + 1, ny] = 0.0
                fringe.appendleft((nx + 1, ny))
        # North
        if ny > 0:
            if occupancy_map[nx, ny - 1] == 0.5:
                occupancy_map[nx, ny - 1] = 0.0
                fringe.appendleft((nx, ny - 1))
        # South
        if ny < sy - 1:
            if occupancy_map[nx, ny + 1] == 0.5:
                occupancy_map[nx, ny + 1] = 0.0
                fringe.appendleft((nx, ny + 1))


def generate_ray_casting_grid_map(
    ox, oy, x_w, y_w, loc_x, loc_y, aabb, xy_resolution, breshen=True
):
    """
    The breshen boolean tells if it's computed with bresenham ray casting
    (True) or with flood fill (False)
    """
    min_x = aabb[2]
    min_y = aabb[0]
    max_x = aabb[5]
    max_y = aabb[3]
    # default 0.5 -- [[0.5 for i in range(y_w)] for i in range(x_w)]
    occupancy_map = np.ones((x_w, y_w)) / 2
    center_x = loc_x  # int(round(-min_x / xy_resolution))  # center x coordinate of the grid map
    center_y = loc_y  # int(round(-min_y / xy_resolution))  # center y coordinate of the grid map
    # occupancy grid computed with bresenham ray casting
    if breshen:
        for x, y in zip(ox, oy):
            # x coordinate of the the occupied area
            ix = int(round((x - min_x) / xy_resolution))
            # y coordinate of the the occupied area
            iy = int(round((y - min_y) / xy_resolution))
            laser_beams = bresenham(
                (center_x, center_y), (ix, iy)
            )  # line form the lidar to the occupied point
            for laser_beam in laser_beams:
                if (
                    laser_beam[0] >= occupancy_map.shape[0]
                    or laser_beam[1] >= occupancy_map.shape[1]
                ):
                    continue
                occupancy_map[laser_beam[0]][laser_beam[1]] = 0.0  # free area 0.0

            if ix < occupancy_map.shape[0] and iy < occupancy_map.shape[1]:
                occupancy_map[ix][iy] = 1.0  # occupied area 1.0
            if ix + 1 < occupancy_map.shape[0] and iy < occupancy_map.shape[1]:
                occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
            if ix < occupancy_map.shape[0] and iy + 1 < occupancy_map.shape[1]:
                occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
            if ix + 1 < occupancy_map.shape[0] and iy + 1 < occupancy_map.shape[1]:
                occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
    # occupancy grid computed with with flood fill
    else:
        occupancy_map = init_flood_fill(
            (center_x, center_y), (ox, oy), (x_w, y_w), (min_x, min_y), xy_resolution
        )
        flood_fill((center_x, center_y), occupancy_map)
        occupancy_map = np.array(occupancy_map, dtype=float)
        for x, y in zip(ox, oy):
            ix = int(round((x - min_x) / xy_resolution))
            iy = int(round((y - min_y) / xy_resolution))
            occupancy_map[ix][iy] = 1.0  # occupied area 1.0
            occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
            occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
            occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
    return occupancy_map, min_x, max_x, min_y, max_y, xy_resolution
