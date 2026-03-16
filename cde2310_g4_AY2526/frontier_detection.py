#!/usr/bin/env python3

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import math
import numpy as np


OCC_THRESHOLD = 10
MIN_FRONTIER_SIZE = 5


class OccupancyGrid2d:
    class CostValues(IntEnum):
        FreeSpace = 0
        LethalObstacle = 100
        NoInformation = -1

    def __init__(self, grid_msg):
        self.map = grid_msg

    def get_cost(self, mx: int, my: int) -> int:
        return self.map.data[self._get_index(mx, my)]

    def get_size_x(self) -> int:
        return self.map.info.width

    def get_size_y(self) -> int:
        return self.map.info.height

    def map_to_world(self, mx: int, my: int):
        wx = self.map.info.origin.position.x + (mx + 0.5) * self.map.info.resolution
        wy = self.map.info.origin.position.y + (my + 0.5) * self.map.info.resolution
        return wx, wy

    def world_to_map(self, wx: float, wy: float):
        if wx < self.map.info.origin.position.x or wy < self.map.info.origin.position.y:
            raise ValueError("World coordinates out of bounds")

        mx = int((wx - self.map.info.origin.position.x) / self.map.info.resolution)
        my = int((wy - self.map.info.origin.position.y) / self.map.info.resolution)

        if mx < 0 or my < 0 or mx >= self.map.info.width or my >= self.map.info.height:
            raise ValueError("World coordinates out of bounds")

        return mx, my

    def _get_index(self, mx: int, my: int) -> int:
        return my * self.map.info.width + mx


class PointFlags(IntEnum):
    MapOpen = 1
    MapClosed = 2
    FrontierOpen = 4
    FrontierClosed = 8


@dataclass
class FrontierPoint:
    map_x: int
    map_y: int
    classification: int = 0


@dataclass
class FrontierRegion:
    x: float
    y: float
    size: int


class FrontierCache:
    def __init__(self):
        self.cache = {}

    def get_point(self, x: int, y: int) -> FrontierPoint:
        key = (x, y)
        if key not in self.cache:
            self.cache[key] = FrontierPoint(x, y)
        return self.cache[key]

    def clear(self):
        self.cache.clear()


def centroid(points):
    arr = np.array(points)
    return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))


def get_neighbors(point: FrontierPoint, costmap: OccupancyGrid2d, cache: FrontierCache):
    neighbors = []
    for x in range(point.map_x - 1, point.map_x + 2):
        for y in range(point.map_y - 1, point.map_y + 2):
            if 0 <= x < costmap.get_size_x() and 0 <= y < costmap.get_size_y():
                if x == point.map_x and y == point.map_y:
                    continue
                neighbors.append(cache.get_point(x, y))
    return neighbors


def is_frontier_point(point: FrontierPoint, costmap: OccupancyGrid2d, cache: FrontierCache) -> bool:
    # Frontier point = unknown cell adjacent to at least one free cell
    # and not directly touching high-cost obstacle cells.
    if costmap.get_cost(point.map_x, point.map_y) != OccupancyGrid2d.CostValues.NoInformation:
        return False

    has_free_neighbor = False
    for n in get_neighbors(point, costmap, cache):
        cost = costmap.get_cost(n.map_x, n.map_y)

        if cost > OCC_THRESHOLD:
            return False

        if cost == OccupancyGrid2d.CostValues.FreeSpace:
            has_free_neighbor = True

    return has_free_neighbor


def find_nearest_free(mx: int, my: int, costmap: OccupancyGrid2d):
    cache = FrontierCache()
    q = deque([cache.get_point(mx, my)])
    visited = set()

    while q:
        loc = q.popleft()
        key = (loc.map_x, loc.map_y)
        if key in visited:
            continue
        visited.add(key)

        if costmap.get_cost(loc.map_x, loc.map_y) == OccupancyGrid2d.CostValues.FreeSpace:
            return loc.map_x, loc.map_y

        for n in get_neighbors(loc, costmap, cache):
            if (n.map_x, n.map_y) not in visited:
                q.append(n)

    return mx, my


def detect_frontiers(costmap: OccupancyGrid2d, robot_pose, min_frontier_size: int = MIN_FRONTIER_SIZE):
    """
    robot_pose should be geometry_msgs/Pose (in map frame)
    returns a list[FrontierRegion]
    """
    cache = FrontierCache()
    cache.clear()

    mx, my = costmap.world_to_map(robot_pose.position.x, robot_pose.position.y)
    free_mx, free_my = find_nearest_free(mx, my, costmap)

    start = cache.get_point(free_mx, free_my)
    start.classification |= PointFlags.MapOpen
    map_queue = deque([start])

    frontiers = []

    while map_queue:
        p = map_queue.popleft()

        if p.classification & PointFlags.MapClosed:
            continue

        if is_frontier_point(p, costmap, cache):
            p.classification |= PointFlags.FrontierOpen
            frontier_queue = deque([p])
            new_frontier = []

            while frontier_queue:
                q = frontier_queue.popleft()

                if q.classification & (PointFlags.MapClosed | PointFlags.FrontierClosed):
                    continue

                if is_frontier_point(q, costmap, cache):
                    new_frontier.append(q)

                    for w in get_neighbors(q, costmap, cache):
                        if not (w.classification & (
                            PointFlags.FrontierOpen | PointFlags.FrontierClosed | PointFlags.MapClosed
                        )):
                            w.classification |= PointFlags.FrontierOpen
                            frontier_queue.append(w)

                q.classification |= PointFlags.FrontierClosed

            world_points = []
            for fp in new_frontier:
                fp.classification |= PointFlags.MapClosed
                world_points.append(costmap.map_to_world(fp.map_x, fp.map_y))

            if len(new_frontier) >= min_frontier_size:
                cx, cy = centroid(world_points)
                frontiers.append(FrontierRegion(x=cx, y=cy, size=len(new_frontier)))

        for v in get_neighbors(p, costmap, cache):
            if not (v.classification & (PointFlags.MapOpen | PointFlags.MapClosed)):
                neighbors = get_neighbors(v, costmap, cache)
                if any(costmap.get_cost(n.map_x, n.map_y) == OccupancyGrid2d.CostValues.FreeSpace for n in neighbors):
                    v.classification |= PointFlags.MapOpen
                    map_queue.append(v)

        p.classification |= PointFlags.MapClosed

    return frontiers


def choose_frontier(frontiers, robot_pose, strategy: str = "nearest"):
    """
    strategy: 'nearest' or 'largest'
    returns FrontierRegion or None
    """
    if not frontiers:
        return None

    if strategy == "largest":
        return max(frontiers, key=lambda f: f.size)

    # default: nearest
    return min(
        frontiers,
        key=lambda f: math.hypot(f.x - robot_pose.position.x, f.y - robot_pose.position.y)
    )