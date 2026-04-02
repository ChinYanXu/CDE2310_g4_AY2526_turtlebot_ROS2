#!/usr/bin/env python3

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import math
import numpy as np
import heapq
import math

OCC_THRESHOLD = 10
MIN_FRONTIER_SIZE = 3


class OccupancyGrid2d:
    class CostValues(IntEnum):
        FreeSpace = 0
        LethalObstacle = 100
        NoInformation = -1

        # Cartographer
        OccupiedThreshold = 70

    # Cartographer
    def is_free(self, mx: int, my: int) -> bool:
        cost = self.get_cost(mx, my)
        return cost != self.CostValues.NoInformation and cost < self.CostValues.OccupiedThreshold

    def is_unknown(self, mx: int, my: int) -> bool:
        return self.get_cost(mx, my) == self.CostValues.NoInformation

    def is_obstacle(self, mx: int, my: int) -> bool:
        cost = self.get_cost(mx, my)
        return cost != self.CostValues.NoInformation and cost >= self.CostValues.OccupiedThreshold

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


import heapq
import math


def get_4_neighbors(mx: int, my: int, costmap: OccupancyGrid2d):
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = mx + dx, my + dy
        if 0 <= nx < costmap.get_size_x() and 0 <= ny < costmap.get_size_y():
            yield nx, ny


def is_traversable_free_cell(mx: int, my: int, costmap: OccupancyGrid2d) -> bool:
    return costmap.get_cost(mx, my) == OccupancyGrid2d.CostValues.FreeSpace


def is_free_cell_adjacent_to_frontier(mx: int, my: int, frontier: FrontierRegion, costmap: OccupancyGrid2d,
                                      search_radius_cells: int = 3) -> bool:
    """
    Returns True if this free cell is near the frontier centroid and touches unknown space.
    Used to find a reachable viewpoint / border cell for a frontier.
    """
    if not is_traversable_free_cell(mx, my, costmap):
        return False

    wx, wy = costmap.map_to_world(mx, my)
    fx, fy = frontier.x, frontier.y

    # Keep candidate cells near the frontier centroid
    resolution = costmap.map.info.resolution
    if math.hypot(wx - fx, wy - fy) > search_radius_cells * resolution:
        return False

    # Cell should border unknown space
    for nx in range(mx - 1, mx + 2):
        for ny in range(my - 1, my + 2):
            if nx < 0 or ny < 0 or nx >= costmap.get_size_x() or ny >= costmap.get_size_y():
                continue
            if nx == mx and ny == my:
                continue
            if costmap.get_cost(nx, ny) == OccupancyGrid2d.CostValues.NoInformation:
                return True

    return False


def get_frontier_goal_cell(frontier: FrontierRegion, costmap: OccupancyGrid2d,
                           search_radius_cells: int = 4):
    """
    Find a reachable free-space cell near the frontier centroid.
    Prefer the closest free border cell to the centroid.
    """
    try:
        cx, cy = costmap.world_to_map(frontier.x, frontier.y)
    except ValueError:
        return None

    candidates = []

    for x in range(cx - search_radius_cells, cx + search_radius_cells + 1):
        for y in range(cy - search_radius_cells, cy + search_radius_cells + 1):
            if x < 0 or y < 0 or x >= costmap.get_size_x() or y >= costmap.get_size_y():
                continue

            if is_free_cell_adjacent_to_frontier(x, y, frontier, costmap, search_radius_cells):
                wx, wy = costmap.map_to_world(x, y)
                dist_to_centroid = math.hypot(wx - frontier.x, wy - frontier.y)
                candidates.append((dist_to_centroid, x, y))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    _, best_mx, best_my = candidates[0]
    return best_mx, best_my


def dijkstra_distance(costmap: OccupancyGrid2d, start_mx: int, start_my: int,
                      goal_mx: int, goal_my: int):
    """
    Compute shortest path distance over free cells using 4-connectivity.
    Returns path length in meters, or math.inf if unreachable.
    """
    if not is_traversable_free_cell(start_mx, start_my, costmap):
        return math.inf
    if not is_traversable_free_cell(goal_mx, goal_my, costmap):
        return math.inf

    pq = [(0.0, start_mx, start_my)]
    dist = {(start_mx, start_my): 0.0}
    visited = set()
    step_cost = costmap.map.info.resolution

    while pq:
        cur_cost, mx, my = heapq.heappop(pq)

        if (mx, my) in visited:
            continue
        visited.add((mx, my))

        if mx == goal_mx and my == goal_my:
            return cur_cost

        for nx, ny in get_4_neighbors(mx, my, costmap):
            if not is_traversable_free_cell(nx, ny, costmap):
                continue

            new_cost = cur_cost + step_cost
            if new_cost < dist.get((nx, ny), math.inf):
                dist[(nx, ny)] = new_cost
                heapq.heappush(pq, (new_cost, nx, ny))

    return math.inf

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
    """
    Frontier cell:
    - current cell is unknown
    - at least one neighbor is free
    """
    if not costmap.is_free(point.map_x, point.map_y):
        return False
    for n in get_neighbors(point, costmap, cache):
        if costmap.is_unknown(n.map_x, n.map_y):
            return True
    return False


def detect_frontiers(costmap: OccupancyGrid2d, robot_pose, min_frontier_size: int = MIN_FRONTIER_SIZE):
    """
    Full-map frontier detection:
    1. collect all frontier cells
    2. cluster connected frontier cells
    3. use cluster centroid as representative frontier goal
    """
    cache = FrontierCache()
    frontier_cells = []

    for y in range(costmap.get_size_y()):
        for x in range(costmap.get_size_x()):
            p = cache.get_point(x, y)
            if is_frontier_point(p, costmap, cache):
                frontier_cells.append(p)

    visited = set()
    frontiers = []

    for cell in frontier_cells:
        key = (cell.map_x, cell.map_y)
        if key in visited:
            continue

        q = deque([cell])
        cluster = []
        visited.add(key)

        while q:
            cur = q.popleft()
            cluster.append(cur)

            for n in get_neighbors(cur, costmap, cache):
                nkey = (n.map_x, n.map_y)
                if nkey in visited:
                    continue

                if is_frontier_point(n, costmap, cache):
                    visited.add(nkey)
                    q.append(n)

        if len(cluster) >= min_frontier_size:
            world_points = [costmap.map_to_world(c.map_x, c.map_y) for c in cluster]
            cx, cy = centroid(world_points)
            frontiers.append(FrontierRegion(x=cx, y=cy, size=len(cluster)))

    return frontiers


# def choose_frontier(frontiers, robot_pose, strategy: str = "nearest"):
#     """
#     strategy: 'nearest' or 'largest'
#     """
#     if not frontiers:
#         return None

#     if strategy == "largest":
#         return max(frontiers, key=lambda f: f.size)

#     return min(
#         frontiers,
#         key=lambda f: math.hypot(
#             f.x - robot_pose.position.x,
#             f.y - robot_pose.position.y
#         )
#     )


def choose_frontier(frontiers, robot_pose, strategy="nearest", costmap=None):
    if not frontiers:
        return None, None

    if strategy == "largest":
        f = max(frontiers, key=lambda f: f.size)
        return f, None

    if strategy == "nearest":
        f = min(
            frontiers,
            key=lambda f: math.hypot(
                f.x - robot_pose.position.x,
                f.y - robot_pose.position.y
            )
        )
        return f, None

    if strategy in ("nearest_dijkstra", "farthest_dijkstra"):
        if costmap is None:
            raise ValueError("costmap is required for Dijkstra-based strategies")

        robot_mx, robot_my = costmap.world_to_map(
            robot_pose.position.x,
            robot_pose.position.y
        )

        reachable = []

        for frontier in frontiers:
            goal_cell = get_frontier_goal_cell(frontier, costmap)
            if goal_cell is None:
                continue

            goal_mx, goal_my = goal_cell
            path_cost = dijkstra_distance(costmap, robot_mx, robot_my, goal_mx, goal_my)

            if math.isfinite(path_cost):
                reachable.append((path_cost, frontier, (goal_mx, goal_my)))

        if not reachable:
            f = min(
                frontiers,
                key=lambda f: math.hypot(
                    f.x - robot_pose.position.x,
                    f.y - robot_pose.position.y
                )
            )
            return f, None

        if strategy == "nearest_dijkstra":
            _, frontier, goal_cell = min(reachable, key=lambda item: item[0])
            return frontier, goal_cell

        _, frontier, goal_cell = max(reachable, key=lambda item: item[0])
        return frontier, goal_cell

    raise ValueError(f"Unknown frontier strategy: {strategy}")


def is_unknown_adjacent_free_cell(mx: int, my: int, costmap: OccupancyGrid2d) -> bool:
    """
    Free-space cell with at least one unknown neighbor.
    This is useful as a candidate 'border' viewpoint.
    """
    if not costmap.is_free(mx, my):
        return False
    for nx in range(mx - 1, mx + 2):
        for ny in range(my - 1, my + 2):
            if nx < 0 or ny < 0 or nx >= costmap.get_size_x() or ny >= costmap.get_size_y():
                continue
            if nx == mx and ny == my:
                continue
            if costmap.is_unknown(nx, ny):
                return True
    return False



def obstacle_clearance_ok(mx: int, my: int, costmap: OccupancyGrid2d, min_clearance_cells: int) -> bool:
    for nx in range(mx - min_clearance_cells, mx + min_clearance_cells + 1):
        for ny in range(my - min_clearance_cells, my + min_clearance_cells + 1):
            if nx < 0 or ny < 0 or nx >= costmap.get_size_x() or ny >= costmap.get_size_y():
                continue
            if costmap.is_obstacle(nx, ny):
                return False
    return True


def near_recent_point(wx: float, wy: float, recent_points, revisit_radius: float) -> bool:
    for px, py in recent_points:
        if math.hypot(wx - px, wy - py) < revisit_radius:
            return True
    return False


def choose_fallback_viewpoint(
    costmap: OccupancyGrid2d,
    robot_pose,
    recent_points,
    min_clearance_cells: int = 2,
    revisit_radius: float = 0.8,
):
    """
    Fallback strategy:
    choose a reachable free-space point near the explored/unknown boundary.

    Scoring:
    - prefer points close to the robot
    - prefer points adjacent to unknown
    - reject points too near obstacles
    - reject recently visited fallback viewpoints
    """
    candidate_points = []

    for y in range(costmap.get_size_y()):
        for x in range(costmap.get_size_x()):
            if not is_unknown_adjacent_free_cell(x, y, costmap):
                continue

            if not obstacle_clearance_ok(x, y, costmap, min_clearance_cells):
                continue

            wx, wy = costmap.map_to_world(x, y)

            if near_recent_point(wx, wy, recent_points, revisit_radius):
                continue

            dist = math.hypot(
                wx - robot_pose.position.x,
                wy - robot_pose.position.y
            )

            candidate_points.append((dist, wx, wy))

    if not candidate_points:
        return None

    candidate_points.sort(key=lambda item: item[0])
    _, best_x, best_y = candidate_points[0]
    return best_x, best_y