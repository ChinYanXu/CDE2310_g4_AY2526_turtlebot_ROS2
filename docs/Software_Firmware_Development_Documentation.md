---
title: Software & Firmware Development Documentation
---

# Software & Firmware Development Documentation

## Navigation

- [Back to main README.md](../README.md)

## Software Interfaces

![Software_Interface_Diagram](software_interface_diagram.png)

> NOTE: The bottom camera is renamed the left camera and the top camera is renamed the right camera to maintain consistancy with the code.

The software architecture is organised around the `coordinator.py` node, which serves as the main decision-making module for exploration, docking, and payload delivery. The SLAM module receives LiDAR data and wheel-encoder odometry to construct an occupancy grid map, which is then used by the frontier detection module to identify candidate exploration goals. The selected frontier goal is passed to the coordinator, which decides whether the robot should continue exploring or transition to a docking task.

The left camera image stream is processed by an ArUco detection node, which publishes `/cam_left/aruco_markers` to the coordinator. The coordinator uses these detections to identify the stationary and midpoint tags and to determine when to stop exploration and begin docking. Once a target is detected, the coordinator interacts with Nav2 through the `NavigateToPose`, `dock_robot`, and `undock_robot` action interfaces to move the robot and align it with the receptacle.

The coordinator also communicates with the payload delivery module through a command-and-status interface. It publishes `/station_cmd` to trigger payload release for either Station A or Station B, while `payload.py` publishes `/mission_complete` back to the coordinator after the dispensing sequence is complete. This allows the coordinator to resume exploration only after the payload task has finished and the robot has undocked.

For the moving-station mission, the right camera image stream is processed by a separate ArUco detection node that publishes `/cam_right/aruco_markers` to `payload.py`. After the coordinator arms the Station B sequence, the payload node waits for the appropriate visual trigger before actuating the servos. This ensures that payload release occurs only when the moving target is in the correct position.

## SLAM: Cartographer
The base capability enabling autonomous navigation for our robot is rooted in Simultaneous Localization and Mapping (SLAM) using LiDAR sensor data. For SLAM, we had two main options: [Cartographer](https://google-cartographer-ros.readthedocs.io/en/latest/) and [SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox). Since we are using Navigation2 stack for path planning and motion control, we first went with SLAM Toolbox which had more documented integration with Nav2 stack. However when testing our frontier detection algorithm using SLAM Toolbox and Nav2, the robot tended to oscillate. There were also internal parameters within SLAM Toolbox, such as tf frequency update, which were not optimised to match our Nav2 parameters. In the first half of our build phase, we decided to switch to Cartographer which showed more consistent results. 

Cartographer's output occupancy grid uses a range of values which represent the probability of occupancy, instead of discrete values used by SLAM Toolbox and Nav2 stack. These values range from 0 (free) to 100 (lethal obstacle). Our frontier detection node in `frontier_detection.py` is written to categorise these values into free (0-40), unknown (40-80) and occupied (80-100) based on custom tunable thresholds. Depending on specific navigation and exploration needs, these values can be adjusted to allow less or more conservative exploration.
```
    class CostValues(IntEnum):
        FreeSpace = 0
        LethalObstacle = 100
        NoInformation = -1

        # Cartographer
        FreeThreshold = 40
        OccupiedThreshold = 80
```

## Frontier Detection: `frontier_detection.py`
Adapted from [SeanReg/nav2_wavefront_frontier_exploration](https://github.com/SeanReg/nav2_wavefront_frontier_exploration), our frontier detection module enables the robot to autonomously explore unknown areas of the maze using the occupancy grid map. A cell is considered a frontier if it is a free cell and at least one of its neighbours is unknown. The frontier detection module groups neighbouring frontier cells into frontier regions and chooses the centroid of a region as the navigation destination for the region. `MIN_FRONTIER_SIZE` is configured to 3 such that a frontier must have at least 3 frontier cells to be considered, filtering out small negligible frontiers. This can be tuned to make navigation more or less sensitive. 

Initially, the nearest frontier region was chosen as the destination, but this caused the robot to only explore a small area in the maze and at slow speeds. Instead, we adapted our frontier selection strategy to choose the centroid of the largest frontier region as the navigation goal. The largest frontier is where there are the most unknown cells and tends to be the border of the mapped region. With this strategy, the frontiers chosen had greater information gain for mapping.

If a suitable frontier region is not found, the modules selects a fallback destination by choosing a free-space point near the explored-unexplored boundary that is sufficiently far from obstacles and recently visited fallback points. This allows the robot to continue roaming even when the map is sufficiently explored.

## Coordinator: `coordinator.py`
The coordinator orchestrates the overall mission logic, including maze exploration, docking, payload delivery triggering, and undocking. Actual navigation is handled by the Nav2 `NavigateToPose` action, while docking and undocking are handled using the Nav2 docking actions. When the robot is in the `EXPLORE` state, it uses frontier detection to search the maze. If the stationary or moving station ArUco tag is detected, the coordinator transitions to the corresponding docking state by cancelling the current exploration goal. Once docking succeeds, the coordinator triggers payload delivery and waits for the payload node to report completion before commanding the robot to undock and resume exploration.

![Coordinator State Diagram](Coordinator_state_diagram.png)

| State | Function |
|---|---|
| EXPLORE | The robot explores the maze using frontier detection. |
| GO_TO_STATIONARY | The robot has detected the ArUco tag of the stationary station and is attempting to dock to it. |
| GO_TO_MIDPOINT | The robot has detected the ArUco tag associated with the moving-station mission and is attempting to dock. |
| WAIT_A_COMPLETE | The robot has triggered payload delivery for stationary station and is waiting for the payload sequence to finish. |
| WAIT_B_COMPLETE | The robot has triggered payload delivery for moving station and is waiting for the payload sequence to finish. |

## Nav2 Planner & Controller
For our Navigation2 stack parameters, we referred extensively to [AutomaticAddison's tuning guide](https://automaticaddison.com/ros-2-navigation-tuning-guide-nav2) as well as [Nav2's tuning guide](https://docs.nav2.org/tuning/index.html). The Nav2 planner server is in charge of the algorithms for path planning, given a `NavigateToPose` pose. The Nav2 controller server supplies the robot with control instructions and abstracts away movement control in order to complete tasks like following the path given by the planner server, or avoiding obstacles. 

Our chosen planner server is **Smac Lattice Planner** which was recommended for our Turtlebot3 type, a non-circular, differential robot. This planner is able to incorporate the exact footprint of our robot, allowing us to make full use of the small width of our robot to navigate through tight spaces. 

Our chosen controller server is **Regulated Pure Pursuit** which is the best controller for exact path following while incorporating effective obstacle avoidance using heuristics. Other controllers like MPPI and DWB are optimised for dynamic obstacle avoidance, a capability that our robot does not require in this mission. This controller also allows full optimisation of movement according to the planner server.

## Docking
For docking, we initially tried manual docking using `NavigateToPose` goals. However, due to obstacle avoidance measures from the Nav2 controller, the robot was unable to dock close enough to the receptable using this method. Hence, we decided to integrate Nav2's Docking server which abstracted away precise movement control. 

## Servo Code
The `payload.py` program controls the two servos used for payload release. It subscribes to `/station_cmd` for mission commands and publishes `/mission_complete` when dispensing is done. For the stationary station, the payload sequence is executed immediately after receiving `START_A`. For the moving station, the node is first armed by `START_B`, and payload release is then triggered step-by-step based on ArUco detections from the right camera. This allows the moving station payload to be dispensed only when the target is in the correct position.

