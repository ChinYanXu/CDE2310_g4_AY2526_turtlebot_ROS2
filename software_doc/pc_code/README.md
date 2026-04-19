# Remote PC setup instructions

## Basic setup

1. Install ROS2 and Turtlebot3 packages by following the instructions [here](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/#pc-setup) 

2. Install Navigation2 Docking server.
``` bash
sudo apt install ros-humble-opennav-docking
```

3. Create a new directory to clone our cde2310_g4_ay2526 package.
``` bash 
$ mkdir -p ~/colcon_ws/src 
```

4. Create a ros2 package and git clone cde2310_g4_ay2526 code into the newly created package.
```bash
$ cd ~/colcon_ws/src
$ git clone https://github.com/ChinYanXu/CDE2310_g4_AY2526_turtlebot_ROS2.git ./cde2310_g4_ay2526 -b pc_code
```

5. Replace navigation_launch.py in your Navigation2 stack to include docking server node.
```bash
$ sudo mv ~/colcon_ws/src/cde2310_g4_ay2526/launch/navigation_launch.py /opt/ros/humble/share/nav2_bringup/launch/
```

6. You can now build the package on your laptop. 
```bash
$ cd ~/colcon_ws
$ colcon build
```

7. Reconfigure the planner server's `lattice_filepath` in burger.yaml. Find the following line and edit it to the correct path:
```
      lattice_filepath: "/home/tanyunqi/burger_primitive.json"
```
> You should also modify your robot footprint under the ``footprint`` field under ``global_costmap`` using the right hand rule based on your CAD to prevent any inconsistencies.

8. Copy the navigation parameters file to your turtlebot3_navigation2 package
```bash
$ cp ~/colcon_ws/src/cde2310_g4_ay2526/nav2_params/burger.yaml ~/turtlebot3_ws/src/turtlebot3/turtlebot3_navigation2/param/humble
```

9. Run frontier explorer or coordinator using the following commands.
```bash
$ source ~/colcon_ws/install/setup.bash
$ ros2 run cde2310_g4_ay2526 frontier_explorer
$ ros2 run cde2310_g4_ay2526 coordinator
```

10. For mission execution and visualisation, launch mission.launch.py or run the following commands.
```bash
$ source ~/turtlebot3_ws/install/setup.bash
$ source ~/colcon_ws/install/setup.bash
$ ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=False 
$ ros2 launch nav2_bringup navigation_launch.py \
  map:=$(ros2 pkg prefix turtlebot3_navigation2)/share/turtlebot3_navigation2/map/map.yaml \
  params_file:=$(ros2 pkg prefix turtlebot3_navigation2)/share/turtlebot3_navigation2/param/humble/burger.yaml \
  use_sim_time:=False
$ ros2 launch nav2_bringup rviz_launch.py use_sim_time:=False
$ ros2 run cde2310_g4_ay2526 frontier_explorer
```

## Camera visualisation setup
1. Install ros2_aruco package and opencv dependencies.
``` bash
$ pip install opencv-contrib-python
$ cd ~/colcon_ws/src
$ git clone https://github.com/JMU-ROBOTICS-VIVA/ros2_aruco.git
```

2. Build and source package.
``` bash
$ cd ~/colcon_ws
$ colcon build --packages-select ros2_aruco
$ source install/setup.bash
```

3. Launch GUI_OVERLAY.py
``` bash
$ cd ~/colcon_ws/src/cde2310_g4_ay2526
$ python3 GUI_OVERLAY.py --camera <cam_left/cam_right>
```

## Custom simulation setup
1. Install Turtlebot3 Simulation Package by following the instructions [here](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/).

2. Move simulation files into respective folders in your turtlebot3_gazebo package.
```bash
$ cd ~/colcon_ws/src/cde2310_g4_ay2526/simulation_files
$ mv worlds/custom_maze_CDE2310.sdf ~/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/worlds
$ mv models/custom_maze ~/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models
$ mv launch/custom_maze.launch.py ~/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch
```

3. Rebuild your turtlebot3_gazebo package.
```bash
$ cd ~/turtlebot3_ws
$ colcon build --packages-select turtlebot3_gazebo
$ source install/setup.bash
```

4. You can now launch the custom maze with the following command.
```bash
$ ros2 launch turtlebot3_gazebo custom_maze.launch.py
```
