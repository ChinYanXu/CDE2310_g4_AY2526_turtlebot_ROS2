# Remote PC setup instructions

1. Install ROS2 and Turtlebot3 packages by following the instructions [here](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/#pc-setup) 

2. Create a new directory to clone the auto_nav package.
``` bash 
$ mkdir -p ~/colcon_ws/src 
```

3. Create a ros2 package and git clone auto_nav code into the newly created package.
```bash
$ cd ~/colcon_ws/src
$ git clone https://github.com/ChinYanXu/CDE2310_g4_AY2526_turtlebot_ROS2.git . -b pc_code
```

4. You can now build the package on your laptop. 
```bash
$ cd ~/colcon_ws
$ colcon build
```

5. Copy the navigation parameters file to your turtlebot3_navigation2 package
```bash
$ cp ~/colcon_ws/src/CDE2310_g4_AY2526_turtlebot_ROS2/nav2_params/burger.yaml ~/turtlebot3_ws/src/turtlebot3/turtlebot3_navigation2/param/humble
```

6. Reconfigure the planner server's lattice_filepath in burger.yaml. Find the following line and edit it to the correct path:
```
      lattice_filepath: "/home/tanyunqi/burger_primitive.json"
```

7. Run frontier explorer or coordinator using the following commands.
```bash
$ source ~/colcon_ws/install/setup.bash
$ ros2 run cde2310_g4_ay2526 frontier_explorer
$ ros2 run cde2310_g4_ay2526 coordinator
```
