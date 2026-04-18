# RPI LAUNCH STACK

before getting started you should make sure that you have followed the setup guide for your rpi in the [turtlebot setup guide](https://emanual.robotis.com/docs/en/platform/turtlebot3/sbc_setup/#sbc-setup)

please note that for the rpi camera setup, follow the v4l2 instructions
sudo apt-get install ros-humble-usb-cam


this repository includes the launch files for:
 - Camera Image Node
 - Aruco Detection Node
 - Camera to baselink tf static transform files




### setup [ON RPI]
1. mkdir launchfiles && cd launchfiles
2. git clone --branch rpi_code --single-branch https://github.com/ChinYanXu/CDE2310_g4_AY2526_turtlebot_ROS2.git
3. rosdep update
4. rosdep install --from-paths src --ignore-src -r -y
5. pip3 install opencv-contrib-python transforms3d
6. colcon build
7. source install/setup.bash
###### *check that numpy version is 1.26.4, if it isnt then pip [uninstall numpy -y] and [pip install numpy==1.26.4]
8. chmod +x one.sh
###### move camera calibration files to correct destination
9. cd ~/launchfiles/camera_calibration_files
10. cp * ~/.ros/camera-info


<h2>launch all files</h2>
- [inside launchfiles folder]
- ./one.sh rpi