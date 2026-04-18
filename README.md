RPI CAMERA STACK 

before getting started you should make sure that you have followed the setup guide for your rpi in the [turtlebot setup guide](https://emanual.robotis.com/docs/en/platform/turtlebot3/sbc_setup/#sbc-setup)

please note that for the rpi camera setup, follow the v4l2 instructions
sudo apt-get install ros-humble-usb-cam


this repository includes the launch files for:
 - Camera Image Node
 - Aruco Detection Node
 - Camera to baselink tf static transform files




<h2>how to setup cameras:</h2>
[ON RPI]
mkdir launchfiles && cd launchfiles
git clone --branch rpi_code --single-branch https://github.com/ChinYanXu/CDE2310_g4_AY2526_turtlebot_ROS2.git
rosdep update
rosdep install --from-paths src --ignore-src -r -y
pip3 install opencv-contrib-python transforms3d
colcon build
source install/setup.bash
chmod +x one.sh


<h2>launch all files</h2>
[inside launchfiles folder]
./one.sh rpi