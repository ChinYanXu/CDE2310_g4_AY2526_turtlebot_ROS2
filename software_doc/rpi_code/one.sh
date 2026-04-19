#!/usr/bin/env bash
set -eo pipefail

MODE="${1:-}"

# Extra args (only used for laptop mode)
CAMERA_ROOT="${2:-}"
NAV_ROOT="${3:-}"

ROS_SETUP="/opt/ros/humble/setup.bash"

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

open_term() {
  local title="$1"
  local cmd="$2"

  if have_cmd gnome-terminal; then
    gnome-terminal --title="$title" -- bash -ic "$cmd; exec bash"
  elif have_cmd xterm; then
    xterm -T "$title" -e bash -ic "$cmd; exec bash" &
  else
    echo "No supported terminal emulator found." >&2
    exit 1
  fi
}

launch_laptop() {
  echo "Launching laptop stack..."

  # Validate inputs
  if [[ -z "$CAMERA_ROOT" || -z "$NAV_ROOT" ]]; then
    echo "ERROR: laptop mode requires CAMERA_ROOT and NAV_ROOT"
    echo "Usage: $0 laptop <camera_root> <nav_root>"
    exit 1
  fi

  LAPTOP_WS_SETUP="$NAV_ROOT/install/setup.bash"

  open_term "rslam" "
    source \"$ROS_SETUP\"
    source ~/.bashrc
    [ -f \"$LAPTOP_WS_SETUP\" ] && source \"$LAPTOP_WS_SETUP\"
    rslam
  "

  open_term "nav2" "
    source \"$ROS_SETUP\"
    source ~/.bashrc
    [ -f \"$LAPTOP_WS_SETUP\" ] && source \"$LAPTOP_WS_SETUP\"
    ros2 launch nav2_bringup navigation_launch.py \
      map:=\$(ros2 pkg prefix turtlebot3_navigation2)/share/turtlebot3_navigation2/map/map.yaml \
      params_file:=\$(ros2 pkg prefix turtlebot3_navigation2)/share/turtlebot3_navigation2/param/humble/burger.yaml \
      use_sim_time:=false
  "

  open_term "coordinator" "
    source \"$ROS_SETUP\"
    source ~/.bashrc
    [ -f \"$LAPTOP_WS_SETUP\" ] && source \"$LAPTOP_WS_SETUP\"
    ros2 run cde2310_g4_ay2526 coordinator
  "
}

launch_rpi() {
  echo "Launching RPi stack..."
  RPI_WS_SETUP="$HOME/turtlebot3_ws/install/setup.bash"

  echo "Starting rosbu..."
  (
    source "$ROS_SETUP"
    source ~/.bashrc
    ros2 launch turtlebot3_bringup robot.launch.py
  ) &

  sleep 2

  echo "Starting camfile..."
  (
    source "$ROS_SETUP"
    [ -f "$RPI_WS_SETUP" ] && source "$RPI_WS_SETUP"
    cd ~/aruco
    ros2 launch camult.py
  ) &
  wait
}

usage() {
  cat <<EOF
Usage:
  $0 laptop <camera_root> <nav_root>
  $0 rpi

Examples:
  $0 laptop ~/test ~/turtlebot3_ws
  $0 rpi
EOF
}

case "$MODE" in
  laptop)
    launch_laptop
    ;;
  rpi)
    launch_rpi
    ;;
  *)
    usage
    exit 1
    ;;
esac
