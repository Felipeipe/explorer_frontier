#!/bin/bash

INSTALL_ROS=""
VALID_FLAG=false

for arg in "$@"; do
  case "$arg" in
    --install-ros)
      INSTALL_ROS=true
      VALID_FLAG=true
      ;;
    --skip-ros)
      INSTALL_ROS=false
      VALID_FLAG=true
      ;;
    *)
      echo "[ERROR] Unknown option: $arg"
      echo "Uses: $0 [--install-ros | --skip-ros]"
      exit 1
      ;;
  esac
done

if [ "$VALID_FLAG" = false ]; then
  echo "[ERROR] You must specify a flag for running the script as ./install.sh --flag:"
  echo "  --install-ros    Installs ROS2 Jazzy and Gazebo Harmonic (Required for running the project's code)"
  echo "  --skip-ros       Skips all ROS related installation"
  exit 1
fi

ROS2 Jazzy installation
if [ "$INSTALL_ROS" = true ]; then
  echo "[INFO] Installing ROS2 Jazzy..."

  sudo apt update && sudo apt install locales
  sudo locale-gen en_US en_US.UTF-8
  sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  export LANG=en_US.UTF-8

  sudo apt install software-properties-common
  sudo add-apt-repository universe
  sudo apt update && sudo apt install curl -y

  export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
  curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb

  sudo apt update && \
    sudo apt upgrade && \
    sudo apt install ros-dev-tools && \
    sudo apt install ros-jazzy-desktop

  # GAZEBO
  sudo apt-get install ros-jazzy-ros-gz
else
  echo "[INFO] Skipping ROS and Gazebo installation..."
fi

mkdir -p ~/explorer_bot/src
cd ~/explorer_bot/src
vcs import < ~/current_state.repos
cd ~/explorer_bot

source /opt/ros/jazzy/setup.bash
rosdep install -i --from-paths src --rosdistro jazzy -y

cd ~/explorer_bot/src/gz_worlds_and_maps/world/src/hospital
unzip models_part1.zip
unzip models_part2.zip
unzip models_part3.zip
unzip models_part4.zip

colcon build --symlink-install
source ~/explorer_bot/install/setup.bash
echo "[INFO]: Done! To run this project, open two terminals,"
echo "run 'source /opt/ros/jazzy/setup.bash' and "
echo "'source ~/explorer_bot/install/setup.bash' on each one of them"
echo "and then on one terminal run 'ros2 launch explorer_bringup explorer_full.launch.py'"
echo "and the other one run 'ros2 launch explorer_frontier frontier_explorer.launch.py'"
