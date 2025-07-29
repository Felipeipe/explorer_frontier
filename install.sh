#!/bin/bash
# This script must be ran with the current_state.repos in the ${HOME} directory
# and can be ran anywhere inside your machine 

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
