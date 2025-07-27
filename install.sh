#!/bin/bash
# This script must be ran with the current_state.repos in the ${HOME} directory
# and can be ran anywhere inside your machine 

mkdir -p ~/explorer_bot/src
cd ~/explorer_bot/src
vcs import < ~/current_state.repos
cd ~/explorer_bot
source /opt/ros/jazzy/setup.bash
rosdep install -i --from-paths src --rosdistro jazzy -y
colcon build --symlink-install
