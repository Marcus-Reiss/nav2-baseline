# Here you can config commands when the container is inicialized

#!/bin/bash

set -e
source /opt/ros/humble/setup.bash

# Verifique se o setup.bash do workspace existe antes de carregá-lo
if [ -f /home/ROS/ros_ws/install/setup.bash ]; then
    source /home/ROS/ros_ws/install/setup.bash
else
    echo "WARNING: /home/ROS/ros_ws/install/setup.bash not found. Did you forget to build the workspace?"
fi

exec "$@"
