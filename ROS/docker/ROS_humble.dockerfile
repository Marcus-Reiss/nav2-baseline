# Use an official ROS Humble full desktop image as the base
FROM osrf/ros:humble-desktop-full

# Avoid prompts from APT during build by specifying non-interactive as the frontend
ARG DEBIAN_FRONTEND=noninteractive
# Set the timezone environment variable
ENV TZ=America/Sao_Paulo

# Define user-related arguments to create a non-root user inside the container
ARG USERNAME=ros_ws
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create a new group and user with the specified UID and GID, create a config directory, and set ownership
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir /home/$USERNAME/.config && chown $USER_UID:$USER_GID /home/$USERNAME/.config

# Update the package list, install sudo, configure sudoers for the new user without password prompts, and clean up APT lists
RUN apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Update package list and install git
RUN apt-get update \
    && apt-get install -y git-all

# Update package list and install various packages for development with ROS, including visualization tools, development libraries, and Python packages
RUN apt-get update \
    && apt-get install -y ros-humble-rviz2

RUN apt-get update && apt-get install -y \
    ros-humble-gazebo-ros-pkgs

RUN apt-get update \
    && apt-get install -y wget git build-essential\
    && apt-get install -y liburdfdom-dev liboctomap-dev libassimp-dev\
    && apt-get install -y python3-opencv\
    && apt-get install -y python3 python3-pip \
    && apt-get install -y python3-colcon-common-extensions\
    && apt-get install -y python3-colcon-argcomplete\
    && apt-get install -y python3-tk\
    && apt-get install -y ros-humble-turtlebot3*\
    && apt-get install -y ros-humble-navigation2\
    && apt-get install -y ros-humble-nav2-bringup\
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy 
# Copy custom entrypoint script and .bashrc configuration from the host to the container
COPY config/entrypoint.sh /entrypoint.sh
COPY config/bashrc /home/${USERNAME}/.bashrc

# Set the custom script to be the container's entrypoint, this script is executed when the container starts
ENTRYPOINT [ "/bin/bash", "/entrypoint.sh" ]

# Default command when the container is run, if no other commands are specified
CMD [ "bash" ]