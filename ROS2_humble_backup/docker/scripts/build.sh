#!/bin/bash
# Use Bash as the shell for this script.

# Check if the current directory when running the command is correct.
if [[ $PWD = *ROS ]]; then
    # If the current directory ends with 'ROS', change directory to 'docker'.
    cd docker
elif [[ ! $PWD = *ROS/docker ]]; then
    # If the current directory is not 'ROS/docker', print an error message.
    echo -e "You must be in either 'ROS' or the 'ROS/docker' directory to run this command."
    echo -e "Você precisa estar na pasta 'ROS' ou em 'ROS/docker' para rodar esse commando."
    # Exit the script with status 1 to indicate an error.
    return 1
fi

# --network=host \           -> Use the host's network, not creating a separate network stack for the Docker container.
# -f ROS_humble.dockerfile \ -> Specify the Dockerfile to use, named 'ROS_humble.dockerfile'.
# -t ros_ws:humble \         ->  Tag the built image as 'ros_ws:humble'.
# --rm \                     -> Remove intermediate containers after a successful build.
# .                          -> Set the context of the build to the current directory.

# Build a Docker image from a Dockerfile located in the current directory.
docker build \
    --network=host \
    -f ROS_humble.dockerfile \
    -t ros_ws:humble \
    --rm \
    .