#!/bin/bash
# Specifies that the script should be executed using the Bash shell.

if [[ $PWD = *ROS ]]; then
# Checks if the current directory path ends with 'ROS'.

    # Automatically detects an NVIDIA GPU and attempts to utilize it.
    # if [ -z "$(lspci | grep NVIDIA)" ]; then
    #     USE_GPUS=""
    #     # If no NVIDIA GPU is detected, set USE_GPUS to an empty string.
    #     echo "NVIDIA's GPU Não foi detectada."
    #     # Prints message indicating no NVIDIA GPU was detected.
    # else
    #     USE_GPUS="--gpus all"
    #     # If an NVIDIA GPU is detected, enable GPU support for the Docker container.
    #     echo "NVIDIA's GPU foi detectada. Ativando '--gpus all' flag."
    #     # Prints message indicating NVIDIA GPU was detected and the GPU flag is activated.
    # fi

    # $USE_GPUS \                       -> Runs the Docker container with GPU support if available.
    # --user ros_ws \                   -> Sets the user inside the container to 'ros_ws'.
    # -e QT_X11_NO_MITSHM=1 \           -> Sets an environment variable to disable MIT-SHM to help with X11 forwarding.
    # --network=host \                  -> Uses the host's network stack inside the container.
    # --ipc=host \                      -> Uses the host's IPC namespace, allowing the container to communicate with X server.
    # --privileged \                    -> Grants extended privileges to this container.
    # --oom-kill-disable                -> Prevents the container from being killed if it runs out of memory.
    # --device /dev/video0              -> Gives the container access to the video device /dev/video0.
    # -v /dev/video0:/dev/video0        -> Mounts the host's video0 device inside the container.
    # -v /dev/dri:/dev/dri \            -> Mounts the host's Direct Rendering Manager (DRM) device to support GPU rendering.
    # -v $PWD/ros_ws:/home/ROS/ros_ws \ -> Mounts the current directory's ROS workspace into the container.
    # -e DISPLAY=$DISPLAY \             -> Passes the host's display environment variable to the container.
    # ros_ws:humble                     -> Specifies the Docker image to use.


    docker run -it  --rm \
        $USE_GPUS \
        	--name ros_humble \
            --user ros_ws \
            -e QT_X11_NO_MITSHM=1 \
            --network=host \
            --ipc=host \
            --privileged \
            --oom-kill-disable \
            --device /dev/video0 \
            -v /dev/video0:/dev/video0 \
            -v /dev/dri:/dev/dri \
            -v $PWD/ros_ws:/home/ROS/ros_ws \
            -e DISPLAY=$DISPLAY \
            --workdir /home/ROS/ros_ws \
            ros_ws:humble
                      
elif [[ ! $PWD = *ROS/docker ]]; then
# Checks if the current directory is not 'ROS/docker'.
    echo -e "You must be in 'ROS' directory to run this command."
    # Informs the user that they must be in the 'ROS' directory to run the script.
    echo -e "Você deve estar na pasta 'ROS' para rodar esse comando."
    # Informs the user in Portuguese of the same requirement.
    return 1
    # Exits the script with a status of 1, indicating an error.
fi

