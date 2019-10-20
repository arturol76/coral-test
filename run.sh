#!/bin/bash
show_help()
{
    echo ------------------------------------------------------------------
    echo Creates and runs on the target "docker_ip" host a container with name 
    echo "container_name" using image "docker_image".
    echo Container will expose SSH service on port "ssh_docker_port".
    echo
    echo NOTE: it will create and mount a named volume for the persistent storage of .ssh files.
    echo
    echo USAGE:
    echo ./run.sh docker_ip docker_image container_name ssh_docker_port
    echo
    echo EXAMPLE:
    echo ./run.sh 192.168.2.96 arturol76/myapp myapp 7122
    echo ------------------------------------------------------------------
    echo
}

num_of_params=4
docker_host=$1
docker_image=$2
container_name=$3
docker_port=$4

#checks number of parameters
if [ "$#" -ne $num_of_params ]; then
    echo "Illegal number of parameters."
    echo
    show_help
    exit 1
fi

echo creating volumes...
volume_ssh=${container_name}_ssh
docker -H $docker_host volume create ${volume_ssh}

volume_ssh=${container_name}_app
docker -H $docker_host volume create ${volume_app}

#if docker is already running, stops it
if [ "$(docker -H $docker_host ps -a | grep $container_name)" ]; then
    echo container $docker_host already exists

    echo stopping it...
    docker -H $docker_host stop $container_name

    echo removing it...
    docker -H $docker_host rm $container_name

    echo pulling image...
    docker -H $docker_host pull $docker_image
fi

# create your container
echo creating the container...
docker -H $docker_host create \
    --restart always \
    -p ${docker_port}:22/tcp \
    -p 8000:8000/tcp \
    -e TZ="Europe/Rome" \
    -v ${volume_ssh}:"/root/.ssh":rw \
    --privileged \
    --name $container_name \
    $docker_image 
#    /bin/sh -c "uvicorn main:app --reload --host 0.0.0.0"

#-v ${container_name}_app:"/app" \

#run
echo starting the container...
docker -H $docker_host start \
    $container_name

exit 0