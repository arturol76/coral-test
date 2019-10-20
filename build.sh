show_help()
{
    echo ------------------------------------------------------
    echo Builds the image "docker_image" using the dockerfile 
    echo on the target host "docker_ip".
    echo
    echo USAGE:
    echo ./build.sh docker_ip docker_image
    echo
    echo EXAMPLE:
    echo ./build.sh 192.168.2.96 arturol76/myapp
    echo ------------------------------------------------------
    echo
}

num_of_params=2
docker_host=$1
docker_image=$2

#checks number of parameters
if [ "$#" -ne $num_of_params ]; then
    echo "Illegal number of parameters."
    echo
    show_help
    exit 1
fi

echo

echo building image $docker_image...
docker -H $docker_host build -t $docker_image .

read -p "Do you want to push image to docker repository? " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo
    echo pushing to $docker_image...
    docker -H $docker_host push $docker_image
else
    echo
fi

exit 0