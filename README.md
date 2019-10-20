# phusion-app-template
Docker template to containerize an application using [arturol76/phusion-baseimage](https://github.com/arturol76/phusion-baseimage) (i.e. phusion) as base image.

Why phusion and not ubuntu? Have a look to this:
https://blog.phusion.nl/2015/01/20/docker-and-the-pid-1-zombie-reaping-problem/

## Sample app
Have a look to the [app](./app).

## Your app
Put you code in the /app folder.

IMPORTANT: in order to start your application at container's startup, remember to edit "myapp.run" accordingly. That's the way to start an application (DO NOT add any CMD to the Dockerfile). 

## Build
To build the docker on a local docker host:

```
docker build -t YOUR_IMAGE_NAME .
```

To build it on a remote docker host:

```
docker -H YOUR_DOCKER_HOST_IP build -t YOUR_IMAGE_NAME .
```

...or just you the the provided build script [build.sh](./build.sh) as follows:

```
./build.sh YOUR_DOCKER_HOST_IP YOUR_IMAGE_NAME
```

## Run
TBD

## SSH Keys
Have a look to the [keys](./keys).
IMPORTANT: the container must be running.

## VSCODE: sample config file for the SFTP extension
Assuming that:
* you use pageant to manage the key authentication (recommended!)
* your docker host ip is 192.168.2.96

```
{
    "name": "mydocker",
    "host": "192.168.2.96",
    "protocol": "sftp",
    "port": 7122,
    "username": "root",
    "remotePath": "/",
    "agent": "pageant",
    "passphrase": true,
    "uploadOnSave": true
}
```

If the container is running, vscode will automatically upload the code you are editing to the remote container via SSH.