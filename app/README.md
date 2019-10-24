## Sample APP
A dockerized application to play with Google CORAL EdgeTPU.
This repo has been generated using [arturol76/phusion-app-template](https://github.com/arturol76/phusion-app-template) as a template.

WORK IN PROGRESS!! DO NOT USE.

### APP
Simple API for object detection using [FastAPI](https://github.com/tiangolo/fastapi) and uvicorn.
Test it connecting a browser to the following url:

http://192.168.2.96:8001/

(assuming that the docker's ip is 192.168.2.96)

### How to run the APP at container's startup
Phusion uses runit.
To start (and monitor) your application via runit, edit "myapp.run" to your needs.

As an example, this is what "myapp.run" does in order to start this sample application:

```
#!/bin/sh
cd /app/fastapi && uvicorn --reload --host 0.0.0.0 --port YOUR_DESIRED_PORT main:app
```

