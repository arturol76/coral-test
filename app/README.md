## Sample APP

### APP
Sample APP using [FastAPI](https://github.com/tiangolo/fastapi) and uvicorn to expose a simple API.
Test it connecting a browser to the following url:

http://192.168.2.96:8000/


(assuming that the docker's ip is 192.168.2.96)

### How to run the APP at container's startup
Phusion uses runit.
To start (and monitor) your application via runit, edit "myapp.run" to your needs.

As an example, this is what "myapp.run" does in order to start this sample application:

```
#!/bin/sh
cd /app && uvicorn --reload --host 0.0.0.0 main:app
```