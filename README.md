# coral-test
A docker to play with Google CORAL EdgeTPU.
This repo has been generated using [arturol76/phusion-app-template](https://github.com/arturol76/phusion-app-template) as a template.

The APP is using [pliablepixels/mlapi](https://github.com/pliablepixels/mlapi) as a starting point.

WORK IN PROGRESS!! DO NOT USE.

## Versions: 

## Build

## Run
mlapi:
```
cd /app/mlapi && python3 ./api.py
```

fastapi:
```
cd /app/fastapi && uvicorn --reload --host 0.0.0.0 --port 5001 main:app
```

## Change Log