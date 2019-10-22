from fastapi import FastAPI
from pydantic import BaseModel

import modules.face as FaceDetect
import modules.object as ObjectDetect
import modules.object_coral as ObjectDetectCoral
import modules.globals as g
import modules.db as Database
import modules.utils as utils

import modules.log as log

import datetime

from enum import Enum

upload_folder = "."

app = FastAPI()

class ModelName(str, Enum):
    yolo = "object"
    coral = "coral"
    face = "face"

class DetectRequest(BaseModel):
    url: str = None
    file: str = None
    model: ModelName
    delete: bool = None
    optional: str = None

coral_m = ObjectDetectCoral.ObjectCoral()
object_m = ObjectDetect.Object()

@app.get("/")
async def root():
    return {"message": "Hello World"}

#http://192.168.2.96:8000/api/v1/detect
@app.post("/api/v1/detect")
async def detect(item: DetectRequest):
    start = datetime.datetime.now()
    
    args = dict()
    args['url'] = item.url
    args['file'] = item.file
    args['delete'] = item.delete

    if item.model == ModelName.face:
        m = FaceDetect.Face()
    elif item.model == ModelName.coral:
        m = coral_m
    elif item.model == ModelName.yolo:
        m = object_m
    else:
        #abort(400, msg='Invalid Model:{}'.format(args['type']))
        return

    fip,ext = utils.get_file(args, upload_folder)
    detections = m.detect(fip, ext, args)
    stop = datetime.datetime.now()
    elapsed_time = stop - start
    log.logging.info('detection took {}'.format(elapsed_time))
    return detections

@app.post("/detect2/{model}")
async def detect2(model: str, item: DetectRequest, type: str):
    print(model, " ", "type")
     
    return item