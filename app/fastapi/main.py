from fastapi import FastAPI, HTTPException
from starlette.staticfiles import StaticFiles
import os

from pydantic import BaseModel
from typing import List, Set, Tuple, Dict

import modules.face as FaceDetect
import modules.object as ObjectDetect
import modules.object_coral as ObjectDetectCoral
import modules.face_coral as FaceDetectCoral
import modules.rekognition as RekognitionDetect
import modules.yolo as YoloDetect
import modules.detectors as Detectors

import modules.globals as g
import modules.db as Database
import modules.utils as utils
import modules.log as log

import datetime
from enum import Enum

class DetectRequest(BaseModel):
    url: str = None
    file: str = None
    model: str
    delete: bool = None
    model_list: List[str] = None

upload_folder = "./images"
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)
    log.logging.info('creating folder {}'.format(upload_folder))
else:
    log.logging.info('folder {} already exists'.format(upload_folder))

#fast api
app = FastAPI()

#serving static files
app.mount("/images", StaticFiles(directory=upload_folder), name="static")


api_detectors = Detectors.Detectors()

#initialization of some models to save time (is it true?)
api_detectors.add(ObjectDetect.Detector())
api_detectors.add(FaceDetect.Detector())
api_detectors.add(ObjectDetectCoral.Detector())
api_detectors.add(FaceDetectCoral.Detector())
api_detectors.add(RekognitionDetect.Detector())
api_detectors.add(YoloDetect.Detector(YoloDetect.YoloModel.yolov3))
api_detectors.add(YoloDetect.Detector(YoloDetect.YoloModel.yolov3_spp))
api_detectors.add(YoloDetect.Detector(YoloDetect.YoloModel.yolov3_tiny))

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/api/v1/detectors")
async def get_detectors():
    response = {
        "detectors": list(api_detectors.get())
    } 

    return response

#http://192.168.2.96:8001/api/v1/detect
@app.post("/api/v1/detect")
async def detect(request: DetectRequest):
    start = datetime.datetime.now()
    
    args = dict()
    args['url'] = request.url
    args['file'] = request.file
    args['delete'] = request.delete
    args['gender'] ="male"

    fip,ext = utils.get_file(args, upload_folder)
    fi = fip + ext
    fo = fip + '-object' + ext

    try:
        detections = api_detectors.detect(request.model, fi, fo, args)
    #except Exception as error:
    except:
        raise HTTPException(status_code=404, detail='Invalid Model: {}'.format(request.model))

    stop = datetime.datetime.now()
    elapsed_time = stop - start
    log.logging.info('detection took {}'.format(elapsed_time))

    details = {
        "fi": fi,
        "fo": fo,
        "detection_time": elapsed_time
    }

    response = {
        "request":      request,
        "detection":    detections,
        "details":      details    
    }
    
    return response

@app.post("/detect2/{model}")
async def detect2(model: str, item: DetectRequest, type: str):
    return item