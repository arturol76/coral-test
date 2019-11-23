#!/usr/bin/python3
from fastapi import FastAPI, HTTPException, Form, File, UploadFile, Query
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

import modules.utils as utils

import datetime
from enum import Enum

import connectors.zoneminder as zmconnector

import urllib.error

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d - [%(filename)s:%(funcName)s] %(message)s', datefmt='%d-%b-%y %H:%M:%S')
#logger = logging.getLogger(__name__) #gets THIS logger
logger = logging.getLogger() #gets ROOT logger
logger.setLevel(logging.DEBUG) #set level for THIS logger
#logger.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d - [%(filename)s:%(funcName)s] %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger.debug("api.py debug")

class ApiRequest(BaseModel):
    filename: str = ""
    model: List[str] = [""]
    image_save: bool = True
    bbox_save: bool = False
    
class ApiResponse(BaseModel):
    request:        ApiRequest = None
    response:       Detectors.RunBatchResponse = None

def upload_folder_init(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)
        logger.info('creating folder {}'.format(folder))
    else:
        logger.info('folder {} already exists'.format(folder))
    return folder

def init_api(folder):
    #fast api
    app = FastAPI(
        title="mlserver",
        description="my Machine Learning Server",
        version="0.1.0",
    )

    #serving static files
    app.mount("/images", StaticFiles(directory=folder), name="static")
    return app

#initialization of some models to save time (is it true?)
def load_detectors():
    api_detectors = Detectors.Detectors()
    api_detectors.add(ObjectDetect.Detector())
    api_detectors.add(FaceDetect.Detector())
    api_detectors.add(ObjectDetectCoral.Detector())
    api_detectors.add(FaceDetectCoral.Detector())
    api_detectors.add(RekognitionDetect.Detector())
    api_detectors.add(YoloDetect.Detector(YoloDetect.YoloModel.yolov3))
    api_detectors.add(YoloDetect.Detector(YoloDetect.YoloModel.yolov3_spp))
    api_detectors.add(YoloDetect.Detector(YoloDetect.YoloModel.yolov3_tiny))
    api_detectors.init_all()
    return api_detectors


api_detectors = load_detectors()
upload_folder = upload_folder_init("./images")
app = init_api(upload_folder)
zm = zmconnector.ZmConnector(
    "https://zoneminder.arturol76.net/zm",
    upload_folder
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/api/v1/detectors")
def get_detectors():
    response = {
        "detectors": list(api_detectors.get())
    } 

    return response

#refer to https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#query-parameter-list-multiple-values
@app.post("/api/v1/detect/file")
async def api_detect_file(
        *,
        file: UploadFile = File(...), 
        model: List[str] = Query(...), 
        image_save: bool = True,
        bbox_save: bool = False
    ):
    
    request = ApiRequest()
    request.bbox_save = bbox_save
    request.image_save = image_save

    api_response = ApiResponse()
    
    try:
        fip = utils.get_file_from_form(file, upload_folder)

        batch_response = api_detectors.run(
                fip, 
                model,
                image_save,
                bbox_save
            )

        request.filename = file.filename
        request.model = model

        api_response.request = request
        api_response.response = batch_response
        
        return api_response
    
    except Exception as error:
        logger.error('exception: {}, {}'.format(type(error), error))
        error_obj = {
            "exception_type":   '{}'.format(type(error)),
            "message":          '{}'.format(error)
        }
        raise HTTPException(status_code=404, detail=error_obj)

@app.post("/api/v1/detect/zm")
async def api_detect_zm(
        *,
        model: List[str] = Query(...), 
        image_save: bool = True,
        bbox_save: bool = False,
        eid: str,
        fid: str
    ):

    request = ApiRequest()
    request.bbox_save = bbox_save
    request.image_save = image_save

    api_response = ApiResponse()

    try:
        fip1, fip2 = zm.download_files(eid, fid)

        batch_response = api_detectors.run(
                fip1,            #filename with path of input image
                model,
                image_save,
                bbox_save
            )


        #TEST
        merged = api_detectors.merge(batch_response)
        api_detectors.nms(merged, 0.5, 0.8)
        
        request.filename = fip1
        request.model = model

        api_response.request = request
        api_response.response = batch_response
        
        return api_response
    
    except urllib.error.HTTPError as error:
        logger.error('exception: {}'.format(error))
        raise HTTPException(status_code=error.code, detail='{}'.format(error.reason))

    except Exception as error:
        logger.error('exception: {}, {}'.format(type(error), error))
        error_obj = {
            "exception_type":   '{}'.format(type(error)),
            "message":          '{}'.format(error)
        }
        raise HTTPException(status_code=404, detail=error_obj)
    
    