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

import modules.globals as g
import modules.db as Database
import modules.utils as utils

import datetime
from enum import Enum

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d - [%(filename)s:%(funcName)s] %(message)s', datefmt='%d-%b-%y %H:%M:%S')
#logger = logging.getLogger(__name__) #gets THIS logger
logger = logging.getLogger() #gets ROOT logger
logger.setLevel(logging.DEBUG) #set level for THIS logger
#logger.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d - [%(filename)s:%(funcName)s] %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger.debug("api.py debug")

class API_DetectRequest(BaseModel):
    url: str = None
    models_list: List[str] = None
    delete: bool = None

def upload_folder_init(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)
        logger.info('creating folder {}'.format(folder))
    else:
        logger.info('folder {} already exists'.format(folder))

    return folder

upload_folder = upload_folder_init("./images")

#fast api
app = FastAPI(
    title="mlserver",
    description="my Machine Learning Server",
    version="0.1.0",
)

#serving static files
app.mount("/images", StaticFiles(directory=upload_folder), name="static")

#initialization of some models to save time (is it true?)
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

def detect_do(
        fip: str,
        ext: str,
        models_list: List[str],
        delete: bool
    ):
    
    executed_succesfully = []
    failed = []

    start_total = datetime.datetime.now()
    args = dict()
    
    args['delete'] = delete
    args['gender'] ="male"

    fi = fip + ext
    
    response_list = []
    
    for model_item in models_list: 
        try:
            start = datetime.datetime.now()
            fo = fip + '-' + model_item + ext
            detections = api_detectors.detect(model_item, fi, fo, args)
            stop = datetime.datetime.now()
            elapsed_time = stop - start
            logger.info('detection took {}'.format(elapsed_time))

            details = {
                "fi": fi,
                "fo": fo,
                "detection_time": elapsed_time
            }

            response_item = {
                "model": model_item,
                "details": details,
                "error": None,
                "model_response": detections
            }
            response_list.append(response_item)

            executed_succesfully.append(model_item)
   
        except Exception as error:
            logger.error('exception: {}'.format(error))
            response_item = {
                "model": model_item,
                "details": None,
                "error": "invalid model: {}".format(model_item),
                "model_response": None
            }
            response_list.append(response_item)
            failed.append(model_item)
            #raise HTTPException(status_code=404, detail='Invalid Model: {}'.format(model_item))

    stop_total = datetime.datetime.now()
    elapsed_time_total = stop_total - start_total
    logger.info('TOTAL detection took {}'.format(elapsed_time_total))

    return executed_succesfully, failed, elapsed_time_total, response_list

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/api/v1/detectors")
def get_detectors():
    response = {
        "detectors": list(api_detectors.get())
    } 

    return response

#http://192.168.2.96:8001/api/v1/detect
@app.post("/api/v1/detect/url")
async def api_detect_url(request: API_DetectRequest):

    fip, ext = utils.get_file_from_url(request.url, upload_folder)

    executed_succesfully, failed, elapsed_time_total, response_list = detect_do(
            fip, 
            ext, 
            request.models_list,
            request.delete
        )

    response = {
        "request":      request,
        "executed_ok":  executed_succesfully,
        "failed":       failed,       
        "total_time":   elapsed_time_total,
        "reponse_list": response_list
    }
    
    return response

@app.post("/api/v1/detect/form")
async def api_detect_form(
        *, 
        file: UploadFile = File(...), 
        models_list: List[str] = Form(...), 
        delete: bool = Form(...)
    ):
    
    logger.info('filename={},content_type={}'.format(file.filename, file.content_type))

    fip, ext = utils.get_file_from_form(file, upload_folder)

    executed_succesfully, failed, elapsed_time_total, response_list = detect_do(
            fip, 
            ext, 
            models_list,
            delete
        )

    request = {
        "filename": file.filename,
        "models_list": models_list,
        "delete": delete
    }

    response = {
        "request":      request,
        "executed_ok":  executed_succesfully,
        "failed":       failed,       
        "total_time":   elapsed_time_total,
        "reponse_list": response_list
    }

    return response

#refer to https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#query-parameter-list-multiple-values
@app.post("/api/v1/detect/file")
async def api_detect_file(
        *,
        file: UploadFile = File(...), 
        model: List[str] = Query(...), 
        delete: bool
    ):
    
    logger.info('filename={},content_type={}'.format(file.filename, file.content_type))

    fip, ext = utils.get_file_from_form(file, upload_folder)

    executed_succesfully, failed, elapsed_time_total, response_list = detect_do(
            fip, 
            ext, 
            model,
            delete
        )

    request = {
        "filename": file.filename,
        "model": model,
        "delete": delete
    }

    response = {
        "request":      request,
        "executed_ok":  executed_succesfully,
        "failed":       failed,       
        "total_time":   elapsed_time_total,
        "reponse_list": response_list
    }

    return response