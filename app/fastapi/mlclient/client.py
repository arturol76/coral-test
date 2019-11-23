import os
import requests
import datetime
from typing import List

import logging
logger = logging.getLogger(__name__)

class Client:
    def __init__(
            self,
            url: str
        ):
        
        #api's base url
        #example: https://mlserver.arturol76.net:8001/api/v1
        self.url = url
        
        logger.debug('Setting url: {}'.format(self.url))

    def getDetectors(self):
        # api-endpoint 
        api_endpoint = self.url + "/detectors"
        
        # defining a params dict for the parameters to be sent to the API 
        #PARAMS = {'address':location} 
        
        # sending get request and saving the response as response object 
        #r = requests.get(url = URL, params = PARAMS) 
        r = requests.get(url = api_endpoint) 
        
        # extracting data in json format 
        data = r.json() 
        
        return data

    def processFile(
            self,
            filepath: str,
            models: List[str],
            input_delete: bool = False,
            bbox_save: bool = True
        ):
        
        image = open(filepath, 'rb') 

        return self.processImage(image, models, input_delete, bbox_save)

    def processImage(
            self,
            image_cv,
            models: List[str],
            input_delete: bool = False,
            bbox_save: bool = True
        ):
        
        # api-endpoint 
        api_endpoint = self.url + "/detect"
        
        # defining a params dict for the parameters to be sent to the API 
        params = {
            'input_delete': input_delete,
            'bbox_save': bbox_save,
            'model': models
        }

        files = {'file': image_cv}
        
        # sending get request and saving the response as response object 
        r = requests.post(url = api_endpoint, params = params, files=files) 
        
        # extracting data in json format 
        data = r.json() 
        
        return data