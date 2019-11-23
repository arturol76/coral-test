#!/usr/bin/python3
import argparse
import os
import requests
import datetime

import mlclient.client as mlclient

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d - [%(filename)s:%(funcName)s] %(message)s', datefmt='%d-%b-%y %H:%M:%S')
#logger = logging.getLogger(__name__) #gets THIS logger
logger = logging.getLogger() #gets ROOT logger
logger.setLevel(logging.DEBUG) #set level for THIS logger
#logger.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d - [%(filename)s:%(funcName)s] %(message)s', datefmt='%d-%b-%y %H:%M:%S')

def parse_cmdline_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-c', 
        '--config',
        default='./config/config.json',
        dest='config_json',
        help='Configuration file'
    )

    parser.add_argument(
        '-u',
        '--uvicorn',
        action='store_true', 
        default=False,
        dest='uvicorn_start',
        help='Starts uvicorn programmatically')

    parser.add_argument(
        '-f',
        '--file',
        #action='store_true', 
        default=False,
        dest='file',
        required=True,
        help='image file'
        )

    return parser.parse_args()

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    
    #parse command line arguments
    args = parse_cmdline_args()

    #init client
    mlclient = mlclient.Client("https://mlserver.arturol76.net:8001/api/v1")
    
    #get detectors
    data = mlclient.getDetectors()
    logger.debug("response={}".format(data))

    #process image from file
    data = mlclient.processFile(args.file, ["object"], False, True)
    logger.debug("response={}".format(data))

    stop_time = datetime.datetime.now()
    elapsed_time = stop_time - start_time
    logger.info('execution took {}'.format(elapsed_time))



    

