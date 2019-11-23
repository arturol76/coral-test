#!/usr/bin/python3
import connectors.zoneminder as zmconnector
import argparse
import os
import datetime

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d - [%(filename)s:%(funcName)s] %(message)s', datefmt='%d-%b-%y %H:%M:%S')
#logger = logging.getLogger(__name__) #gets THIS logger
logger = logging.getLogger() #gets ROOT logger
logger.setLevel(logging.DEBUG) #set level for THIS logger

def parse_cmdline_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-e', 
        '--eid',
        default=None,
        dest='eid',
        help='event id',
        required=True
    )

    parser.add_argument(
        '-f', 
        '--fid',
        default=None,
        dest='fid_type',
        help='fid type (alarm, snapshot, bestmatch)',
        required=True
    )

    parser.add_argument(
        '-p', 
        '--image_path',
        default="./",
        dest='image_path',
        help='path where to store images',
        required=False
    )

    return parser.parse_args()

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    
    #parse command line arguments
    parser = parse_cmdline_args()

    zm = zmconnector.ZmConnector(
        "https://zoneminder.arturol76.net/zm",
        parser.image_path
    )

    zm.download_files(parser.eid, parser.fid_type)

    stop_time = datetime.datetime.now()
    elapsed_time = stop_time - start_time
    logger.debug('execution took {}'.format(elapsed_time))