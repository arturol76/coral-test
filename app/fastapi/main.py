#!/usr/bin/python3
import argparse
import uvicorn
import os

import logging
logger = logging.getLogger(__name__)

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

    return parser.parse_args()

def start_uvicorn():
    ssl_keyfile_env = None
    ssl_certfile_env = None
    
    if 'SSL_KEYFILE' in os.environ and 'SSL_CERTFILE' in os.environ:
        if os.path.exists(os.environ['SSL_KEYFILE']) and os.path.exists(os.environ['SSL_CERTFILE']):
            ssl_keyfile_env = os.environ['SSL_KEYFILE']
            ssl_certfile_env = os.environ['SSL_CERTFILE']
        else:
            logger.error("KEY/CERT files not not found. Check files existance and/or ENV vars")
    else:
        logger.debug("KEY/CERT ENV vars not not found.")
    
    uvicorn.run(
        "api:app", 
        host="0.0.0.0",
        port=5001, 
        log_level="info",
        ssl_keyfile=ssl_keyfile_env,
        ssl_certfile=ssl_certfile_env,
        reload=True
    )

if __name__ == "__main__":
    #parse command line arguments
    parser = parse_cmdline_args()

    #starting uvicorn
    if parser.uvicorn_start == True:
        start_uvicorn()
