import os
import urllib.request
import urllib.error
import datetime
import time
import typing
import ssl
from enum import Enum

import logging
logger = logging.getLogger(__name__)

class FidType(Enum):
    alarm = "alarm"
    snapshot = "snapshot"
    bestmatch = "bestmatch"

class ZmConnector:
    def __init__(
            self,
            portal: str,
            image_path: str,
            wait: float = 0,
            basic_user: str = "",
            basic_password: str = "",
            user: str = "",
            password: str = ""
        ):
        
        self.portal = portal
        logger.debug('Setting url: {}'.format(self.portal))

        self.wait = wait

        self.basic_user = basic_user
        self.basic_password = basic_password

        self.user = user
        self.password = password

        self.image_path = image_path


        self.opener = self.create_opener(
            self.portal,
            self.basic_user,
            self.basic_password
        )
                

    def create_opener(
            self,
            portal: str,
            basic_user: str,
            basic_password: str
        ):

        #SSL CONTEXT
        ctx = ssl.create_default_context()
        logger.debug('SSL context created.')
        
        #MAIN HANDLER
        if portal.lower().startswith('https://'):
            main_handler = urllib.request.HTTPSHandler(context=ctx)
        else:
            main_handler = urllib.request.HTTPHandler()
        logger.debug('main handler created.')

        #OPENER
        if basic_user != None and basic_user != "": 
            logger.debug('Basic auth config found, associating handlers')
            password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            top_level_url = portal
            password_mgr.add_password(None, top_level_url, basic_user, basic_password)
            handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
            opener = urllib.request.build_opener(handler, main_handler)
        else:
            opener = urllib.request.build_opener(main_handler)

        logger.debug('opener created.')

        return opener
        
    def download_image(
            self,
            portal: str,
            eid: str,
            fid_type: FidType,
            user: str,
            password: str,
            image_path: str
        ):

        url = portal + '/index.php?view=image&eid=' + eid + '&fid=' + fid_type 
        durl = url

        if user != None and user != "":
            durl += '&username=' + user + '&password=*****'
            url += '&username=' + user + '&password=' + urllib.parse.quote(password,safe='')
        
        logger.debug('Trying to download {}'.format(durl))
        
        try:
            input_file = self.opener.open(url)
        
        except urllib.error.HTTPError as e:
            logger.error(e)
            raise

        filename = image_path + '/' + eid + '-' + fid_type + '.jpg'

        logger.debug('saving to file {}'.format(filename))
        with open(filename, 'wb') as output_file:
            output_file.write(input_file.read())

        return filename

    def download_files(
            self,
            eid: str,
            fid_type: FidType
        ):

        if self.wait > 0:
            logger.debug('sleeping for {} seconds before downloading'.format(self.wait))
            time.sleep(self.wait)

        if fid_type == FidType.bestmatch:
            # download both alarm and snapshot
            filename1 = self.download_image(self.portal, eid, FidType.alarm, self.user, self.password, self.image_path)
            filename2 = self.download_image(self.portal, eid, FidType.snapshot, self.user, self.password, self.image_path)

        else:
            # only download one
            filename1 = self.download_image(self.portal, eid, fid_type, self.user, self.password, self.image_path)
            filename2 = None

        return filename1, filename2