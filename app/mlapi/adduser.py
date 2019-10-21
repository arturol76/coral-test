from tinydb import TinyDB, Query, where
from passlib.hash import sha256_crypt
import modules.globals as g
import getpass

import argparse

import modules.db as Database

db = Database.Database()

ap = argparse.ArgumentParser()
    
ap.add_argument(
    '-u', 
    '--user',
    default=None,
    required=True,
    dest='user',
    help='username'
)

ap.add_argument(
    '-p', 
    '--password',
    default=None,
    required=True,
    dest='password',
    help='password'
)

args = vars(ap.parse_args())

if args['user'] != None and args['password'] != None:
    db.add_user(args['user'],args['password'])
    print ('User: {} created'.format(args['user']))