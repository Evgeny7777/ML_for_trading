import errno
from functools import wraps
import os
from pathlib import Path
import logging                

import boto3

from py.config import Config as conf
from py.logger import l

def disable_logging(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        logging.disable(logging.DEBUG)
        result = method(self, *method_args, **method_kwargs)
        logging.disable(logging.NOTSET)
        return result
    return _impl

class S3_client:
    
    @disable_logging
    def __init__(self):
        self.session = boto3.session.Session()
        self.bucket = conf.AWS_BUCKET_NAME
        self.client = self.session.client(
            service_name='s3',
            aws_access_key_id=conf.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=conf.AWS_SECRET_ACCESS_KEY
        )
        boto3.set_stream_logger('boto3.resources', logging.INFO)
    
    @disable_logging
    def upload_binary(self, binary_data, key):
        response = self.client.put_object(
        Bucket=self.bucket,
        Body=binary_data, 
        Key=str(key)
        )
        status_code = response['ResponseMetadata']['HTTPStatusCode']
        l.debug(f'file uploaded to s3 with key {key}, response status is {status_code}')

    @disable_logging
    def upload_file(self, file_name, key):
        _ = self.client.upload_file(
        Filename=file_name,
        Bucket=self.bucket,
        Key=key
        )
        # status_code = response['ResponseMetadata']['HTTPStatusCode']
        # l.debug(f'file uploadefd to s3 with key {key}, response status is {status_code}')
    
    @disable_logging
    def download_if_not_cached_and_get_path(self, key):
        l.debug(f"received key: {key}")
        filename = conf.AWS_LOCAL_CACHE_FOLDER+key
        my_file = Path(filename)
        if not my_file.is_file(): # if we don't have cached file already
            l.debug('File is not found. Downloading')
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=str(key)
            )
            data = response['Body'].read()
            self._makedirs(filename)
            with open(filename, "wb") as f:
                f.write(data)
        else:
            l.debug('File is already cached')
        return filename
    
    @disable_logging
    def read_binary(self, key, try_cached=True):
        if try_cached:
            filename = conf.AWS_LOCAL_CACHE_FOLDER+key
            my_file = Path(filename)
            
            if my_file.is_file(): # if we have cached file already
                return open(filename, 'rb').read()

        response = self.client.get_object(
            Bucket=self.bucket,
            Key=str(key)
        )
        data = response['Body'].read()

        self._makedirs(filename)
        with open(filename, "wb") as f:
            f.write(data)

        return data

    @disable_logging
    def remove_file(self, key, also_from_cache=True):

        _ = self.client.delete_object(
            Bucket=self.bucket,
            Key=str(key)
        )
        
        # if not response['DeleteMarker']:
        #     raise ValueError(f'Cannot delete key {key} for s3 ')

        if also_from_cache:
            filename = conf.AWS_LOCAL_CACHE_FOLDER+key
            my_file = Path(filename)
            
            if my_file.is_file(): # if we have cached file already
                os.remove(filename) # will remove a file.
                l.debug(f'removed file {filename}')
            
    @staticmethod
    def _makedirs(filename):
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise