import logging

from py.config import Config as conf

l = logging.getLogger('evo')
l.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f'{conf.LOG_FOLDER}debug.log')
fh.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh2 = logging.FileHandler(f'{conf.LOG_FOLDER}info.log')
fh2.setLevel(logging.INFO)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
fh.setFormatter(formatter)
fh2.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
l.addHandler(fh)
l.addHandler(fh2)
l.addHandler(ch)