import logging

def newLogger(name):
  logger = logging.getLogger(name)
  formatter = logger.Formatter('%(asctime)s:%(levelname)s::  %(message)s')
  file_handler = logging.FileHandler('models.log')
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)