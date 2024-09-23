# utils/logger.py

import logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Create handlers
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # Add handlers to logger
    logger.addHandler(ch)
    return logger
