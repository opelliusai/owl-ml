'''
Created on: Apr. 10 2024

@author: OpelliusAI
@summary: Log management
'''

import os
import logging.handlers
from src.config.run_config import paths, infolog

def setup_logging():
    '''
    Logging setup, using run_config infolog information
    '''
    main_path = paths["main_path"]
    log_folder = infolog["logs_folder"]
    logfile_name = infolog["logfile_name"]
    logfile_path = os.path.join(main_path, log_folder, logfile_name)
    
    # Ensure the log folder exists
    os.makedirs(os.path.join(main_path, log_folder), exist_ok=True)
    
    # Create a formatter and handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        logfile_path, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Create a logger and add handlers to it
    logger = logging.getLogger(infolog["project_name"])
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Setup logging once and export the logger
logger = setup_logging()
