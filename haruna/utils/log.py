import logging
import os
from datetime import datetime

LOG_DIR = 'logs'
FORMAT = '[%(asctime)s][%(levelname)s][%(name)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d_%H:%M:%S'

LEVEL_MAP = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOSET': logging.NOTSET,
}


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # set level
    logger.setLevel(logging.DEBUG)

    # formatter
    formatter = logging.Formatter(FORMAT, datefmt=DATE_FORMAT)

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_level = LEVEL_MAP.get(os.getenv('LOG_LEVEL', 'INFO').upper())
    stream_handler.setLevel(stream_level)
    logger.addHandler(stream_handler)

    # file handler
    today = datetime.now().strftime('%y%m%d')
    log_file = os.path.join(LOG_DIR, '{}.log'.format(today))
    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # prevent twice log
    logger.propagate = False
    return logger
