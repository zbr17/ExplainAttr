import logging
from termcolor import colored
import functools
import sys
import os

global _logger
_logger = None

@functools.lru_cache()
def config(output_dir, dist_rank=0, name='LOG'):
    global _logger
    # create logger
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    os.makedirs(output_dir, exist_ok=True)
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        _logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(file_handler)

def info(msg, *args, **kwargs):
    _logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    _logger.warning(msg, *args, **kwargs)