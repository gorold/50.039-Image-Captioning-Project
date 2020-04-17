import os

def log_print(text, logger, log_only = False):
    if not log_only:
        print(text)
    if logger is not None:
        logger.info(text)