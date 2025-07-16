import logging
import logging.handlers
import os
import sys

def setup_queue_logging(log_queue):
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()
    
    h = logging.handlers.QueueHandler(log_queue)
    root.addHandler(h)
    root.setLevel(logging.INFO)

def setup_main_logger(log_dir: str, instance_name: str, log_queue):
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, f"{instance_name}.log")

    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)

    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    listener = logging.handlers.QueueListener(log_queue, file_handler, stream_handler)
    
    return listener