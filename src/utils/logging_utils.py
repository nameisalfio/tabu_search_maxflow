import logging
import logging.handlers
import os
import sys

def setup_queue_logging(log_queue):
    """
    Configura un logger per un processo worker per inviare messaggi a una coda.
    Pulisce eventuali handler preesistenti per evitare duplicazioni.
    """
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()
    
    h = logging.handlers.QueueHandler(log_queue)
    root.addHandler(h)
    root.setLevel(logging.INFO)

def setup_main_logger(log_dir: str, instance_name: str, log_queue):
    """
    Configura il logger del processo principale per scrivere su file e console
    e crea un listener che inoltra i messaggi dalla coda agli stessi handler.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, f"{instance_name}.log")

    # Handler per il file di log
    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)

    # Handler per la console (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)

    # Configura il logger del processo principale
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # Il listener ascolta la coda e inoltra i record agli handler
    listener = logging.handlers.QueueListener(log_queue, file_handler, stream_handler)
    
    return listener