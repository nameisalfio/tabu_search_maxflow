# src/utils/logging_utils.py

import logging
import logging.handlers
import os
import sys

def setup_queue_logging(log_queue):
    """
    Configura un logger per un processo worker per inviare messaggi a una coda.
    Questa funzione rimane invariata.
    """
    h = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    # Pulisce eventuali handler ereditati per sicurezza
    if root.hasHandlers():
        root.handlers.clear()
    root.addHandler(h)
    root.setLevel(logging.INFO)

def setup_main_logger(log_dir: str, instance_name: str, log_queue):
    """
    *** MODIFICATA ***
    Configura il logger del processo principale E crea il listener per la coda.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, f"{instance_name}.log")

    # Handler per scrivere su file
    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)

    # Handler per stampare in console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)

    # *** NUOVO ***: Configura il logger del processo principale (quello che chiama questa funzione)
    # per usare direttamente gli handler, senza passare dalla coda.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Pulisce eventuali handler precedenti
    if root.hasHandlers():
        root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # Il listener prende i messaggi dalla coda (dai worker) e li invia
    # AGLI STESSI handler usati dal processo principale.
    listener = logging.handlers.QueueListener(log_queue, file_handler, stream_handler)
    
    return listener