import logging
logger = logging.getLogger(__name__)

def setup_logging(level="INFO", origin=False):
    if isinstance(level, str):
        level_obj = getattr(logging, level.upper())
    else:
        level_obj = level

    logging.basicConfig(
        level=level_obj,
        format="%(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True
    )

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)

    if origin:
        logging.basicConfig(format="%(levelname)s - %(filename)s - %(message)s")

    return logging.getLogger(__name__)