import logging
from logging.handlers import RotatingFileHandler
from transformers import logging as hf_logging

def setup_logger(name: str = "BeevibeLogger", log_file: str = "beevibe.log", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger.
        log_file (str): The file where logs will be stored. Defaults to 'beevibe.log'.
        level (int): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate log messages
    logger.propagate = False

    if not logger.handlers:

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Create a rotating file handler
        file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=2)
        file_handler.setLevel(level)

        # Define log message format
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    hf_logging.set_verbosity_error()

    return logger