"""Sets logging defaults for the entire project"""

import logging
import traceback
from enum import StrEnum
from logging import Logger, handlers
from pathlib import Path
from typing import Callable, Literal, Optional

BACKUP_COUNT = 10
MAX_BYTES = 200000
FILE_PATH = Path.cwd() / "logs"
LOGGER_LEVEL = logging.DEBUG


class LogColor(StrEnum):
    RED = "\033[31m{text}\033[0m"
    GREEN = "\033[32m{text}\033[0m"
    YELLOW = "\033[33m{text}\033[0m"
    BLUE = "\033[34m{text}\033[0m"


class CustomLogger:
    def __init__(self, logger: Logger):
        self.logger = logger

    def _check_len(self, args: tuple):
        if len(args) > 1:
            # TODO: Update logging to handle more than 1 arg passed in
            raise NotImplementedError(
                "Logging currently only supports 1 arg. Update this class to support more if needed"
            )

    def print_log(
        self,
        log_func: Callable,
        color: LogColor,
        text: str,
        args: tuple,
        text_prefix: Optional[str] = "",
    ):
        self._check_len(args)
        text = f"{text_prefix} {text}"
        colored_text = color.value.format(text=text)
        if args:
            # HACK: Only works for a single arg
            colored_args = color.value.format(text=args[0])
            log_func(colored_text, colored_args)
        else:
            log_func(colored_text)

    def exception(self, text, *args):
        self.logger.error("(Traceback) %s", traceback.format_exc())
        self.print_log(log_func=self.logger.error, color=LogColor.RED, text=text, args=args, text_prefix="(Exception)")

    def error(self, text, *args):
        self.logger.error("(Traceback) %s", traceback.format_exc())
        self.print_log(log_func=self.logger.error, color=LogColor.RED, text=text, args=args, text_prefix="(Error)")

    def traceback(self):
        self.logger.error(traceback.format_exc())

    def warning(self, text, *args):
        self.print_log(
            log_func=self.logger.warning, color=LogColor.YELLOW, text=text, args=args, text_prefix="(Warning)"
        )

    def info(self, text, *args):
        self.print_log(log_func=self.logger.info, color=LogColor.GREEN, text=text, args=args, text_prefix="(Info)")

    def debug(self, text, *args):
        self.print_log(log_func=self.logger.debug, color=LogColor.BLUE, text=text, args=args, text_prefix="(Debug)")


def set_logging(
    handler_option: Literal["stream", "file", "both"],
    name: str,
    log_name="file",
) -> CustomLogger:
    """Sets the logging for the project

    Parameters
    ----------
    handler_option
        The options for which type of handler you want for the logger
    name
        The name of the module you are running or any other name you want to
        identify the log line by
    log_name
        The name of the log file
    """

    logger = logging.getLogger(name + "_" + handler_option)
    logger.propagate = False

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set the log levels for SQLAlchemy engine and pool loggers
    # logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
    # logging.getLogger("sqlalchemy.pool").setLevel(logging.DEBUG)

    # Create stream logging
    if handler_option == "stream" or handler_option == "both":
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG)
        c_format = logging.Formatter("%(asctime)-15s - %(name)s - %(levelname)s - %(message)s")
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

    # Create file logging
    if handler_option == "file" or handler_option == "both":
        FILE_PATH.mkdir(parents=True, exist_ok=True)  # Create if doesn't exist
        logging_file_name = FILE_PATH / f"{log_name}.log"

        f_handler = handlers.RotatingFileHandler(logging_file_name, backupCount=BACKUP_COUNT, maxBytes=MAX_BYTES)
        f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    logger.setLevel(LOGGER_LEVEL)

    return CustomLogger(logger)
