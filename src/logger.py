import logging
from loguru import logger
import apprise
import sys
from typing import Optional, overload
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

APPRISE_GMAIL = os.getenv("APPRISE_GMAIL")
APPRISE_PWD = os.getenv("APPRISE_PWD")

__all__ = ["loguru_setup"]

class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[91m",  # Red
    }
    RESET = "\033[0m"

    def format(self, record):
        log_msg = super().format(record)
        color = self.COLORS.get(record.levelname, "")
        if color:
            return f"{color}{log_msg}{self.RESET}"
        return log_msg


class LightLogger(logging.Logger):
    def __init__(
        self,
        name: str = "LightLogger",
        log_path: str = "../logs",
        log_file: Optional[str] = None,
        level: str = "INFO",
    ):
        """A lightweight logger that inherits from the base Logger class.

        Args:
            name (str, optional): name of the project. Defaults to "LightLogger".
            log_path (str, optional): path to the log directory. Defaults to "../logs".
            log_file (Optional[str], optional): name of the log file. Defaults to the same as name.
        """
        super().__init__(name, getattr(logging, level))
        if log_file is None:
            log_file = name
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Output to both file and stdout with color for terminal
        file_handler = logging.FileHandler(
            f"{log_path}/{log_file}_{current_date}.log", mode="w"
        )
        stream_handler = logging.StreamHandler()
        # Add color support for terminal output
        stream_handler.setFormatter(
            ColorFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - Line: %(lineno)d",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logging.basicConfig(
            level=getattr(logging, level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - Line: %(lineno)d",
            # Output to both file and stdout
            handlers=[file_handler, stream_handler],
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(name)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log an info message."""
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log a debug message with optional exception info."""
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log a critical message."""
        self.logger.critical(msg, *args, **kwargs)


def loguru_setup() -> logger: # type: ignore
    """Setup loguru logger with default configuration."""
    logger.remove()  # Remove the default logger
    logger.add(
        sys.stdout,
        colorize=True,
        level="DEBUG",
    )
    logger.add(
        "logs/debug.log",
        rotation="100MB",
        level="DEBUG",
        serialize=True,
    )
    notifier = apprise.Apprise()
    notifier.add(f"mailto://{APPRISE_GMAIL}:{APPRISE_PWD}@gmail.com")
    logger.add(
        notifier.notify,
        level="ERROR",
        format="{message}",
        serialize=True,
        filter={"apprise": False},
    )

    return logger


if __name__ == "__main__":
    logger = loguru_setup()
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    # logger.critical("This is a critical message.")
