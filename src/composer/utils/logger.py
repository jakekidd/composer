import os
import logging
from datetime import datetime, timezone
from typing import Optional

class Severity:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"

class Logger:
    def __init__(self, log_dir: str = 'logs/', log_file: str = 'app.log', should_print: bool = False) -> None:
        self.log_dir = log_dir
        self.log_file = log_file
        self.should_print = should_print
        self.logger = logging.getLogger('symphonic_logger')
        self.logger.setLevel(logging.DEBUG)
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(log_dir, log_file))
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s [%(asctime)s] [%(filename)s:%(lineno)d]: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log(self, severity: str, file_name: str, function_name: str, message: str, class_name: Optional[str] = None, timestamp: Optional[str] = None) -> None:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        if class_name is None:
            log_message = f"{severity} [{timestamp}] [{file_name}:{function_name}]: {message}"
        else:
            log_message = f"{severity} [{timestamp}] [{file_name}:{class_name}:{function_name}]: {message}"
        
        # Log to file
        if severity == Severity.DEBUG:
            self.logger.debug(log_message)
        elif severity == Severity.INFO:
            self.logger.info(log_message)
        elif severity == Severity.WARN:
            self.logger.warning(log_message)
        elif severity == Severity.ERROR:
            self.logger.error(log_message)

        # Print to console
        if self.should_print:
            if severity == Severity.DEBUG:
                print(f"\033[94m{log_message}\033[0m")  # Blue
            elif severity == Severity.INFO:
                print(f"\033[92m{log_message}\033[0m")  # Green
            elif severity == Severity.WARN:
                print(f"\033[93m{log_message}\033[0m")  # Yellow
            elif severity == Severity.ERROR:
                print(f"\033[91m{log_message}\033[0m")  # Red

    def debug(self, file_name: str, function_name: str, message: str, class_name: Optional[str] = None, timestamp: Optional[str] = None) -> None:
        self.log(Severity.DEBUG, file_name, function_name, message, class_name, timestamp)

    def info(self, file_name: str, function_name: str, message: str, class_name: Optional[str] = None, timestamp: Optional[str] = None) -> None:
        self.log(Severity.INFO, file_name, function_name, message, class_name, timestamp)

    def warn(self, file_name: str, function_name: str, message: str, class_name: Optional[str] = None, timestamp: Optional[str] = None) -> None:
        self.log(Severity.WARN, file_name, function_name, message, class_name, timestamp)

    def error(self, file_name: str, function_name: str, message: str, class_name: Optional[str] = None, timestamp: Optional[str] = None) -> None:
        self.log(Severity.ERROR, file_name, function_name, message, class_name, timestamp)

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)
