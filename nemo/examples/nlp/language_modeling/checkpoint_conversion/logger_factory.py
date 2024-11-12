import logging
import sys

class LoggerFactory:
    @staticmethod
    def create_logger(name=None, level=logging.INFO):
        """create a logger

        Args:
            name (str): name of the logger
            level: level of logger

        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        log_format_with_line_numbers = '%(asctime)s [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s'
        formatter = logging.Formatter(log_format_with_line_numbers)

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        return logger_
