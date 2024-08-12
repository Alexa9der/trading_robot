import logging

# Setting up basic logging configuration
logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Creating a logger
logger = logging.getLogger()


def log_message(message: str) -> None:
    """
    Logs the passed message at the INFO level.

    :param message: The text of the message to log.
    """
    logger.info(message)


__all__ = ['log_message']


if __name__ == "__main__":
    # Example of using the function
    log_message('This is a test message for the logger.')