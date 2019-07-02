import sys


def log(message):
    """Print message to console or terminal.

    :param message: [str] The message to print.
    """

    sys.stdout.write(message + '\n')
    sys.stdout.flush()
