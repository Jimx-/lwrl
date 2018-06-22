import logging
from logging.config import dictConfig


def init_logging():
    logging_config = dict(
        version=1,
        formatters={
            'f': {
                'format':
                '%(asctime)s (%(threadName)s) %(levelname)-8s %(message)s'
            }
        },
        handlers={
            'h': {
                'class': 'logging.StreamHandler',
                'formatter': 'f',
                'level': logging.DEBUG
            }
        },
        root={
            'handlers': ['h'],
            'level': logging.DEBUG,
        },
    )

    dictConfig(logging_config)


def begin_section(section):
    return '<' + '=' * 20 + ' ' + section + ' ' + '=' * 20 + '>'
