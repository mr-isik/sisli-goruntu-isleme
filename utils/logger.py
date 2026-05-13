"""
Merkezi loglama yapılandırması.
"""

import logging
import sys

import config


def get_logger(name: str) -> logging.Logger:
    """
    Modül bazlı logger oluşturur.

    Args:
        name: Logger adı (genellikle __name__).

    Returns:
        Yapılandırılmış logger nesnesi.
    """
    logger = logging.getLogger(name)

    # Zaten handler varsa tekrar ekleme
    if not logger.handlers:
        logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

        formatter = logging.Formatter(
            fmt=config.LOG_FORMAT,
            datefmt=config.LOG_DATE_FORMAT,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
