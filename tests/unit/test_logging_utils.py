import inspect
import logging

from doc_auth.utils.logging_utils import get_logger


def test_get_logger_returns_configured_logger() -> None:
    sig = inspect.signature(get_logger)
    params = list(sig.parameters.values())

    if not params:
        logger = get_logger()
    elif params[0].default is not inspect.Parameter.empty:
        logger = get_logger()
        named_logger = get_logger("doc_auth.test.logger")
        assert isinstance(named_logger, logging.Logger)
        assert named_logger.name == "doc_auth.test.logger"
    else:
        logger = get_logger("doc_auth.test.logger")
        assert logger.name == "doc_auth.test.logger"

    assert isinstance(logger, logging.Logger)
    logger.info("unit test log message")
