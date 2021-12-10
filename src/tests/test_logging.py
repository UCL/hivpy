import logging
import os.path as path

from hivpy.config import LoggingConfig


def test_logging_levels(tmp_path, capsys):
    d = tmp_path / "log"
    d.mkdir()
    log_cfg = LoggingConfig(log_dir=d, logfile="hivpy.log")
    log_cfg.start_logging()
    logger = logging.getLogger("Testing")
    TEST_DEBUG = "Test debug."
    TEST_INFO = "Test info."
    TEST_WARNING = "Test warning."
    TEST_ERROR = "Test error."
    TEST_CRITICAL = "Test critical."
    logger.debug(TEST_DEBUG)
    logger.info(TEST_INFO)
    logger.warning(TEST_WARNING)
    logger.error(TEST_ERROR)
    logger.critical(TEST_CRITICAL)
    console_out = capsys.readouterr().err
    file_out = open(path.normpath(path.join(d, "hivpy.log")), 'r')
    file_text = file_out.read()
    file_out.close()
    # File Logging Checks
    assert TEST_DEBUG in file_text
    assert TEST_INFO in file_text
    assert TEST_WARNING in file_text
    assert TEST_ERROR in file_text
    assert TEST_CRITICAL in file_text
    # Console Logging Checks
    assert not (TEST_DEBUG in console_out)
    assert not (TEST_INFO in console_out)
    assert TEST_WARNING in console_out
    assert TEST_ERROR in console_out
    assert TEST_CRITICAL in console_out
