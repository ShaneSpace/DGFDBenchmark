import logging
# https://zhuanlan.zhihu.com/p/454463040

def create_logger(file_name = 'log_files'):
    # Set up root logger, and add a file handler to root logger
    logging.basicConfig(filename = file_name +'.log',
                        level = logging.INFO,
                        format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    # Create logger, set level, and add stream handler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    shandler = logging.StreamHandler()
    logger.addHandler(shandler)

    return logger


# Example
# Log message of severity INFO or above will be handled
# parent_logger = create_logger(file_name = 'log_files')
# parent_logger.debug('Debug message')
# parent_logger.info('Info message')
# parent_logger.warning('Warning message')
# parent_logger.error('Error message')
# parent_logger.critical('Critical message')