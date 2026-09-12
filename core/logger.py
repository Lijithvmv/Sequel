import logging
from datetime import datetime
from functools import wraps


def get_logger(name):
    return logging.getLogger(name)

def set_session_id(session_id):
    # Dummy implementation for compatibility
    pass

def log_performance(f):
    """Decorator to log performance metrics"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        logger = get_logger(f.__module__)
        start_time = datetime.now()
        try:
            result = f(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Function {f.__name__} completed in {duration:.2f} seconds")
            return result
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"Function {f.__name__} failed after {duration:.2f} seconds: {str(e)}")
            raise
    return decorated_function 