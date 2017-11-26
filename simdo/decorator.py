from functools import wraps
import time


TIME_FORMAT = "%b %d %Y - %H:%M:%S"


def timer(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("- Function {} tooks {}'s".format(func.__name__, end - start))
        return result

    return wrapper


def timer_format(time_format=TIME_FORMAT):
    def decorator(func):
        def decorated_func(*args, **kwargs):
            print("- Running '{}' on {}".format(
                func.__name__,
                time.strftime(time_format)
            ))
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print("- Finished '{}', execution time = {}'s ".format(
                func.__name__,
                end - start
            ))
            return result
        # decorated_func.__name__ = func.__name__
        return decorated_func
    return decorator


def log_method_calls(time_format=TIME_FORMAT):
    def wrapper(klass):
        for attr in dir(klass):
            if attr.startswith('__'):
                continue
            attr_ = getattr(klass, attr)
            decorated_a = timer_format(time_format)(attr_)
            setattr(klass, attr, decorated_a)
        return klass
    return wrapper
