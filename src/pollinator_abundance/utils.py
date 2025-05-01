import time
def time_counter(f):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result =  f(*args, **kwargs)
        print(f"{f.__name__}: {time.time() - start_time:.6f} seconds to execute")
        return result
    return wrapper