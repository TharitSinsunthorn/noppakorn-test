import time
from numba import njit


def decorator_timer(some_function):
    from time import time

    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        end = time()-t1
        return result, end
    return wrapper

@decorator_timer
def func1():
    for i in range(100000, -1, -1):
        pass

@decorator_timer
@njit
def func2():
    for i in range(100000, -1, -1):
        pass

while(True):
    _, times = func1()
    _, times2 = func2()
    
    print(f"calculation time: {times} seconds")
    print(f"calculation time jit: {times2} seconds")
    print(f"Speed compare: {times2/times}\n")
    
    time.sleep(1)
