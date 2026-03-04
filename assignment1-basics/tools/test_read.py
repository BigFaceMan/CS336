import os
import time
import psutil 
import resource 

# read_path = "/home/spsong/Code/cs336/assignment1-basics/data/test_read.txt"
# with open(read_path) as f:
#     for idx, text in enumerate(f):
#         print(idx)
#         print(text)

def log(func):
    def wrapper(*args, **kwargs):
        print("func run")
        st = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"cost {end - st}")
        return result
    return wrapper

def memory_limit(max_mem):
    def decorator(f):
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            prev_limits = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (process.memory_info().rss + max_mem, -1))
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                # Even if the function above fails (e.g., it exceeds the
                # memory limit), reset the memory limit back to the
                # previous limit so other tests aren't affected.
                resource.setrlimit(resource.RLIMIT_AS, prev_limits)

        return wrapper

    return decorator

@log
def hihi():
    print("hihiihihiihihiihii")

wp = log(hihi)

@memory_limit(int(1e6))
def new_mem(sz):
    ls = [i for i in range(sz)]

nnm = memory_limit(int(1e6))(new_mem)


def gen_read():
    for i in range(100):
        yield [i, i + 1]

gr = gen_read()

for k in gen_read():
    print(k)



if __name__ == "__main__":
    new_mem(1000)