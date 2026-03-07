
import time
import colorama
def test_log(test_name):
    def decorator(f):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print(f"{colorama.Fore.BLUE}------------------------------Test {test_name}------------------------------")
            result = f(*args, **kwargs)
            end_time = time.time()
            print(
                f"{colorama.Fore.GREEN} {test_name} passed, cost {end_time - start_time} s {colorama.Style.RESET_ALL}"
            )
            return result

        return wrapper

    return decorator