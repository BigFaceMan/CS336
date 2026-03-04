import cProfile
import pstats

def profile(func):
    def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumtime').print_stats(10)
            return result
    return wrapper