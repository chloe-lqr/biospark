import cProfile

__all__ = ['ProfileDecorator']

class ProfileDecorator(object):
    def __call__(self, func):
        def profiled_func(*args, **kwargs):
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                return result
            finally:
                profile.print_stats(sort='cumtime')
        return profiled_func