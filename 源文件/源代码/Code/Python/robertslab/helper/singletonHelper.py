import threading

__all__ = ['SingletonMixin', 'SingletonMixinThreadsafe']

class SingletonMixin(object):
    __singleton_instance = None

    @classmethod
    def instance(cls):
        if not cls.__singleton_instance:
            cls.__singleton_instance = cls()
        return cls.__singleton_instance

class SingletonMixinThreadsafe(object):
    # from https://gist.github.com/werediver/4396488
    # Based on tornado.ioloop.IOLoop.instance() approach
    __singleton_lock = threading.Lock()
    __singleton_instance = None

    @classmethod
    def instance(cls):
        if not cls.__singleton_instance:
            with cls.__singleton_lock:
                if not cls.__singleton_instance:
                    cls.__singleton_instance = cls()
        return cls.__singleton_instance