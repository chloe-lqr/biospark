from collections import OrderedDict
from operator import iadd
from six import print_
import time

__all__ = ['TimeDecorator', 'TimeDecoratorFactory', 'TimeDecoratorSpark', 'TimeWith', 'TimeWithSpark']

class TimeDecoratorFactory(object):
    '''
    modified from https://zapier.com/engineering/profiling-python-boss/
    '''
    totals = OrderedDict()

    @classmethod
    def initSparkTimingAccumulators(cls, sc, names):
        for name in names:
            cls.totals[name] = sc.accumulator(0.0)

    @classmethod
    def setCallPrint(cls, doPrint=True):
        if doPrint:
            cls.__call__ = cls._callPrint
        else:
            cls.__call__ = cls._callNoPrint

    def _callPrint(cls, f):
        def f_timer(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            end = time.time()
            elapsed = end - start
            cls.totals[f.__name__] = iadd(cls.totals.get(f.__name__, 0), elapsed)
            print_(f.__name__, 'took', elapsed, 'time')
            return result
        return f_timer

    def _callNoPrint(cls, f):
        def f_timer(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            end = time.time()
            cls.totals[f.__name__] = iadd(cls.totals.get(f.__name__, 0), (end - start))
            return result
        return f_timer

TimeDecorator = TimeDecoratorFactory()
TimeDecorator.setCallPrint(doPrint=False)

class TimeWith(object):
    '''
    modified from https://zapier.com/engineering/profiling-python-boss/
    '''
    totals = OrderedDict()

    @classmethod
    def initSparkTimingAccumulators(cls, sc, names):
        for name in names:
            cls.totals[name] = sc.accumulator(0.0)

    @classmethod
    def setCheckpointPrint(cls, doPrint=True):
        if doPrint:
            cls.checkpoint = cls._checkpointPrint
        else:
            cls.checkpoint = cls._checkpointNoPrint

    def __init__(self, name=''):
        self.name = name
        self.start = time.time()

    @property
    def elapsed(self):
        return time.time() - self.start

    def _checkpointNoPrint(self, name=''):
        self.__class__.totals[self.name]+=self.elapsed
        # self.__class__.totals[name] = iadd(self.totals.get(name, 0), self.elapsed)

    def _checkpointPrint(self, name=''):
        elapsed = self.elapsed
        self.__class__.totals[self.name] = iadd(self.totals.get(self.name, 0), elapsed)
        print_('{timer} {checkpoint} took {elapsed} seconds'.format(timer=self.name,
                                                                   checkpoint=name,
                                                                   elapsed=self.elapsed).strip())

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.checkpoint('finished')
        pass
TimeWith.setCheckpointPrint(doPrint=False)

class TimeSpark(object):
    def __init__(self, sc=None, count=0, funcs=None, names=None):
        self.freeAccumulators = []
        self.checkpointCounts = OrderedDict()
        self.checkpointTimes = OrderedDict()

        if sc is not None:
            self.initTimingAccumulators(sc=sc, count=count, funcs=funcs, names=names)
    
    def initName(self, name):
        self.checkpointCounts[name] = self.checkpointCounts[name] if name in self.checkpointCounts else self.freeAccumulators.pop()
        self.checkpointTimes[name] = self.checkpointTimes[name] if name in self.checkpointTimes else self.freeAccumulators.pop()
    
    def initTimingAccumulators(self, sc, count=0, funcs=None, names=None):
        for i in range(count*2):
            self.freeAccumulators.append(sc.accumulator(0))
        if names:
            for name in names:
                self.checkpointCounts[name] = sc.accumulator(0)
                self.checkpointTimes[name] = sc.accumulator(0.0)

    def __str__(self):
        nameLen = max(map(len, self.checkpointCounts.keys()))
        s = ''
        for (name,count),(name,time) in zip(self.checkpointCounts.items(), self.checkpointTimes.items()):
            s+='{name:{nameLen}}: {time:8.3e} {count}\n'.format(name=name, nameLen=nameLen, time=time.value, count=count.value)
        return s

class TimeDecoratorSpark(TimeSpark):
    def __call__(self, f):
        self.initName(f.__name__)

        def f_timer(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            end = time.time()
            self.checkpointCounts[f.__name__]+=1
            self.checkpointTimes[f.__name__] += end - start
            return result
        return f_timer

class TimeWithSpark(TimeSpark):
    @property
    def elapsed(self):
        return time.time() - self.start

    def __call__(self, name=''):
        self.name = name
        return self

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.checkpoint('finished')
        pass

    def checkpoint(self, checkpointName=None):
        self.checkpointCounts[self.name]+=1
        self.checkpointTimes[self.name]+=self.elapsed

        self.start = time.time()