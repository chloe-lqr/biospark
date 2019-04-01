import cProfile
import pstats

from robertslab.helper.pysparkImportHelper import PysparkImport
pyspark = PysparkImport()
from pyspark.profiler import Profiler as BaseSparkProfiler, PStatsParam
# BaseSparkProfiler, PStatsParam = PysparkImport(subpkgs=['profiler.Profiler', 'profiler.PStatsParam'])

__all__ = ['SparkProfileDecorator']

class SparkProfileDecorator(BaseSparkProfiler):
    def __init__(self, sc):
        super(SparkProfileDecorator, self).__init__(ctx=sc)
        # Creates a new accumulator for combining the profiles of different
        # partitions of a stage
        self._accumulator = sc.accumulator(None, PStatsParam)

    def __call__(self, func):
        def profiledFunc(*args, **kwargs):
            """ Runs and profiles the method passed in. A profile object is converted to stats and accumulated. """
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                return result
            finally:
                stats = pstats.Stats(profile)
                stats.stream = None  # make it picklable
                stats.strip_dirs()
                # Adds a new profile to the existing accumulated value
                self._accumulator.add(stats)
        return profiledFunc

    def print_stats(self, sort='cumtime'):
        self.stats().sort_stats(sort).print_stats()

    def stats(self):
        return self._accumulator.value