#!/usr/bin/env python
from inspect import getargspec,isclass
import os,sys
import numpy as np; np.set_printoptions(precision=1, threshold=1e6, linewidth=1e6)
import pytest
import unittest

thisScriptPath = os.path.realpath(__file__)

# from robertslab.test.md.mdIO.trajectoryToSFile import TrajectoryToSFileTestBase
from robertslab.test.md.rmsd.sisd.rmsdSISD import RMSDSISDTestBase

class TestRunner(object):
    def __init__(self, varsDict, localsDict):
        self.varsDict = varsDict
        self.localsDict = localsDict

    def GetTestBases(self):
        return [TestBase for TestBase in self.varsDict.values() if isclass(TestBase) and TestBase.__name__[-8:]=='TestBase']
    
    def GetTestCases(self, exportToLocals=False):
        testCases = []
        for TestBase in self.GetTestBases():
            testCaseName = TestBase.__name__[:-4] + 'Case'
            TestCase = type(testCaseName, (TestBase, unittest.TestCase), {})

            if exportToLocals:
                self.localsDict[testCaseName] = TestCase
            
            testCases.append(TestCase)
        return testCases
    
    def RunUnittest(self, failfast=False, **kwargs):
        self.GetTestCases(exportToLocals=True)
        unittest.main(failfast=failfast, **kwargs)
    
    def SimpleRun(self):
        for TestCase in self.GetTestCases():
            TestCase.simpleRun()

if __name__ == '__main__':
    testRunner = TestRunner(varsDict=vars(), localsDict=locals())
    testRunner.RunUnittest()
    # testRunner.SimpleRun()
    
    
    
#     pytest.main(args= thisScriptPath + ' -s')