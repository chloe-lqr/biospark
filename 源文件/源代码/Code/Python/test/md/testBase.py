from inspect import ismethod
import numpy as np

class TestBase(object):
    @classmethod
    def simpleRun(cls):
        obj = cls(methodName='simpleRun')
        obj.setUpClass()
        for testMethod in (obj.__getattribute__(methodName) for methodName in dir(obj) if ismethod(obj.__getattribute__(methodName)) and methodName[:4]=='test'):
            obj.setUp()
            testMethod()
            obj.tearDown()
        obj.tearDownClass()
    
    def setUp(self):
        pass
        
    def tearDown(self):
        pass
    
    def loadData(self, full=False, **kwargs):
        pass

    def assertArraysAllClose(self, arr1, arr2):
        try:
            testBool = np.allclose(arr1, arr2)
        except AttributeError:
            testBool = False
        self.assertTrue(testBool, msg='arrays not all close: %s\n%s' % (arr1.ravel()[:100].tolist(), arr2.ravel()[:100].tolist()))

    def assertArraysEqual(self, arr1, arr2):
        try:
            testBool = (arr1==arr2).all()
        except AttributeError:
            testBool = False
        self.assertTrue(testBool, msg='arrays not equal: %s\n%s' % (arr1.ravel()[:100].tolist(), arr2.ravel()[:100].tolist()))
        
    def assertSetsEqual(self, set1, set2):
        testBool = set1==set2
        self.assertTrue(testBool, msg='sets not equal: %s\n%s' % (set1, set2))