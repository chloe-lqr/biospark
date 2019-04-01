import os
from copy import deepcopy

import mdtraj as md
import numpy as np

from robertslab.helper.timerHelper import timeDecorator, timeWith
from robertslab.md.mdIO.trajectoryToSFile import TrajectoryToSFile
from robertslab.md.rmsd.sisd.rmsdSISD import RMSDSISD
from robertslab.test.md.testBase import TestBase

thisScriptDir = os.path.dirname(os.path.realpath(__file__))
testDataDir = os.path.join(thisScriptDir, '..', '..', 'testData')

testDataName = '3d6z_aligned_vmd_wat_ion'
topPath = os.path.join(testDataDir, '.'.join((testDataName, 'pdb')))
trajPath = os.path.join(testDataDir, '.'.join((testDataName, 'dcd')))

class RMSDSISDTestSet(object):
    @timeDecorator
    def test_rmsd_all_atoms(self):
        '''
        tests basic rmsd function
        '''
        selection = None
        presliceSFMD = False
        repeats = 50

        self.generateSFMD(selection, presliceSFMD, repeats=repeats)
        self.loadMDTrajObjects(selection, repeats=repeats)
        self.mdtrajRMSD = md.rmsd(self.mdtraj, self.mdtrajReference)

        #with timeWith(vars()['self']._testMethodName) as timer:
        with timeWith('test_rmsd_all_atoms__inner_loop') as timer:
            self.loadSFMD(selection, presliceSFMD)
            timer.checkpoint('done loading data from .sfmd file')
            self.sparkmdRMSD = self.rmsdSISD.run()

        self.assertArraysAllClose(self.mdtrajRMSD, self.sparkmdRMSD)

    @timeDecorator
    def test_rmsd_CA_only(self):
        '''
        tests loading an atom-complete set of data from an sfmd file, slicing it, and then running rmsd on it
        '''
        selection = 'name CA'
        presliceSFMD = False

        self.generateSFMD(selection, presliceSFMD)
        self.loadMDTrajObjects(selection)
        self.mdtrajRMSD = md.rmsd(self.mdtraj, self.mdtrajReference)

        with timeWith('test_rmsd_CA_only__inner_loop') as timer:
            self.loadSFMD(selection, presliceSFMD)
            timer.checkpoint('done loading data from .sfmd file')
            self.sparkmdRMSD = self.rmsdSISD.run()
            timer.checkpoint('test_rmsd_CA_only__innermost_loop')

        self.assertArraysEqual(self.mdtrajRMSD, self.sparkmdRMSD)

    @timeDecorator
    def test_rmsd_CA_only_presliced_sfmd(self):
        '''
        tests loading a pre-sliced set of data from an sfmd, and then running rmsd on it
        '''
        selection = 'name CA'
        presliceSFMD = True
        sfileSuffix = '__CA_only'

        self.generateSFMD(selection, presliceSFMD, sfileSuffix)
        self.loadMDTrajObjects(selection)
        self.mdtrajRMSD = md.rmsd(self.mdtraj, self.mdtrajReference)

        with timeWith('test_rmsd_CA_only_presliced_sfmd__inner_loop') as timer:
            self.loadSFMD(selection, presliceSFMD)
            timer.checkpoint('done loading data from .sfmd file')
            self.sparkmdRMSD = self.rmsdSISD.run()
            timer.checkpoint('test_rmsd_CA_only_presliced_sfmd__innermost_loop')

        self.assertArraysEqual(self.mdtrajRMSD, self.sparkmdRMSD)


class RMSDSISDTestBase(TestBase, RMSDSISDTestSet):
    @classmethod
    def setUpClass(cls):
        # load data directly from the original md trajectory file into a couple of mdtraj trajectories
        cls._mdtrajReference = md.load(topPath)
        cls._top = cls._mdtrajReference.top
        cls._mdtraj = md.load(trajPath, top=cls._top)

    def generateSFMD(self, selection=None, presliceSFMD=False, sfileSuffix=None, repeats=1):
        # generate the SFileMD
        if presliceSFMD:
            self.trajectoryToSFile = TrajectoryToSFile(trajPath, self._top, repeats=repeats, selection=selection, sfileSuffix=sfileSuffix, writeTarget='cat')
        else:
            self.trajectoryToSFile = TrajectoryToSFile(trajPath, self._top, repeats=repeats, selection=None, sfileSuffix=sfileSuffix, writeTarget='cat')
        self.trajectoryToSFile.run()

    def loadMDTrajObjects(self, selection=None, repeats=1):
        # slice the atoms in the mdtraj/mdtrajReference objects to match what we put in our sfilemd file
        if selection is not None:
            self.atom_selection = self._mdtraj.top.select(selection)
            self.mdtraj = self._mdtraj.atom_slice(self.atom_selection)
            self.mdtrajReference = self._mdtrajReference.atom_slice(self.atom_selection)
        else:
            self.mdtraj = deepcopy(self._mdtraj)
            self.mdtrajReference = deepcopy(self._mdtrajReference)
        if repeats > 1:
            self.mdtraj.xyz = np.tile(self.mdtraj.xyz, (repeats, 1, 1))

    def loadSFMD(self, selection=None, presliceSFMD=False):
        # create an RMSDSISD object from our new SFileMD
        refFrame = deepcopy(self._mdtrajReference.xyz)
        self.rmsdSISD = RMSDSISD(sfilePath=self.trajectoryToSFile.sfilePath, refFrame=refFrame, selection=selection, sliceFrames=(not presliceSFMD), top=self._top)

    def loadData(self, selection=None, presliceSFMD=False, sfileSuffix=None):
        # generate the SFileMD
        self.generateSFMD(selection, presliceSFMD, sfileSuffix)
        # slice the atoms in the mdtraj/mdtrajReference objects to match what we put in our sfilemd file
        self.loadMDTrajObjects(selection)
        # create an RMSDSISD object from our new SFileMD
        self.loadSFMD(selection, presliceSFMD)