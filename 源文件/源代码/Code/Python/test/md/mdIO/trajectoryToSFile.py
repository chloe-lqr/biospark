import os

import mdtraj as md
import numpy as np

from robertslab.helper.timerHelper import timeDecorator
from robertslab.md.mdIO.sfileMD import SFileMD
from robertslab.md.mdIO.trajectoryToSFile import TrajectoryToSFile
from robertslab.test.md.testBase import TestBase

thisScriptDir = os.path.dirname(os.path.realpath(__file__))
testDataDir = os.path.join(thisScriptDir, '..', 'testData')

testDataName = '3d6z_aligned_vmd_wat_ion'
topPath = os.path.join(testDataDir, '.'.join((testDataName, 'pdb')))
trajPath = os.path.join(testDataDir, '.'.join((testDataName, 'dcd')))

class TrajectoryToSFileTestSet(object):
    @timeDecorator
    def test_frame_numbers(self):
        '''
        tests conversion of xyz data from a standard md trajectory file to an md SFile
        '''
        self.loadData(selection=None)

        frameNumArr = np.array([frame.frameNumber for frame in self.frames])
        intendedFrameNumArr = np.arange(len(self.frames))
        self.assertArraysEqual(intendedFrameNumArr, frameNumArr)

    @timeDecorator
    def test_frame_coordinates_all_atoms(self):
        '''
        tests conversion of xyz data from a standard md trajectory file to an md SFile
        '''
        self.loadData(selection=None)

        for mdtrajFrame,sfileFrame in zip(self.mdtraj.xyz, self.frames):
            self.assertArraysEqual(mdtrajFrame, sfileFrame.xyz)

    @timeDecorator
    def test_frame_coordinates_exclude_hydrogens(self):
        '''
        tests conversion of xyz data (sans hydrogens) from a standard md trajectory file to an md SFile
        '''
        self.loadData(selection='not type H', suffix='__no_H')

        for mdtrajFrame,sfileFrame in zip(self.mdtraj.xyz, self.frames):
            self.assertArraysEqual(mdtrajFrame, sfileFrame.xyz)

    @timeDecorator
    def test_frame_coordinates_protein_only(self):
        '''
        tests conversion of xyz data (protein heavy atoms only) from a standard md trajectory file to an md SFile
        '''
        self.loadData(selection='protein', suffix='__protein_only')

        for mdtrajFrame,sfileFrame in zip(self.mdtraj.xyz, self.frames):
            self.assertArraysEqual(mdtrajFrame, sfileFrame.xyz)

    @timeDecorator
    def test_frame_coordinates_CA_only(self):
        '''
        tests conversion of xyz data (alpha carbon atoms only) from a standard md trajectory file to an md SFile
        '''
        self.loadData(selection='name CA', suffix='__CA_only')

        for mdtrajFrame,sfileFrame in zip(self.mdtraj.xyz, self.frames):
            self.assertArraysEqual(mdtrajFrame, sfileFrame.xyz)

class TrajectoryToSFileTestBase(TestBase, TrajectoryToSFileTestSet):
    @classmethod
    def setUpClass(cls):
        cls.top = md.load(topPath).top

    def loadData(self, selection=None, suffix=None, repeats=30):
        # generate the md SFile
        self.trajectoryToSFile = TrajectoryToSFile(trajPath, self.top, repeats=repeats, selection=selection, sfileSuffix=suffix, writeTarget='cat')
        self.trajectoryToSFile.run()

        # load data directly from the original md trajectory file into an mdtraj trajectory
        self.mdtraj = md.load(trajPath, top=self.top)

        # slice the atoms in the mdtraj object to match what we put in our md SFile
        if selection is not None:
            self.mdtraj.atom_slice(self.mdtraj.top.select(selection), inplace=True)

        # repeat the mdtraj.xyz data, if appropriate
        if repeats > 0:
            self.mdtraj.xyz = np.tile(self.mdtraj.xyz, (repeats, 1, 1))

        # open our new md SFile and generate some Frame objects from it
        self.sfileMD = SFileMD(self.trajectoryToSFile.sfilePath)
        self.sfileMD.readFrames()
        self.frames = self.sfileMD.frames