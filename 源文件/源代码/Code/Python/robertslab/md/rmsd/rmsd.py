import mdtraj
import numpy as np

from robertslab.md.mdIO import Frame

class RMSD(object):
    @staticmethod
    def _simpleRMSD(arr1, arr2):
        return np.sqrt(((arr1 - arr2)**2).sum()/float(arr1.shape[1]))

    @staticmethod
    def _qcpRMSD(framesXYZ, refXYZ, referenceFrameIndex=0, parallel=False):
        framesG = mdtraj._rmsd._center_inplace_atom_major(framesXYZ)
        refG = mdtraj._rmsd._center_inplace_atom_major(refXYZ)

        return mdtraj._rmsd.getMultipleRMSDs_atom_major(refXYZ, framesXYZ, refG, framesG, referenceFrameIndex, parallel=parallel)

    @staticmethod
    def _qcpRMSD_ManualTrace(framesXYZ, refXYZ, framesAtomIndices=None, refAtomIndices=None, referenceFrameIndex=0, parallel=False):
        if framesAtomIndices is None:
            framesAtomIndices = slice(None)

        if refAtomIndices is None:
            refAtomIndices = framesAtomIndices

        if not isinstance(refAtomIndices, slice) and (
                    len(refAtomIndices) != len(framesAtomIndices)):
            raise ValueError("Number of atoms must be consistent!")        
        
        nFrames = framesXYZ.shape[0]
        alignFramesXYZ = np.asarray(framesXYZ[:, framesAtomIndices, :], order='c')
        displaceFramesXYZ = np.asarray(framesXYZ, order='c')
        alignRefXYZ = np.array(refXYZ[referenceFrameIndex, refAtomIndices, :],
                                 copy=True, order='c').reshape(1, -1, 3)

        offset = np.mean(alignFramesXYZ, axis=1, dtype=np.float64).reshape(nFrames, 1, 3)
        alignFramesXYZ -= offset
        if alignFramesXYZ.ctypes.data != displaceFramesXYZ.ctypes.data:
            # when framesAtomIndices is None, these two arrays alias the same memory
            # so we only need to do the centering once
            displaceFramesXYZ -= offset

        refOffset = alignRefXYZ[0].astype('float64').mean(0)
        alignRefXYZ[0] -= refOffset

        framesG = np.einsum('ijk,ijk->i', alignFramesXYZ, alignFramesXYZ)
        refG = np.einsum('ijk,ijk->i', alignRefXYZ , alignRefXYZ)

        mdtraj._rmsd.superpose_atom_major(
            alignRefXYZ, alignFramesXYZ, refG, framesG, displaceFramesXYZ,
            0, parallel=True)

        return mdtraj._rmsd.getMultipleRMSDs_atom_major(refXYZ, framesXYZ, refG, framesG, referenceFrameIndex, parallel=parallel)

    def __init__(self, frames, refFrame=None, align=False, outPath=None, selection=None, sliceFrames=False, sliceRefFrame=True, top=None):
        """
        sliceFrames: if True, slice the frames as well as the refFrame
        """
        self.align = align
        self.frames = frames
        self.outPath = outPath
        self.sliceFrames = sliceFrames
        self.sliceRefFrame = sliceRefFrame
        self.sliceRef = True
        self.rmsdList = []

        if refFrame is None:
            # if refFrame is not specified, use the 0th frame as the refFrame
            self.refFrame = frames.pop(0)
            # in this case be sure to slice the "refFrame" the same as the other frames
            self.sliceRefFrame = self.sliceFrames
        else:
            if isinstance(refFrame, Frame):
                self.refFrame = refFrame
            else:
                self.refFrame = Frame(refFrame)

        if isinstance(selection, str):
            self.atomIndices = top.select(selection)
        else:
            self.atomIndices = selection

        self.initRMSDFunc()

    def initRMSDFunc(self):
        if self.align:
            self.rmsdFunc = RMSD._qcpRMSD
        else:
            self.rmsdFunc = RMSD._simpleRMSD

    def calculate(self):
        # ensure that refFrame is 3D, even if only 1 deep
        refShapeNew = (-1,) + self.refFrame.xyz.shape[-2:]
        refSlice = self.refFrame.xyz.reshape(refShapeNew)
        if self.atomIndices is not None and self.sliceRefFrame:
            refSlice = np.asarray(refSlice[:,self.atomIndices,:], order='c')

        for i,frame in enumerate(self.frames):
            # ensure that frame is 3D, even if only 1 deep
            frameShapeNew = (-1,) + frame.xyz.shape[-2:]
            frameSlice = frame.xyz.reshape(frameShapeNew)
            if self.atomIndices is not None and self.sliceFrames:
                frameSlice = np.asarray(frameSlice[:,self.atomIndices,:], order='c')

            self.rmsdList.append(self.rmsdFunc(frameSlice, refSlice))

        if len(self.frames) == 1:
            return self.rmsdList[0][0]

        return self.rmsdList

    def write(self):
        rmsdIO = RMSDIO(path=self.outPath, rmsdList=self.rmsdList)
        rmsdIO.write()