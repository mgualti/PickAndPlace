'''Provides a class for representing data returned from the grasp detector.'''

from copy import copy
from numpy import cross, eye # scipy

class Grasp():


  def __init__(self, approach, axis, top, bottom, height, width, offsets,
    binormal=None, score=None, image=None):
    '''Creates a Grasp object with everything needed.'''

    self.approach = approach
    self.axis = axis
    self.top = top
    self.bottom = bottom
    self.height = height
    self.width = width
    self.offsets = offsets

    self.binormal = cross(approach, axis) if binormal is None else binormal
    self.center = 0.5*bottom + 0.5*top
    self.fingerNormals = (self.binormal, -self.binormal)
    self.score = 0 if score is None else score
    self.image = image

    # colums are ordered according to right_gripper frame
    self.poses = []
    for offset in offsets:
      T = eye(4)
      T[0:3,0] = axis
      T[0:3,1] = binormal
      T[0:3,2] = approach
      T[0:3,3] = top - offset*approach
      self.poses.append(T)


  def Flip(self):
    '''Creates a new grasp flipped 180 degrees about the approach direction.'''

    approach = copy(self.approach)
    axis = -copy(self.axis)
    top = copy(self.top)
    bottom = copy(self.bottom)
    height = self.height
    width = self.width
    offsets = copy(self.offsets)
    binormal = -copy(self.binormal)
    score = self.score
    image = copy(self.image)

    return Grasp(approach, axis, top, bottom, height, width, offsets, binormal, score, image)
