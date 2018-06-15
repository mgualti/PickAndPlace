'''A class for managing 3DNet objects.'''

# python
import os
# scipy
from numpy.random import rand, randint

class ThreeDNet:

  def __init__(self):
    '''TODO'''

    self.dir = "/home/mgualti/Data/3DNet/Cat10_ModelDatabase"

    # 3D Net objects all have height of 1m
    self.classes     = ["bottle", "bottle_test", "bottle_train", "mug", "mug_test", "mug_train", "tetra_pak", "tetra_pak_test", "tetra_pak_train"]
    self.scales      = [    0.20,          0.20,           0.20,  0.12,       0.12,        0.12,        0.15,             0.15,              0.15]
    self.minScales   = [    0.10,          0.10,           0.10,  0.06,       0.06,        0.06,        0.10,             0.10,              0.10]
    self.maxSales    = [    0.20,          0.20,           0.20,  0.12,       0.12,        0.12,        0.20,             0.20,              0.20]

  def GetObjectByName(self, objectClass, objectName):
    '''Gets the full object name given the short name and object class.'''

    if objectClass not in self.classes:
      raise Exception("Unrecognized category.")
    classIndex = self.classes.index(objectClass)

    fullObjectName = self.dir + "/" + objectClass + "/" + objectName + ".ply"

    if not os.path.isfile(fullObjectName):
      raise Exception("Unrecognized object name, {}.".format(fullObjectName))

    return fullObjectName, self.scales[classIndex]

  def GetObjectNames(self, objectClass):
    '''Gets the names of all objects in the specified category.'''

    if objectClass not in self.classes:
      raise Exception("Unrecognized category.")

    dirFileNames = os.listdir(self.dir + "/" + objectClass)

    meshFileNames = []
    for name in dirFileNames:
      if len(name) > 3 and name[-4:] == ".ply":
        meshFileNames.append(name[:-4])

    return meshFileNames

  def GetRandomObjectSet(self, objectClass, nObjects, randomScale):
    '''Gets a list of objects and corresponding scales from the requested object class.'''

    if objectClass not in self.classes:
      raise Exception("Unrecognized category.")
    classIndex = self.classes.index(objectClass)

    dirFileNames = os.listdir(self.dir + "/" + objectClass)

    allMeshFileNames = []
    for name in dirFileNames:
      if len(name) > 3 and name[-4:] == ".ply":
        allMeshFileNames.append(name)

    meshFileNames = []; meshScales = []
    for obj in xrange(nObjects):
      meshIdx = randint(len(allMeshFileNames)-1)
      meshFileNames.append(self.dir + "/" + objectClass + "/" + allMeshFileNames[meshIdx])
      if randomScale:
        minScale = self.minScales[classIndex]
        maxScale = self.maxSales[classIndex]
        meshScales.append(minScale + (maxScale-minScale)*rand())
      else:
        meshScales.append(self.scales[classIndex])

    return meshFileNames, meshScales

  def GetRandomObjectFromClass(self, objectClass, randomScale):
    '''Gets the full file name and scale of a random object in the specified category.
      - Input objectClass: The name of the 3D Net class folder to choose objects from.
      - Input randomScale: If True, scale is selected uniformly at random from the pre-specied range
        for the object class. If False, scale is fixed for the object class.
      - Returns objectName: Full file name of the object randomly chosen.
      - Returns scale: Size of the object.
    '''

    if objectClass not in self.classes:
      raise Exception("Unrecognized category.")
    classIndex = self.classes.index(objectClass)

    dirFileNames = os.listdir(self.dir + "/" + objectClass)

    meshFileNames = []
    for name in dirFileNames:
      if len(name) > 3 and name[-4:] == ".ply":
        meshFileNames.append(name)

    meshIdx = randint(len(meshFileNames)-1)
    meshFileName = self.dir + "/" + objectClass + "/" + meshFileNames[meshIdx]

    if randomScale:
      minScale = self.minScales[classIndex]
      maxScale = self.maxSales[classIndex]
      scale = minScale + (maxScale-minScale)*rand()
    else:
      scale = self.scales[classIndex]

    return meshFileName, scale