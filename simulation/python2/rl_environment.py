'''Reinforcement learning (RL) environment.'''

# python
# scipy
from numpy.linalg import inv, norm
from numpy.random import rand, randint, randn
from numpy import arccos, array, cross, dot, eye, exp, hstack, ones, pi, vstack, where, zeros
# openrave
import openravepy

class RlEnvironment:

  def __init__(self, showViewer=True):
    '''Initializes openrave environment, etc.'''

    # Parameters

    self.projectDir = "/home/mgualti/mgualti/PickAndPlace/simulation"

    # Create openrave environment

    self.env = openravepy.Environment()
    if showViewer: self.env.SetViewer('qtcoin')
    self.showViewer = showViewer
    self.env.Load(self.projectDir + "/openrave/environment_1.xml")
    self.robot = self.env.GetRobots()[0]
    self.robot.SetDOFValues(array([0.034]))

    # don't want to be affected by gravity, since it is floating
    for link in self.robot.GetLinks():
      link.SetStatic(True)

    #self.physicsEngine = openravepy.RaveCreatePhysicsEngine(self.env, "ode")
    #self.env.SetPhysicsEngine(self.physicsEngine)
    #self.env.GetPhysicsEngine().SetGravity([0,0,-9.8])
    #self.env.GetPhysicsEngine().SetGravity([0,0,0])
    self.env.StopSimulation()

    tableObj = self.env.GetKinBody("table")
    self.tablePosition = tableObj.GetTransform()[0:3,3]
    self.tableExtents = tableObj.ComputeAABB().extents()

  def AssignPointsToObjects(self, rlAgent, objectHandles, viewCenter, viewKeepout, viewWorkspace):
    '''Takes a dual cloud for each object in objHandles and returns the full cloud with object
       indicies assigned to each point.
    '''

    # hide all of the objects
    objectPoses = []; hiddenPose = eye(4); hiddenPose[2,3] = -1.0
    for objectHandle in objectHandles:
      objectPoses.append(objectHandle.GetTransform())
      objectHandle.SetTransform(hiddenPose)

    # take image of each object
    objCloud = zeros((0,3)); objIdxs = zeros(0)
    for i, objectHandle in enumerate(objectHandles):
      objectHandle.SetTransform(objectPoses[i])
      cloud, viewPoints, viewPointIndices = rlAgent.GetDualCloud(
        viewCenter, viewKeepout, viewWorkspace)
      objCloud = vstack((objCloud, cloud))
      objIdxs = hstack((objIdxs, i*ones(cloud.shape[0])))
      objectHandle.SetTransform(hiddenPose)

    # return objects to their original positions
    for i, objectHandle in enumerate(objectHandles):
      objectHandle.SetTransform(objectPoses[i])

    return objCloud, objIdxs

  def GetObjectWithMaxGraspPoints(self, grasp, objHandles, objCloud, objCloudIdxs):
    '''Returns the object handle and object pose of the object which has the most points inside of
      the specified grasp.
    '''

    # parameters
    depth = norm(grasp.top-grasp.bottom)
    width = grasp.width / 2.0
    height = grasp.height

    # transform points into grasp frame
    C = vstack((objCloud.T, ones(objCloud.shape[0])))
    C = dot(inv(grasp.poses[1]), C)

    # determine which points are in the grasp
    mask = ((C[0,:] >= -height) & (C[0,:] <= height) & \
            (C[1,:] >= -width)  & (C[1,:] <= width)  & \
            (C[2,:] >= -depth)  & (C[2,:] <= 0)      )

    # find the object with the most points
    maxCount = 0; objHandle = None
    for i in xrange(len(objHandles)):
      count = sum(objCloudIdxs[mask]==i)
      if count > maxCount:
        maxCount = count
        objHandle = objHandles[i]

    if maxCount == 0:
      raw_input("Warning: Grasp has no points!")
      return None, None

    return objHandle, objHandle.GetTransform()

  def GetObjectPose(self, objectHandle):
    '''Returns the pose of an object.'''

    with self.env:
      return objectHandle.GetTransform()

  def GetState(self, rlAgent, grasp, place):
    '''Gets a very particular state vector corresponding to the current state of the environment.
      - Returns: s=(image,(isGrasped,isPlaced,placeArray))
    '''

    # determine if a grasp is selected
    isGrasped = not (grasp is None)
    if not isGrasped:
      grasp = rlAgent.emptyGrasp

    # determine state of hand placement pose
    isPlaced = False
    placeArray = zeros(len(rlAgent.placePoses))
    if place is not None:
      for idx, pose in enumerate(rlAgent.placePoses):
        d = sum(sum(abs(pose - place)))
        if d < 1e-5:
          placeArray[idx] = 1
          isPlaced = True
          break
      if not isPlaced:
        raise Exception("Input place pose does not match poses available to agent.")

    # assemble state vector
    s = (grasp, hstack((array([isGrasped, isPlaced]), placeArray)))
    return s

  def MoveObjectToPose(self, objectHandle, T):
    '''Moves the object to the pose specified by the 4x4 matrix T.'''

    with self.env:
      objectHandle.SetTransform(T)

  def PlaceObjectRandomOrientation(self, objectName, scale=1):
    '''Places the specified object in the center with a (semi-)random orientation.'''

    with self.env:

      # load object into environment
      self.env.Load(objectName, {'scalegeometry':str(scale)})

      idx1 = objectName.rfind("/")
      idx2 = objectName.rfind(".")
      shortObjectName = objectName[idx1+1:idx2]
      obj = self.env.GetKinBody(shortObjectName)

      # set physical properties
      l = obj.GetLinks()[0]
      l.SetMass(1)
      l.SetStatic(False)

      # choose a semi-random orientation
      orientOptions = [\
        [(1,0,0),  pi/2, (0, 1,0), 1], \
        [(1,0,0), -pi/2, (0,-1,0), 1],\
        [(0,1,0),  pi/2, (-1,0,0), 0],\
        [(0,1,0), -pi/2, ( 1,0,0), 0],\
        [(0,0,1),     0, (0,0, 1), 2],\
        [(0,0,1),    pi, (0,0,-1), 2]]
        # [axis, angle, newAxis, newAxisIndex]

      optionIndex = randint(0, len(orientOptions))
      orientOption= orientOptions[optionIndex]
      randomAngle = 2*pi * rand()

      R1 = openravepy.matrixFromAxisAngle(orientOption[0], orientOption[1])
      R2 = openravepy.matrixFromAxisAngle(orientOption[2], randomAngle)

      # adjust position to be above table
      boundingBox = obj.ComputeAABB().extents()

      T = eye(4)
      T[2, 3] = boundingBox[orientOption[3]] + self.tableExtents[2]

      # set object transform
      # print orientOption, randomAngle*(180/pi), boundingBox[orientOption[3]]
      objPose = dot(T, dot(R1, R2))
      obj.SetTransform(objPose)

    return obj, objPose

  def PlaceObjectSet(self, objectNames, objectScales):
    '''TODO'''

    # parameters
    positionMean = array([0, 0, 0.03]); positionSigma = array([0.08, 0.08, 0.02])

    with self.env:

      # load objects into environment
      objectHandles = []
      for i in xrange(len(objectNames)):
        self.env.Load(objectNames[i], {'scalegeometry':str(objectScales[i])})
        idx1 = objectNames[i].rfind("/")
        idx2 = objectNames[i].rfind(".")
        shortObjectName = objectNames[i][idx1+1:idx2]
        objectHandle = self.env.GetKinBody(shortObjectName)
        objectHandle.SetName(shortObjectName + "-" + str(i))
        objectHandles.append(objectHandle)

      # place objects randomly
      objectPoses = []
      for i, objectHandle in enumerate(objectHandles):
        # sample orientation
        q = randn(4)
        q = q / norm(q)
        objectPose = openravepy.matrixFromQuat(q)
        # sample position
        position = positionMean + positionSigma*randn(3)
        objectPose[0:3,3] = position
        # set object transform
        objectHandle.SetTransform(objectPose)
        objectPoses.append(objectPose)

    return objectHandles, objectPoses

  def RemoveObject(self, objectHandle):
    '''Removes the object in the environment by handle.'''

    with self.env:
      self.env.Remove(objectHandle)

  def RemoveObjectSet(self, objectHandles):
    '''Removes all of the objects in the list objectHandles.'''

    with self.env:
      for objectHandle in objectHandles:
        self.env.Remove(objectHandle)

  def RewardBinary(self, objectHandle, targetObjectAxis, maxAngleFromObjectAxis, maxObjectTableGap):
    '''Gives a unit reward if no collisions, object orientation is aligned, and object is not too high above table.'''

    if self.env.CheckCollision(objectHandle):
      print("Object in collision.")
      return 0

    bTo = objectHandle.GetTransform()
    angle = arccos(dot(targetObjectAxis, bTo[0:3,2]))
    if angle > maxAngleFromObjectAxis:
      print("Object angle of {} (deg) exceeds threshold".format(angle*(180/pi)))
      return 0

    e = objectHandle.ComputeAABB().extents()
    c = objectHandle.ComputeAABB().pos()
    boundingBox = vstack((
      array([c[0]+e[0],c[1]+e[1],c[2]+e[2]]), \
      array([c[0]+e[0],c[1]+e[1],c[2]-e[2]]), \
      array([c[0]-e[0],c[1]+e[1],c[2]+e[2]]), \
      array([c[0]-e[0],c[1]+e[1],c[2]-e[2]]), \
      array([c[0]+e[0],c[1]-e[1],c[2]+e[2]]), \
      array([c[0]+e[0],c[1]-e[1],c[2]-e[2]]), \
      array([c[0]-e[0],c[1]-e[1],c[2]+e[2]]), \
      array([c[0]-e[0],c[1]-e[1],c[2]-e[2]])  ))

    gapFromTable = min(boundingBox[:, 2]) - self.tableExtents[2]
    if gapFromTable > maxObjectTableGap:
      print("Gap from table of {} (m) exceeds threshold.".format(gapFromTable))
      return 0

    print("Good place!")
    return 1