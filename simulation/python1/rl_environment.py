'''Reinforcement learning (RL) environment.'''

# python
# scipy
from numpy.random import rand, randint
from numpy import arccos, array, dot, eye, hstack, pi, vstack, zeros
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

  def GetObjectPose(self, objectHandle):
    '''Returns the pose of an object.'''

    with self.env:
      return objectHandle.GetTransform()

  def GetInitialState(self, rlAgent):
    '''Gets the state of the environment before any actions have been taken.
      - Returns: s=(Ig/Ib,[flags,p]) where flags=(isSelected,isPlacedTempGood,isPlacedTempBad,
                 isPlacedFinalGood,isPlacedFinalBad) and where p=[placeTemp, placeFinal].
      - Returns selectedGrasp: There currently is no selected grasp.
    '''

    I = rlAgent.emptyImage
    f = array([0, 1, 0, 0, 0], dtype='float')
    p = zeros(len(rlAgent.placePoses), dtype='float')

    return (I, hstack((f, p))), None

  def GetPlaceQuality(self, objectHandle, targetObjectAxis, maxAngleFromObjectAxis, maxObjectTableGap):
    '''Returns binary variables relevant to place quality.'''

    inCollision = self.env.CheckCollision(objectHandle)
    if inCollision: print("Object in collision.")
    
    bTo = objectHandle.GetTransform()
    angle = dot(targetObjectAxis, bTo[0:3,2])
    angle = max(min(angle, 1.0), -1.0) # fix precision issues
    angle = arccos(angle)
    badOrientation = angle > maxAngleFromObjectAxis
    if badOrientation: print("Object angle of {} (deg) exceeds threshold".format(angle*(180/pi)))

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
    tooHigh = gapFromTable > maxObjectTableGap
    if tooHigh: print("Gap of {} is too high.".format(gapFromTable))
    
    return inCollision, badOrientation, tooHigh

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
        [(1,0,0),  pi/2, (0, 1,0), 1],\
        [(1,0,0), -pi/2, (0,-1,0), 1],\
        [(0,1,0),  pi/2, (-1,0,0), 0],\
        [(0,1,0), -pi/2, ( 1,0,0), 0],\
        [(0,0,1),     0, (0,0, 1), 2],
        [(1,0,0),    pi, (0,0,-1), 2]]
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

  def RemoveObject(self, objectHandle):
    '''Removes the object in the environment by handle.'''

    with self.env:
      self.env.Remove(objectHandle)

  def Transition(self, rlAgent, objectHandle, state, selectedGrasp, action, grasp, place,
    targetObjectAxis, maxAngleFromObjectAxis, maxObjectTableGap):
    '''Steps the simulator forward and determines the reward and next state.
      - Returns nextState: Tuple (I, [flags, placeVector]) of the next state for taking action in
                           the current state.
      - Returns nextSelectedGrasp: The grasp selected in the next state.
      - Returns reward: The reward for using action to get to nextState.
    '''

    if grasp is not None:
      # propagate select action
      I = grasp.image
      nextSelectedGrasp = grasp
      placeVector = state[1][5:]
      isSelected = True
      isPlacedTempGood = True
      isPlacedTempBad = False
      isPlacedFinalGood = False
      isPlacedFinalBad = False
      # reward in this case is always 0
      reward = 0.0
    else:
      # propagate simulation
      rlAgent.MoveHandToPose(place)
      rlAgent.MoveObjectToHandAtGrasp(selectedGrasp, objectHandle)
      collide, orient, gap = self.GetPlaceQuality(
        objectHandle, targetObjectAxis, maxAngleFromObjectAxis, maxObjectTableGap)
      okTemp = not collide and not gap
      okFinal = okTemp and not orient
      # propagate state vector
      I = state[0]
      nextSelectedGrasp = None
      placeIndex = rlAgent.GetPlaceIndex(place)
      isFinalPlace = rlAgent.isFinalPlacePose[placeIndex]
      placeVector = zeros(len(rlAgent.placePoses))
      placeVector[placeIndex] = 1
      isSelected = False
      isPlacedTempGood = not isFinalPlace and okTemp
      isPlacedTempBad = not isFinalPlace and not okTemp
      isPlacedFinalGood = isFinalPlace and okFinal
      isPlacedFinalBad = isFinalPlace and not okFinal
      # check reward
      reward = 1.0 if isPlacedFinalGood else 0.0      
      if isPlacedTempGood: print("Good temp place!")
      if isPlacedFinalGood: print("Good final place!")      

    # assemble vector for next state
    flags = array([isSelected, isPlacedTempGood, isPlacedTempBad, isPlacedFinalGood,
      isPlacedFinalBad], dtype='float')
    nextState = (I, hstack((flags, placeVector)))

    return nextState, nextSelectedGrasp, reward