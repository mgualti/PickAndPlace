'''Reinforcement learning (RL) agent and related utility functions.'''

# python
import time
from copy import copy
# scipy
from scipy.io import savemat
from numpy.linalg import inv, norm
from numpy.random import choice, permutation, randint, rand
from numpy import arccos, array, cos, cross, dot, eye, linspace, ones, pi, sin, vstack, zeros
# openrave
import openravepy
# caffe
import h5py
import caffe
# self
import point_cloud
from grasp import Grasp
from grasp_proxy_matlab import GraspProxyMatlab

# AGENT ============================================================================================

class RlAgent:

  def __init__(self, rlEnvironment):
    '''Initializes agent in the given environment.'''

    # parameters

    self.caffeDir = "/home/mgualti/mgualti/PickAndPlace/simulation/caffe/"
    self.caffeWeightsFilePrefix = self.caffeDir + "image_iter_"
    self.caffeModelFileName = self.caffeDir + "networkDeploy-image.prototxt"
    self.caffeSolverFileName = self.caffeDir + "solver-image.prototxt"
    self.caffeTrainFileName = self.caffeDir + "train.h5"
    self.caffeTestFileName = self.caffeDir + "test.h5"

    self.emptyGraspImage = zeros((12,60,60))
    self.emptyGrasp = Grasp(array([0,0,0]), array([0,0,0]), array([1,0,0]), array([0,1,0]), 0, [],
      array([0,0,1]), 0, self.emptyGraspImage)

    # simple assignments
    self.rlEnv = rlEnvironment
    self.env = self.rlEnv.env
    self.robot = self.rlEnv.robot
    self.plotCloudHandle = None
    self.plotGraspsHandle = None
    self.nViewSamples = 1000

    # get sensor from openrave
    self.sensor = self.env.GetSensors()[0]
    self.sTh = dot(inv(self.sensor.GetTransform()), self.robot.GetTransform())

    # possible sensor "up" choices
    theta = linspace(0, 2*pi, 20)
    x = cos(theta); y = sin(theta)

    self.sensorUpChoices = []
    for i in xrange(len(theta)):
      self.sensorUpChoices.append(array([x[i], y[i], 0]))

    # specify possible place configurations

    # In this case the configurations are relative to the sensor, which happens to be near the
    # center of the closing region of the hand. It's easier to set these positions relative to the
    # center of the closing region of the hand.

    placeOrientations = [ \
      array([[-1,0,0], [0,1, 0], [0,0,-1]]).T, \
      array([[ 0,0,1], [1,0, 0], [0,1, 0]]).T]
    sideOrientations = [0, 1]

    placePositions = zeros((21, 3))
    placePositions[:, 0] = -1.0
    placePositions[:, 2] = linspace(0.05, 0.25, 21)

    self.placePoses = []; self.isSidePlace = []
    for i in xrange(len(placeOrientations)):
      for j in xrange(len(placePositions)):
        P = eye(4)
        P[0:3,0:3] = placeOrientations[i]
        P[0:3,3] = placePositions[j]
        P = dot(P, self.sTh)
        self.MoveHandToPose(P)
        if not self.env.CheckCollision(self.robot):
          self.placePoses.append(P)
          self.isSidePlace.append(sideOrientations[i])

    print("There are {} place poses.".format(len(self.placePoses)))

    # initialize grasp proxy
    self.graspProxy = GraspProxyMatlab()

    # initialize caffe
    caffe.set_device(0)
    caffe.set_mode_gpu()
    self.caffeNet = caffe.Net(self.caffeModelFileName, caffe.TEST)
    self.caffeFirstTrain = True

  def DetectGrasps(self, cloud, viewPoints, viewPointIndices, nSamples, scoreThresh, detectMode):
    '''Detects grasps in the point cloud.
      - Input cloud: nx3 cloud of points in 3d.
      - Input viewPoints: nx3 list of view points from which cloud was taken.
      - Input viewPointIndices: 0-based index of *start* of range from corresponding viewpoint.
      - Input nSamples: number of points in cloud to sample for grasps.
      - Input scoreThresh: Grasp detector score threshold (only used for detectMode=1).
      - Input detectMode: 0 for sampling candidates only, 1 for using trained CNN to score grasps.
      - Returns grasps: List of grasps found by the detector.
    '''

    # setup
    offsets = [0.04, 0.0]

    viewPointIndices = viewPointIndices + 1
    tableUpAxis = 3
    tablePosition = copy(self.rlEnv.tablePosition)
    tablePosition[tableUpAxis-1] += 0.002
    tableFingerLength = 0.05
    minWidth = 0.002; maxWidth = 0.100

    # detect
    if detectMode == 0:
      grasps = self.graspProxy.SampleGrasps(cloud, viewPoints, viewPointIndices, nSamples, minWidth,
        maxWidth, tablePosition, tableUpAxis, tableFingerLength, offsets)
    elif detectMode == 1:
      grasps = self.graspProxy.DetectGrasps(cloud, viewPoints, viewPointIndices, nSamples,
        scoreThresh, minWidth, maxWidth, tablePosition, tableUpAxis, tableFingerLength, offsets)
    else:
      raise Exception("Unrecognized grasp detection mode {}.".format(detectMode))

    return grasps

  def DownsampleAndLabelData(self, D, batchSize):
    '''Samples data to a batch size, uniformly at random, and labels the data with the current value function approximation.'''

    if len(D) <= batchSize:
      return self.LabelData(D)

    idxs = choice(len(D), batchSize, replace=False)
    batchD = [D[i] for i in idxs]
    return self.LabelData(batchD)

  def FilterGraspsWithNoPoints(self, grasps, cloud):
    '''Gets rid of any grasps that do not have points between the fingers.'''

    keepGrasps = []
    for grasp in grasps:

      # parameters
      depth = norm(grasp.top-grasp.bottom)
      width = grasp.width / 2.0
      height = grasp.height

      # transform points into grasp frame
      C = vstack((cloud.T, ones(cloud.shape[0])))
      C = dot(inv(grasp.poses[1]), C)

      # determine which points are in the grasp
      mask = ((C[0,:] >= -height) & (C[0,:] <= height) & \
              (C[1,:] >= -width)  & (C[1,:] <= width)  & \
              (C[2,:] >= -depth)  & (C[2,:] <= 0)      )

      if sum(mask) > 0:
        keepGrasps.append(grasp)

    return keepGrasps

  def GetAction(self, state, grasps, epsilon):
    '''Chooses the next action from (grasp, place, end) with an epsilon-greedy policy.
      - Warning: Cannot place until grasped.
    '''

    # 1. Initialization
    grasp = None; place = None
    isGrasped = state[1][0]

    # 2. Epsilon
    if rand() <= epsilon:
      # first, choose an action type
      nChoices = 2 if isGrasped else 1
      choice = randint(nChoices)
      # now choose details of action
      if choice == 0:
        grasp = self.GetRandomGrasp(grasps)
      else:
        place = self.GetRandomPlacePose()
      # finished
      return grasp, place

    # 3. Greedy

    # evaluate grasps
    bestValue = float("-Inf")
    for g in grasps:
      s = self.rlEnv.GetState(self, g, None)
      self.caffeNet.blobs["imagestate"].data[0] = s[0].image
      self.caffeNet.blobs["state"].data[0] = s[1]
      self.caffeNet.forward()
      value = self.caffeNet.blobs["ip3"].data[0,0]
      if value > bestValue:
        grasp = g
        bestValue = value
        place = None

    # evaluate places
    if isGrasped:
      for p in self.placePoses:
        s = self.rlEnv.GetState(self, state[0], p)
        self.caffeNet.blobs["imagestate"].data[0] = s[0].image
        self.caffeNet.blobs["state"].data[0] = s[1]
        self.caffeNet.forward()
        value = self.caffeNet.blobs["ip3"].data[0,0]
        if value > bestValue:
          place = p
          bestValue = value
          grasp = None

    return grasp, place

  def GetCloud(self, workspace=None):
    '''Agent gets point cloud from its sensor from the current position.'''

    self.StartSensor()
    self.env.StepSimulation(0.001)

    data = self.sensor.GetSensorData(openravepy.Sensor.Type.Laser)
    cloud = data.ranges + data.positions[0]

    self.StopSensor()

    if workspace is not None:
      cloud = point_cloud.FilterWorkspace(cloud, workspace)

    return cloud

  def GetDualCloud(self, viewCenter, viewKeepout, workspace):
    '''Gets a point cloud combined form two views, 45 degrees above table and 90 degrees apart.
      - Returns cloud: nx3 combined point cloud.
      - Returns viewPoints:  transforms of sensor pose from where cloud was taken.
      - Returns viewPointIndices: Starting index into the points indicating which pose the points belong.
    '''

    poses = self.GetDualViewPoses(viewCenter, viewKeepout)

    C = []
    cloud = []
    viewPoints = []
    viewPointIndices = []

    for pose in poses:
      viewPointIndices.append(len(C))
      self.MoveSensorToPose(pose)
      C = self.GetCloud(workspace)
      cloud.append(C)
      viewPoints.append(pose[0:3,3].T)

    cloud = vstack(cloud)
    viewPoints = vstack(viewPoints)
    viewPointIndices = array(viewPointIndices)

    return cloud, viewPoints, viewPointIndices

  def GetDualViewPoses(self, viewCenter, viewKeepout):
    '''Gets standard dual poses, 45 degress from table and 90 degrees apart.'''

    p1 = viewCenter + viewKeepout*array([0, -cos(45*(pi/180)), sin(45*(pi/180))])
    p2 = viewCenter + viewKeepout*array([0,  cos(45*(pi/180)), sin(45*(pi/180))])

    upChoice = array([1,0,0])
    viewPose1 = GeneratePoseGivenUp(p1, viewCenter, upChoice)
    viewPose2 = GeneratePoseGivenUp(p2, viewCenter, upChoice)

    return viewPose1, viewPose2

  def GetGrasp(self, grasps, epsilon=0):
    '''Returns the best grasp according to V(s) or a random grasp with probability epsilon.'''

    # act randomly
    doRandomAction = rand() <= epsilon
    if doRandomAction:
      return self.GetRandomGrasp(grasps)

    # act according to value function
    bestGrasps = None; bestValue = float('-Inf')
    for grasp in grasps:
      s = self.rlEnv.GetState(self, grasp, None)
      self.caffeNet.blobs["imagestate"].data[0] = s[0].image
      self.caffeNet.blobs["state"].data[0] = s[1]
      self.caffeNet.forward()
      value = self.caffeNet.blobs["ip3"].data[0,0]
      if value > bestValue:
        bestGrasps = [grasp]
        bestValue = value
      elif value == bestValue:
        bestGrasps.append(grasp)

    # break ties randomly
    print("Best grasp value: {}.".format(bestValue))
    return bestGrasps[randint(len(bestGrasps))]

  def GetGraspCylinder(self, grasps, cloud, viewPoints, viewPointIndices, kClusters, maxGraspToCylinderAngle):
    '''Uses cylinder fitting and heuristics to select the grasp.'''

    # Fit cylinder.

    viewPointIndices = viewPointIndices + 1 # Matlab 1-indexed
    cylinder = self.graspProxy.FitCylinder(cloud, viewPoints, viewPointIndices, kClusters)

    # Find grasps aligned to cylinder.

    alignedGrasps = []
    for grasp in grasps:
      if arccos(min(1.0, abs(dot(grasp.axis, cylinder.axis)))) <= maxGraspToCylinderAngle:
        alignedGrasps.append(grasp)
    if len(alignedGrasps) == 0:
      alignedGrasps = grasps

    # Of the aligned grasps, choose the closest one.

    minDistance = float('inf'); minDistanceGrasp = None
    for grasp in grasps:
      d = norm(cylinder.center-grasp.center)
      if d < minDistance:
        minDistance = d
        minDistanceGrasp = grasp
    grasp = minDistanceGrasp

    if cloud.size == 0: return grasp, cylinder

    # Decide if the grasp should be flipped.

    # transform cloud into cylinder frame
    bTc = eye(4)
    cylinderApproach = array([cylinder.axis[1], -cylinder.axis[0], 0])
    cylinderApproach = cylinderApproach / norm(cylinderApproach) \
      if norm(cylinderApproach) > 0.001 else array([1,0,0])
    bTc[0:3, 0] = cylinderApproach
    bTc[0:3, 1] = cross(cylinder.axis, cylinderApproach)
    bTc[0:3, 2] = cylinder.axis
    bTc[0:3, 3] = cylinder.center
    C = vstack((cloud.T, ones(cloud.shape[0])))
    C = dot(inv(bTc), C)

    # keep points within cylinder caps
    C = C[:, (C[2,:] >= -cylinder.height) & (C[2,:] <= cylinder.height)]
    if C.size == 0: return grasp, cylinder

    # keep points close to cylinder axis
    distSquared = C[0, :]**2 + C[1, :]**2
    C = C[:, distSquared <= (cylinder.radius+0.01)**2]
    if C.size == 0: return grasp, cylinder

    # count the number of points in upper and lower halves of cylinder
    upperCount = sum(C[2,:] >= 0)
    lowerCount = sum(C[2,:] <= 0)

    # flip if the object is top-heavy
    cylinderAxis = cylinder.axis if lowerCount > upperCount else -cylinder.axis
    if dot(grasp.axis, cylinderAxis) < 0:
      grasp = grasp.Flip()

    return grasp, cylinder

  def GetPlacePose(self, grasp, epsilon=0):
    '''Returns the best place pose according to V(s) or a random pose with probability epsilon.'''

    # act randomly
    doRandomAction = rand() <= epsilon
    if doRandomAction:
      return self.GetRandomPlacePose()

    # act according to value function
    bestPoses = None; bestValue = float('-Inf')
    for i, pose in enumerate(self.placePoses):
      s = self.rlEnv.GetState(self, grasp, pose)
      self.caffeNet.blobs["imagestate"].data[0] = s[0].image
      self.caffeNet.blobs["state"].data[0] = s[1]
      self.caffeNet.forward()
      value = self.caffeNet.blobs["ip3"].data[0,0]
      #print("a={}, v={}".format(i+1, value))
      if value > bestValue:
        bestPoses = [pose]
        bestValue = value
      elif value == bestValue:
        bestPoses.append(pose)

    # break ties randomly
    print("Best place value: {}".format(bestValue))
    return bestPoses[randint(len(bestPoses))]

  def GetPlacePoseCylinder(self, grasp, cylinder, maxTableGap):
    '''Gets a place pose that should clear the cylinder from the table.'''

    targetHeight = (cylinder.height / 2.0) + (maxTableGap / 2.0)

    minDist = float('inf'); minDistPose = None
    for i, pose in enumerate(self.placePoses):
      if self.isSidePlace[i]:
        d = abs(targetHeight-pose[2,3])
        if d < minDist:
          minDist = d
          minDistPose = pose

    return minDistPose

  def GetRandomGrasp(self, grasps):
    '''Selects a random grasp from the provided list.'''

    if len(grasps) == 0:
      return None

    return grasps[randint(len(grasps))]

  def GetRandomPlacePose(self):
    '''Selects a random pose for the hand from the discrete list of allowed place poses.'''

    return self.placePoses[randint(len(self.placePoses))]

  def GetStandardViewPose(self, viewCenter, viewKeepout):
    '''Gets a standard pose for the viewer directly above viewCenter in the z direction.'''

    viewPose = eye(4)
    viewPose[0,0] = -1
    viewPose[1,1] = 1
    viewPose[2,2] = -1
    viewPose[0:3,3] = viewCenter
    viewPose[2,3] += viewKeepout
    return viewPose

  def LabelData(self, D, gamma=1.0):
    '''Given a database of (state, nextState, reward), use the network to compute one-step lookahead values.'''

    Dl = []
    for d in D:

      s = d[0]; ss = d[1]; r = d[2]

      if ss is None:
        vss = 0 # terminal state -- known to have 0 value
      else:
        self.caffeNet.blobs["imagestate"].data[0] = ss[0].image
        self.caffeNet.blobs["state"].data[0] = ss[1]
        self.caffeNet.forward()
        vss = self.caffeNet.blobs["ip3"].data[0,0]

      vs = r + gamma*vss # Bellman update
      Dl.append((s, vs))

    return Dl

  def LoadNetworkWeights(self, weightsFileName):
    '''Loads the network weights from the specified file name.'''

    self.caffeNet = caffe.Net(self.caffeModelFileName, caffe.TEST, weights=weightsFileName)
    print("Loaded file " + weightsFileName + " successfully.")

  def MoveHandToPose(self, T):
    '''Moves the hand of the robot to the specified pose.'''

    self.robot.SetTransform(T)
    self.env.UpdatePublishedBodies()

    return True # didMove

  def MoveObjectToHandAtGrasp(self, grasp, objectHandle):
    '''Aligns the grasp on the object to the current hand position and moves the object there.'''

    bTg = grasp.poses[0]
    bTo = objectHandle.GetTransform()
    bTs = self.sensor.GetTransform()

    gTo = dot(inv(bTg), bTo)
    bTo_new = dot(bTs, gTo)

    objectHandle.SetTransform(bTo_new)

  def MoveSensorToPose(self, T):
    '''Moves the hand of the robot to the specified pose.'''

    self.robot.SetTransform(dot(T, self.sTh))
    self.env.UpdatePublishedBodies()

    return True # didMove

  def PerformAction(self, state, grasp, place, objHandle):
    '''Performs the action given by self.GetAction and returns the next state and reward.'''

    isPlaced = state[1][1]

    if isPlaced:
      raise Exception("The object has already been placed when an action is being performed.")

    if grasp is not None and place is None:
      nextState = self.rlEnv.GetState(self, grasp, None)
    elif place is not None and grasp is None:
      self.MoveHandToPose(place)
      self.MoveObjectToHandAtGrasp(state[0], objHandle)
      nextState = self.rlEnv.GetState(self, state[0], place)
    else:
      raise Exception("Only one of grasp, place, or endEpisode actions must be selected.")

    return nextState, self.rlEnv.RewardMultiDetect(self, state, nextState, objHandle)

  def PlotCloud(self, cloud):
    '''Plots a cloud in the environment.'''

    if not self.rlEnv.showViewer:
      return

    if self.plotCloudHandle is not None:
      self.UnplotCloud()

    self.plotCloudHandle = self.env.plot3(\
      points=cloud, pointsize=0.001, colors=zeros(cloud.shape), drawstyle=1)

  def PlotGrasps(self, grasps):
    '''Visualizes grasps in openrave viewer.'''

    if not self.rlEnv.showViewer:
      return

    if self.plotGraspsHandle is not None:
      self.UnplotGrasps()

    if len(grasps) == 0:
      return

    graspLength = 0.06
    graspColor = [0,0,1]

    lineList = []; colorList = []
    for grasp in grasps:

      c = grasp.bottom
      a = c - graspLength*grasp.approach
      l = c - 0.5*grasp.width*grasp.binormal
      r = c + 0.5*grasp.width*grasp.binormal
      lEnd = l + graspLength*grasp.approach
      rEnd = r + graspLength*grasp.approach

      lineList.append(c); lineList.append(a)
      lineList.append(l); lineList.append(r)
      lineList.append(l); lineList.append(lEnd)
      lineList.append(r); lineList.append(rEnd)

    for i in xrange(len(lineList)):
      colorList.append(graspColor)

    self.plotGraspsHandle = self.env.drawlinelist(\
      points=array(lineList), linewidth=3.0, colors=array(colorList))

  def PruneDatabase(self, replayDatabase, maxEntries):
    '''Removes oldest items in the database until the size is no more than maxEntries.'''

    if len(replayDatabase) <= maxEntries:
      return replayDatabase

    return replayDatabase[len(replayDatabase)-maxEntries:]

  def SaveCloud(self, cloud, viewPoints, viewPointIndices, fileName):
    '''Saves point cloud information for testing in Matlab.'''

    viewPointIndices = viewPointIndices + 1 # matlab is 1-indexed
    viewPoints = viewPoints.T
    cloud = cloud.T
    data = {"cloud":cloud, "viewPoints":viewPoints, "viewPointIndices":viewPointIndices}
    savemat(fileName, data)

  def StartSensor(self):
    '''Starts the sensor in openrave, displaying yellow haze.'''

    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.PowerOn)
    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.RenderDataOn)

  def StopSensor(self):
    '''Disables the sensor in openrave, removing the yellow haze.'''

    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.PowerOff)
    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.RenderDataOff)

  def Train(self, Dl, recordLoss=True, stepSize=100, nIterations=5000):
    '''Trains the network on the provided data labels.
      - Input Dl: List of tuples with (state,value).
      - Input recordLoss: If true, saves train and test return values (takes longer).
      - Input stepSize: Records loss values this often.
      - Input nIterations: Number of training iterations to run.
      - Returns: train loss, test loss.
    '''

    if len(Dl) == 0:
      return

    # 1. Load data

    # split data into train/test
    nTest = int(len(Dl)/4.0)
    nTrain = len(Dl) - nTest

    # shuffle data
    pdxs = permutation(len(Dl))
    idxs = pdxs[0:nTrain]
    jdxs = pdxs[nTrain:]

    sampleImage = Dl[0][0][0].image
    nx = sampleImage.shape[0]
    ny = sampleImage.shape[1]
    nz = sampleImage.shape[2]

    I = zeros((nTrain, nx, ny, nz))
    S = zeros((nTrain, len(Dl[0][0][1])))
    L = zeros(nTrain)
    for i, idx in enumerate(idxs):
      I[i, :, :, :] = Dl[idx][0][0].image
      S[i, :] = Dl[idx][0][1]
      L[i] = Dl[idx][1]

    with h5py.File(self.caffeTrainFileName, 'w') as fileHandle:
      fileHandle.create_dataset("imagestate", data=I)
      fileHandle.create_dataset("state", data=S)
      fileHandle.create_dataset("label", data=L)

    I = zeros((nTest, nx, ny, nz))
    S = zeros((nTest, len(Dl[0][0][1])))
    L = zeros(nTest)
    for j, jdx in enumerate(jdxs):
      I[j, :, :, :] = Dl[jdx][0][0].image
      S[j, :] = Dl[jdx][0][1]
      L[j] = Dl[jdx][1]

    with h5py.File(self.caffeTestFileName, 'w') as fileHandle:
      fileHandle.create_dataset("imagestate", data=I)
      fileHandle.create_dataset("state", data=S)
      fileHandle.create_dataset("label", data=L)

    # 2. Optimize

    weightsFileName = self.caffeWeightsFilePrefix + str(nIterations) + ".caffemodel"
    solver = caffe.SGDSolver(self.caffeSolverFileName)

    if self.caffeFirstTrain:
      self.caffeFirstTrain = False
    else:
      solver.net.copy_from(weightsFileName)

    trainLoss = []; testLoss = []

    if recordLoss:

      for iteration in xrange(int(nIterations/stepSize)):
        solver.step(stepSize)
        loss = float(solver.net.blobs["loss"].data)
        trainLoss.append(loss)

        loss = 0
        for testIteration in xrange(stepSize):
          solver.test_nets[0].forward()
          loss += float(solver.test_nets[0].blobs['loss'].data)
        loss /= stepSize
        testLoss.append(loss)

    else:

      solver.step(nIterations)

    self.caffeNet = caffe.Net(self.caffeModelFileName, caffe.TEST, weights=weightsFileName)

    return trainLoss, testLoss

  def UnplotCloud(self):
    '''Removes a cloud from the environment.'''

    if not self.rlEnv.showViewer:
      return

    if self.plotCloudHandle is not None:
      self.plotCloudHandle.Close()
      self.plotCloudHandle = None

  def UnplotGrasps(self):
    '''Removes any grasps drawn in the environment.'''

    if not self.rlEnv.showViewer:
      return

    if self.plotGraspsHandle is not None:
      self.plotGraspsHandle.Close()
      self.plotGraspsHandle = None

# UTILITIES ========================================================================================

def GeneratePoseGivenUp(sensorPosition, targetPosition, upAxis):
  '''Generates the sensor pose with the LOS pointing to a target position and the "up" close to a specified up.

  - Input sensorPosition: 3-element desired position of sensor placement.
  - Input targetPosition: 3-element position of object required to view.
  - Input upAxis: The direction the sensor up should be close to.
  - Returns T: 4x4 numpy array (transformation matrix) representing desired pose of end effector in the base frame.
  '''

  v = targetPosition - sensorPosition
  v = v / norm(v)

  u = upAxis - dot(upAxis, v) * v
  u = u / norm(u)

  t = cross(u, v)

  T = eye(4)
  T[0:3,0] = t
  T[0:3,1] = u
  T[0:3,2] = v
  T[0:3,3] = sensorPosition

  return T