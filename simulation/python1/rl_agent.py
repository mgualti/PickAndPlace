'''Reinforcement learning (RL) agent and related utility functions.'''

# python
from copy import copy
# scipy
from numpy.linalg import inv, norm
from numpy.random import choice, permutation, randint, rand
from numpy import arange, array, cos, cross, dot, empty, eye, linspace, pi, sin, vstack, zeros
# openrave
import openravepy
# caffe
import h5py
import caffe
# self
import point_cloud
from grasp_proxy_matlab import GraspProxyMatlab

# AGENT ============================================================================================

class RlAgent:

  def __init__(self, rlEnvironment):
    '''Initializes agent in the given environment.'''

    # parameters

    self.caffeDir = "/home/mgualti/mgualti/PickAndPlace/simulation/caffe/"
    self.caffeWeightsFilePrefix = self.caffeDir + "dualImage_iter_"
    self.caffeModelFileName = self.caffeDir + "networkDeploy-dualImage.prototxt"
    self.caffeSolverFileName = self.caffeDir + "solver-dualImage.prototxt"
    self.caffeTrainFileName = self.caffeDir + "train.h5"
    self.caffeTestFileName = self.caffeDir + "test.h5"

    self.emptyImage = zeros((12,60,60))

    # simple assignments
    self.rlEnv = rlEnvironment
    self.env = self.rlEnv.env
    self.robot = self.rlEnv.robot
    self.plotCloudHandle = None
    self.plotGraspsHandle = None

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

    placePositions = zeros((16, 3))
    placePositions[:, 2] = arange(0.05, 0.20, 0.01)

    isFinalPlaceOrientation = [1,0]

    self.placePoses = []; self.isFinalPlacePose = []
    for i in xrange(len(placeOrientations)):
      for j in xrange(len(placePositions)):
        P = eye(4)
        P[0:3,0:3] = placeOrientations[i]
        P[0:3,3] = placePositions[j]
        P = dot(P, self.sTh)
        self.MoveHandToPose(P)
        if not self.env.CheckCollision(self.robot):
          isFinalPlacePose = 1 if isFinalPlaceOrientation[i] else 0
          self.isFinalPlacePose.append(isFinalPlacePose)
          if isFinalPlacePose: P[0,3] = -0.5
          self.placePoses.append(P)

    print("There are {} place poses and {} are final placements.".format(
      len(self.placePoses), sum(self.isFinalPlacePose)))

    # initialize grasp proxy
    self.graspProxy = GraspProxyMatlab()

    # initialize caffe    
    caffe.set_mode_gpu()
    caffe.set_device(1)
    self.caffeNet = caffe.Net(self.caffeModelFileName, caffe.TEST)
    self.caffeFirstTrain = True

  def ChooseAction(self, state, grasps, epsilon):
    '''Chooses the next action from (grasp, place, placeTemp) with an epsilon-greedy policy.
      - Input state: The representation of the current state.
      - Input grasps: The available grasps to choose from.
      - Input epsilon: A number in [0,1] indicating the probability of a (uniformly) random action.
      - Returns: (action, grasp, place) where action is the action vector
                 (I, [placeTemp, placeFinal]); where grasp is the selected Grasp object from G or
                 None; and where place is the selected place pose or None.
    '''

    # 1. Initialization

    grasp = None; place = None
    isSelected = state[1][0]
    isPlacedTempGood = state[1][1]
    isPlacedTempBad = state[1][2]
    isPlacedFinalGood = state[1][3]
    isPlacedFinalBad = state[1][4]
    emptyPlace = zeros(len(self.placePoses))

    if isPlacedTempBad or isPlacedFinalGood or isPlacedFinalBad:
      # In terminal state. Return no-op.
      return None, None, None

    if rand() <= epsilon:

      # 2. Epsilon

      if not isSelected:
        grasp = self.GetRandomGrasp(grasps)
      else:
        place = self.GetRandomPlacePose()

    else:

      # 3. Greedy

      bestValue = float("-Inf")
      if not isSelected:

        # evaluate grasps
        for g in grasps:
          self.caffeNet.blobs["state-image"].data[0] = state[0]
          self.caffeNet.blobs["action-image"].data[0] = g.image
          self.caffeNet.blobs["state-vector"].data[0] = state[1]
          self.caffeNet.blobs["action-vector"].data[0] = emptyPlace
          self.caffeNet.forward()
          value = self.caffeNet.blobs["ip3"].data[0,0]
          if value > bestValue:
            grasp = g
            bestValue = value

      else:

        # evaluate places
        for pIdx, p in enumerate(self.placePoses):
          pVect = copy(emptyPlace); pVect[pIdx] = 1
          self.caffeNet.blobs["state-image"].data[0] = state[0]
          self.caffeNet.blobs["action-image"].data[0] = self.emptyImage
          self.caffeNet.blobs["state-vector"].data[0] = state[1]
          self.caffeNet.blobs["action-vector"].data[0] = pVect
          self.caffeNet.forward()
          value = self.caffeNet.blobs["ip3"].data[0,0]
          if value > bestValue:
            place = p
            bestValue = value

    # 4. Compose a and return result

    I = self.emptyImage if grasp is None else grasp.image
    p = emptyPlace
    if place is not None:
      p[self.GetPlaceIndex(place)] = 1

    action = (I, p)
    return action, grasp, place

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
    offsets = [0.08, 0.0]

    viewPointIndices = viewPointIndices + 1
    tableUpAxis = 3
    tablePosition = copy(self.rlEnv.tablePosition)
    tablePosition[tableUpAxis-1] += 0.002
    tableFingerLength = 0.05
    minWidth = 0.002; maxWidth = 0.085

    # detect
    if detectMode == 0:
      grasps = self.graspProxy.SampleGrasps(cloud, viewPoints, viewPointIndices, nSamples, minWidth,
        maxWidth, tablePosition, tableUpAxis, tableFingerLength, offsets)
    elif detectMode == 1:
      grasps = self.graspProxy.DetectGrasps(cloud, viewPoints, viewPointIndices, nSamples,
        scoreThresh, minWidth, maxWidth, tablePosition, tableUpAxis, tableFingerLength, offsets)
    else:
      raise Exception("Unrecognized grasp detection mode {}.".format(detectMode))

    grasps = self.FindInliers(grasps, 3)

    return grasps

  def DownsampleAndLabelData(self, D, batchSize, gamma):
    '''Samples data to a batch size, uniformly at random, and labels the data with the current value function approximation.'''

    if len(D) <= batchSize:
      return self.LabelData(D, gamma)

    idxs = choice(len(D), batchSize, replace=False)
    batchD = [D[i] for i in idxs]
    return self.LabelData(batchD, gamma)

  def FilterGraspsInCollision(self, grasps):
    '''Returns only the grasps that do not result in a collision with either the table or the
       object when the hand is positioned there.'''

    T = self.robot.GetTransform()

    keepGrasps = []
    for grasp in grasps:
      self.MoveSensorToPose(grasp.poses[0])
      if not self.env.CheckCollision(self.robot):
        keepGrasps.append(grasp)

    self.robot.SetTransform(T)
    return keepGrasps

  def FindInliers(self, grasps, minInliers):
    '''Removes grasps that are lonely.'''

    angleToPosition = 0.14 # 20 deg = 1cm
    neighborDistThresh = 0.03

    graspsWithInliers = []
    for i, grasp in enumerate(grasps):
      nInliers = 0
      for j, neighbor in enumerate(grasps):
        if i == j: continue
        d = norm(grasp.bottom - neighbor.bottom) + \
          angleToPosition*(1.0 - dot(grasp.approach, neighbor.approach)) + \
          angleToPosition*(1.0 - dot(grasp.axis, neighbor.axis))
        if d <= neighborDistThresh:
          nInliers += 1
        if nInliers >= minInliers:
          graspsWithInliers.append(grasp)
          break

    return graspsWithInliers

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

  def GetPlaceIndex(self, pose):
    '''Finds the index into self.placePoses that this pose belongs.'''

    for i, p in enumerate(self.placePoses):
      if sum(sum(abs(p - pose))) < 1e-5:
        return i
    return None

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

  def LabelData(self, D, gamma):
    '''Given a database of (state, action, reward, nextState, nextAction) and the CNN approximating
       future return, compute an estimate of the action-value function.
      - Return Dl: List of targets for the network to learn: (state, action, value).
    '''

    Dl = []
    for d in D:

      s = d[0]; a = d[1]; rr = d[2]; ss = d[3]; aa = d[4]

      if aa is None:
        qq = 0 # terminal state -- known to have 0 value
      else:
        self.caffeNet.blobs["state-image"].data[0] = ss[0]
        self.caffeNet.blobs["action-image"].data[0] = aa[0]
        self.caffeNet.blobs["state-vector"].data[0] = ss[1]
        self.caffeNet.blobs["action-vector"].data[0] = aa[1]
        self.caffeNet.forward()
        qq = self.caffeNet.blobs["ip3"].data[0,0]

      q = rr + gamma*qq # Bellman update
      Dl.append((s, a, q))

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
      - Input Dl: List of tuples with (state,action,value).
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

    sampleImage = Dl[0][0][0]
    nx = sampleImage.shape[0]
    ny = sampleImage.shape[1]
    nz = sampleImage.shape[2]
    nvs = len(Dl[0][0][1])
    nva = len(Dl[0][1][1])

    Is = zeros((nTrain, nx, ny, nz))
    Ia = zeros((nTrain, nx, ny, nz))
    Vs = zeros((nTrain, nvs))
    Va = zeros((nTrain, nva))
    L = zeros(nTrain)
    for i, idx in enumerate(idxs):
      Is[i, :, :, :] = Dl[idx][0][0]
      Ia[i, :, :, :] = Dl[idx][1][0]
      Vs[i, :] = Dl[idx][0][1]
      Va[i, :] = Dl[idx][1][1]
      L[i] = Dl[idx][2]

    with h5py.File(self.caffeTrainFileName, 'w') as fileHandle:
      fileHandle.create_dataset("state-image", data=Is)
      fileHandle.create_dataset("action-image", data=Ia)
      fileHandle.create_dataset("state-vector", data=Vs)
      fileHandle.create_dataset("action-vector", data=Va)
      fileHandle.create_dataset("label", data=L)

    Is = zeros((nTest, nx, ny, nz))
    Ia = zeros((nTest, nx, ny, nz))
    Vs = zeros((nTest, nvs))
    Va = zeros((nTest, nva))
    L = zeros(nTest)
    for j, jdx in enumerate(jdxs):
      Is[j, :, :, :] = Dl[jdx][0][0]
      Ia[j, :, :, :] = Dl[jdx][1][0]
      Vs[j, :] = Dl[jdx][0][1]
      Va[j, :] = Dl[jdx][1][1]
      L[j] = Dl[jdx][2]

    with h5py.File(self.caffeTestFileName, 'w') as fileHandle:
      fileHandle.create_dataset("state-image", data=Is)
      fileHandle.create_dataset("action-image", data=Ia)
      fileHandle.create_dataset("state-vector", data=Vs)
      fileHandle.create_dataset("action-vector", data=Va)
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