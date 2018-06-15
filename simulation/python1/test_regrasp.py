#!/usr/bin/env python
'''Testing pick and place problem where one re-grasp is allowed.'''

# python
import sys
# scipy
from scipy.io import savemat
from numpy import arccos, array, dot, pi, zeros
# self
from rl_agent import RlAgent
from three_d_net import ThreeDNet
from rl_environment import RlEnvironment

def main(objectClass):
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================

  # objects
  randomScale = True
  targetObjectAxis = array([0,0,1])
  maxAngleFromObjectAxis = 20*(pi/180)
  maxObjectTableGap = 0.03

  # view
  viewCenter = array([0,0,0])
  viewKeepout = 0.50
  viewWorkspace = [(-1,1),(-1,1),(-1,1)]

  # grasps
  graspDetectMode = 1 # 0=sample, 1=sample+label
  nGraspSamples = 200
  graspScoreThresh = 300

  # testing
  nEpisodes = 300
  maxTimesteps = 10
  gamma = 0.98
  epsilon = 0.0
  weightsFileName = \
    "/home/mgualti/mgualti/PickAndPlace/simulation/caffe/dualImage_iter_5000.caffemodel"

  # visualization/saving
  saveFileName = "results-" + objectClass + ".mat"
  showViewer = False
  showSteps = False

  # INITIALIZATION =================================================================================

  threeDNet = ThreeDNet()
  rlEnv = RlEnvironment(showViewer)
  rlAgent = RlAgent(rlEnv)
  rlAgent.LoadNetworkWeights(weightsFileName)
  nPlaceOptions = len(rlAgent.placePoses)

  placeHistogram = zeros(nPlaceOptions)
  Return = []; graspsDetected = []; topGraspsDetected = []
  goodTempPlaceCount = []; badTempPlaceCount = []
  goodFinalPlaceCount = []; badFinalPlaceCount = []

  # RUN TEST =======================================================================================

  # for each episode/object placement
  for episode in xrange(nEpisodes):

    # place random object in random orientation on table
    fullObjName, objScale = threeDNet.GetRandomObjectFromClass(objectClass, randomScale)
    objHandle, objRandPose = rlEnv.PlaceObjectRandomOrientation(fullObjName, objScale)
    rlAgent.MoveSensorToPose(rlAgent.GetStandardViewPose(viewCenter, viewKeepout))

    if showSteps:
      raw_input("Beginning of episode. Press [Enter] to continue...")

    # move the hand to view position(s) and capture a point cloud
    cloud, viewPoints, viewPointIndices = rlAgent.GetDualCloud(
        viewCenter, viewKeepout, viewWorkspace)
    rlAgent.PlotCloud(cloud)

    if showSteps:
      raw_input("Acquired point cloud. Press [Enter] to continue...")

    # detect grasps in the sensor data
    grasps = rlAgent.DetectGrasps(cloud, viewPoints, viewPointIndices,
      nGraspSamples, graspScoreThresh, graspDetectMode)

    graspsDetected.append(len(grasps))
    topGraspsCount = CountObjectTopGrasps(grasps, objRandPose, maxAngleFromObjectAxis)

    if len(grasps) == 0:
      print("No grasps found. Skipping iteration.")
      rlEnv.RemoveObject(objHandle)
      continue

    rlAgent.PlotGrasps(grasps)

    print("Episode {}.".format(episode))

    if showSteps:
      raw_input("Acquired grasps. Press [Enter] to continue...")

    # initialize recording variables
    episodePlaceHistogram = zeros(nPlaceOptions); episodeReturn = 0
    episodeGoodTempPlaceCount = 0; episodeBadTempPlaceCount = 0
    episodeGoodFinalPlaceCount = 0; episodeBadFinalPlaceCount = 0
    graspDetectionFailure = False

    # initial state and first action
    s, selectedGrasp = rlEnv.GetInitialState(rlAgent)
    a, grasp, place = rlAgent.ChooseAction(s, grasps, epsilon)
    rlAgent.PlotGrasps([grasp])

    # for each time step in the episode
    for t in xrange(maxTimesteps):

      ss, selectedGrasp, rr = rlEnv.Transition(rlAgent, objHandle, s, selectedGrasp, a, grasp,
        place, targetObjectAxis, maxAngleFromObjectAxis, maxObjectTableGap)
      ssIsPlacedTempGood = ss[1][1]; ssIsPlacedTempBad = ss[1][2]
      ssIsPlacedFinalGood = ss[1][3]; ssIsPlacedFinalBad = ss[1][4]

      if showSteps:
        raw_input("Transition {}. Press [Enter] to continue...".format(t))

      # re-detect only if a non-terminal placement just happened
      if ssIsPlacedTempGood and place is not None:
        cloud, viewPoints, viewPointIndices = rlAgent.GetDualCloud(
          viewCenter, viewKeepout, viewWorkspace)
        rlAgent.UnplotGrasps()
        rlAgent.PlotCloud(cloud)
        if showSteps:
          raw_input("Acquired cloud. Press [Enter] to continue...")
        grasps = rlAgent.DetectGrasps(cloud, viewPoints, viewPointIndices,
          nGraspSamples, graspScoreThresh, graspDetectMode)
        graspsDetected.append(len(grasps))
        topGraspsCount = CountObjectTopGrasps(grasps, objRandPose, maxAngleFromObjectAxis)
        topGraspsDetected.append(topGraspsCount)
        if len(grasps) == 0:
          print("Grasp detection failure.")
          graspDetectionFailure = True
          break
        rlAgent.PlotGrasps(grasps)
        if showSteps:
          raw_input("Acquired grasps. Press [Enter] to continue...")

      # get next action
      aa, ggrasp, pplace = rlAgent.ChooseAction(ss, grasps, epsilon)
      if ggrasp is not None: rlAgent.PlotGrasps([ggrasp])

      if showSteps:
        raw_input("Action {}. Press [Enter] to continue...".format(t))

      # record data from transition
      episodeReturn += (gamma**t) * rr
      if place is not None:
        episodeGoodTempPlaceCount += ssIsPlacedTempGood
        episodeBadTempPlaceCount += ssIsPlacedTempBad
        episodeGoodFinalPlaceCount += ssIsPlacedFinalGood
        episodeBadFinalPlaceCount += ssIsPlacedFinalBad
        placeHistogram += a[1]

      # prepare for next time step
      if ssIsPlacedTempBad or ssIsPlacedFinalGood or ssIsPlacedFinalBad: break
      s = ss; a = aa; grasp = ggrasp; place = pplace

    # cleanup this reuse
    if not graspDetectionFailure:
      placeHistogram += episodePlaceHistogram
      Return.append(episodeReturn)
      goodTempPlaceCount.append(episodeGoodTempPlaceCount)
      badTempPlaceCount.append(episodeBadTempPlaceCount)
      goodFinalPlaceCount.append(episodeGoodFinalPlaceCount)
      badFinalPlaceCount.append(episodeBadFinalPlaceCount)

    # cleanup this episode
    rlEnv.RemoveObject(objHandle)
    rlAgent.UnplotGrasps()
    rlAgent.UnplotCloud()

    # Save results
    saveData = {"objectClass":objectClass, "randomScale":randomScale,
      "maxAngleFromObjectAxis":maxAngleFromObjectAxis, "maxObjectTableGap":maxObjectTableGap,
      "nGraspSamples":nGraspSamples, "graspScoreThresh":graspScoreThresh,
      "graspDetectMode":graspDetectMode, "nEpisodes":nEpisodes, "maxTimesteps":maxTimesteps,
      "gamma":gamma, "epsilon":epsilon, "Return":Return, "graspsDetected":graspsDetected,
      "topGraspsDetected":topGraspsDetected, "placeHistogram":placeHistogram,
      "goodTempPlaceCount":goodTempPlaceCount,"badTempPlaceCount":badTempPlaceCount,
      "goodFinalPlaceCount":goodFinalPlaceCount, "badFinalPlaceCount":badFinalPlaceCount}
    savemat(saveFileName, saveData)

def CountObjectTopGrasps(grasps, objectPose, angleThresh):
  '''Returns the number of grasps aligned with the object's z-axis.'''

  bZo = objectPose[0:3,2]

  count = 0
  for grasp in grasps:
    bZg = -grasp.approach
    angle = arccos(dot(bZo, bZg))
    if angle < angleThresh:
      count += 1

  return count

def SenseAndDetect(rlAgent, viewCenter, viewKeepout, viewWorkspace, nGraspSamples,
  graspScoreThresh, detectGraspsMode):
  '''Acquires point cloud and and detects grasps.
    - Input detectGraspsMode: 0 for detecting grasps, 1 for sampling grasps, 2 for geometric
                              conditions on mesh.
    - Return cloud: nx3 numpy array of points.
    - Return grasps: List of grasps detected.
  '''

  if detectGraspsMode == 0 or detectGraspsMode == 1:
    cloud = rlAgent.GetDualCloud(viewCenter, viewKeepout, viewWorkspace)
  elif detectGraspsMode == 2:
    cloud, mesh = rlAgent.GetDualAndFiveCloud(viewCenter, viewKeepout, viewWorkspace)
  else:
    raise Exception("Unrecognized detectGraspsMode {}.".format(detectGraspsMode))

  if detectGraspsMode == 0:
    grasps = rlAgent.DetectGrasps(cloud, nGraspSamples, graspScoreThresh)
  elif detectGraspsMode == 1:
    grasps = rlAgent.DetectGraspsFast(cloud, nGraspSamples)
  elif detectGraspsMode == 2:
    grasps = rlAgent.DetectGroundTruthGrasps(cloud, mesh, nGraspSamples)
  else:
    raise Exception("Unrecognized detectGraspsMode {}.".format(detectGraspsMode))

  return cloud, grasps

if __name__ == "__main__":
  '''Program entrypoint: parse arguments and call main function.'''

  if len(sys.argv) < 1:
    raise Exception("Usage: test_regrasp.py objectClass")

  objectClass = sys.argv[1]

  main(objectClass)
  exit()
