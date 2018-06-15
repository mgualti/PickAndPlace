#!/usr/bin/env python
'''Testing pick and place problem where one re-grasp is allowed.'''

# python
import time
# scipy
from scipy.io import savemat
from numpy import arccos, array, dot, mean, pi, zeros
# self
from rl_agent import RlAgent
from three_d_net import ThreeDNet
from rl_environment import RlEnvironment

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================

  # objects
  objectClass = "mug_train"
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

  # learning
  nTrainingIterations = 100
  nEpisodes = 100
  nReuses = 10
  maxTimesteps = 10
  gamma = 0.98
  epsilon = 1.0
  epsilonDelta = 0.05
  minEpsilon = 0.05
  maxExperiences = 50000
  trainingBatchSize = 50000
  unbiasOnIteration = nTrainingIterations-5

  # visualization/saving
  saveFileName = "results.mat"
  recordLoss = True
  showViewer = False
  showSteps = False

  # INITIALIZATION =================================================================================

  threeDNet = ThreeDNet()
  rlEnv = RlEnvironment(showViewer)
  rlAgent = RlAgent(rlEnv)
  nPlaceOptions = len(rlAgent.placePoses)
  experienceDatabase = []

  # RUN TEST =======================================================================================

  avgReturn = []; avgGraspsDetected = []; avgTopGraspsDetected = []; placeHistograms = []
  avgGoodTempPlaceCount = []; avgBadTempPlaceCount = []
  avgGoodFinalPlaceCount = []; avgBadFinalPlaceCount = []
  trainLosses = []; testLosses = []; databaseSize = []; iterationTime = []

  for trainingIteration in xrange(nTrainingIterations):

    # initialization
    iterationStartTime = time.time()
    print("Iteration: {}, Epsilon: {}".format(trainingIteration, epsilon))

    placeHistogram = zeros(nPlaceOptions)
    Return = []; graspsDetected = []; topGraspsDetected = []
    goodTempPlaceCount = []; badTempPlaceCount = []
    goodFinalPlaceCount = []; badFinalPlaceCount = []

    # check if it's time to unbias data
    if trainingIteration >= unbiasOnIteration:
      maxExperiences = trainingBatchSize # selects all recent experiences, unbiased
      epsilon = 0 # estimating value function of actual policy

    # for each episode/object placement
    for episode in xrange(nEpisodes):

      # place random object in random orientation on table
      fullObjName, objScale = threeDNet.GetRandomObjectFromClass(objectClass, randomScale)
      objHandle, objRandPose = rlEnv.PlaceObjectRandomOrientation(fullObjName, objScale)

      # move the hand to view position(s) and capture a point cloud
      cloud, viewPoints, viewPointIndices = rlAgent.GetDualCloud(
        viewCenter, viewKeepout, viewWorkspace)

      # detect grasps in the sensor data
      grasps = rlAgent.DetectGrasps(cloud, viewPoints, viewPointIndices,
        nGraspSamples, graspScoreThresh, graspDetectMode)
      graspsStart = grasps

      graspsDetected.append(len(grasps))
      topGraspsCount = CountObjectTopGrasps(grasps, objRandPose, maxAngleFromObjectAxis)

      if len(grasps) == 0:
        print("No grasps found. Skipping iteration.")
        rlEnv.RemoveObject(objHandle)
        continue

      rlAgent.PlotCloud(cloud)
      rlAgent.PlotGrasps(grasps)

      for reuse in xrange(nReuses):

        print("Episode {}.{}.{}.".format(trainingIteration, episode, reuse))

        if showSteps:
          raw_input("Beginning of episode. Press [Enter] to continue...")

        # initialize recording variables
        episodePlaceHistogram = zeros(nPlaceOptions); episodeReturn = 0
        episodeGoodTempPlaceCount = 0; episodeBadTempPlaceCount = 0
        episodeGoodFinalPlaceCount = 0; episodeBadFinalPlaceCount = 0
        graspDetectionFailure = False; episodeExperiences = []

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
            grasps = rlAgent.DetectGrasps(cloud, viewPoints, viewPointIndices,
              nGraspSamples, graspScoreThresh, graspDetectMode)
            graspsDetected.append(len(grasps))
            topGraspsCount = CountObjectTopGrasps(
              grasps, rlEnv.GetObjectPose(objHandle), maxAngleFromObjectAxis)
            topGraspsDetected.append(topGraspsCount)
            if len(grasps) == 0:
              print("Grasp detection failure.")
              graspDetectionFailure = True
              break
            rlAgent.PlotCloud(cloud)
            rlAgent.PlotGrasps(grasps)

          # get next action
          aa, ggrasp, pplace = rlAgent.ChooseAction(ss, grasps, epsilon)
          if ggrasp is not None: rlAgent.PlotGrasps([ggrasp])

          if showSteps:
            raw_input("Action {}. Press [Enter] to continue...".format(t))

          # add to database and record data
          episodeExperiences.append((s,a,rr,ss,aa))
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
          experienceDatabase += episodeExperiences
          placeHistogram += episodePlaceHistogram
          Return.append(episodeReturn)
          goodTempPlaceCount.append(episodeGoodTempPlaceCount)
          badTempPlaceCount.append(episodeBadTempPlaceCount)
          goodFinalPlaceCount.append(episodeGoodFinalPlaceCount)
          badFinalPlaceCount.append(episodeBadFinalPlaceCount)
        rlEnv.MoveObjectToPose(objHandle, objRandPose)
        grasps = graspsStart

      # cleanup this episode
      rlEnv.RemoveObject(objHandle)
      rlAgent.UnplotGrasps()
      rlAgent.UnplotCloud()

    # 2. Compute value labels for data.
    experienceDatabase = rlAgent.PruneDatabase(experienceDatabase, maxExperiences)
    Dl = rlAgent.DownsampleAndLabelData(experienceDatabase, trainingBatchSize, gamma)
    databaseSize.append(len(experienceDatabase))

    # 3. Train network from replay database.
    trainLoss, testLoss = rlAgent.Train(Dl, recordLoss=recordLoss)
    trainLosses.append(trainLoss)
    testLosses.append(testLoss)

    epsilon -= epsilonDelta
    epsilon = max(minEpsilon, epsilon)

    # 4. Save results
    avgReturn.append(mean(Return))
    avgGraspsDetected.append(mean(graspsDetected))
    avgTopGraspsDetected.append(mean(topGraspsDetected))
    placeHistograms.append(placeHistogram)
    avgGoodTempPlaceCount.append(mean(goodTempPlaceCount))
    avgBadTempPlaceCount.append(mean(badTempPlaceCount))
    avgGoodFinalPlaceCount.append(mean(goodFinalPlaceCount))
    avgBadFinalPlaceCount.append(mean(badFinalPlaceCount))
    iterationTime.append(time.time()-iterationStartTime)
    saveData = {"objectClass":objectClass, "randomScale":randomScale,
      "maxAngleFromObjectAxis":maxAngleFromObjectAxis, "maxObjectTableGap":maxObjectTableGap,
      "nGraspSamples":nGraspSamples, "graspScoreThresh":graspScoreThresh,
      "graspDetectMode":graspDetectMode, "nTrainingIterations":nTrainingIterations,
      "nEpisodes":nEpisodes, "maxTimesteps":maxTimesteps,  "gamma":gamma, "epsilon":epsilon,
      "minEpsilon":minEpsilon, "epsilonDelta":epsilonDelta, "maxExperiences":maxExperiences,
      "trainingBatchSize":trainingBatchSize, "avgReturn":avgReturn,
      "avgGraspsDetected":avgGraspsDetected, "avgTopGraspsDetected":avgTopGraspsDetected,
      "placeHistograms":placeHistograms, "avgGoodTempPlaceCount":avgGoodTempPlaceCount,
      "avgBadTempPlaceCount":avgBadTempPlaceCount, "avgGoodFinalPlaceCount":avgGoodFinalPlaceCount,
      "avgBadFinalPlaceCount":avgBadFinalPlaceCount, "trainLoss":trainLosses, "testLoss":testLosses,
      "databaseSize":databaseSize, "iterationTime":iterationTime, "placePoses":rlAgent.placePoses}
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

def SaveCloud(cloud, viewPoints, viewPointIndices, fileName):
  '''Saves point cloud information for testing in Matlab.'''

  viewPointIndices = viewPointIndices + 1 # matlab is 1-indexed
  viewPoints = viewPoints.T
  cloud = cloud.T
  data = {"cloud":cloud, "viewPoints":viewPoints, "viewPointIndices":viewPointIndices}
  savemat(fileName, data)

if __name__ == "__main__":
  main()
  exit()
