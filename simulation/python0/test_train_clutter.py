#!/usr/bin/env python
'''Testing reinforcement learning with grasp images as part of the state.'''

# python
import sys
import time
# scipy
from scipy.io import savemat
from numpy import array, mean, pi, zeros
# self
from rl_agent import RlAgent
from three_d_net import ThreeDNet
from rl_environment import RlEnvironment

def main(saveFileSuffix):
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================

  # objects
  objectClass = "bottle_train"
  nObjects = 7
  randomObjectScale = True
  targetObjectAxis = array([0,0,1])
  maxAngleFromObjectAxis = 20*(pi/180)
  maxObjectTableGap = 0.02

  # view
  viewCenter = array([0,0,0])
  viewKeepout = 0.50
  viewWorkspace = [(-1,1),(-1,1),(-1,1)]
  objViewWorkspace = [(-1,1),(-1,1),(0.002,1)]

  # grasps
  graspDetectMode = 1 # 0=sample, 1=sample+label
  nGraspSamples = 500
  graspScoreThresh = 300
  nGraspInliers = 2

  # learning
  nValueIterations = 70
  nDataIterations = 50
  nGraspIterations = 20
  pickEpsilon = 1.0
  placeEpsilon = 1.0
  minPickEpsilon = 0.10
  minPlaceEpsilon = 0.10
  pickEpsilonDelta = 0.05
  placeEpsilonDelta = 0.05
  maxExperiences = 25000
  trainingBatchSize = 25000
  unbiasOnIteration = nValueIterations-5

  # visualization/saving
  saveFileName = "results" + saveFileSuffix + ".mat"
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

  averageReward = []; placeActionCounts = []
  trainLosses = []; testLosses = []
  databaseSize = []; iterationTime = []

  for valueIterationIdx in xrange(nValueIterations):

    print("Iteration {}. Epsilon pick: {}, place: {}".format(\
      valueIterationIdx, pickEpsilon, placeEpsilon))

    # Collect data for this training iteration.

    iterationStartTime = time.time()
    R = []; placeCounts = zeros(nPlaceOptions)

    # check if it's time to unbias data
    if valueIterationIdx >= unbiasOnIteration:
      maxExperiences = trainingBatchSize # selects all recent experiences, unbiased
      pickEpsilon = 0 # estimating value function of actual policy
      placeEpsilon = 0 # estimating value function of actual policy

    for dataIterationIdx in xrange(nDataIterations):

      # place clutter on table

      fullObjNames, objScales = threeDNet.GetRandomObjectSet(
        objectClass, nObjects, randomObjectScale)
      objHandles, objPoses = rlEnv.PlaceObjectSet(fullObjNames, objScales)

      objCloud, objCloudIdxs = rlEnv.AssignPointsToObjects(
        rlAgent, objHandles, viewCenter, viewKeepout, objViewWorkspace)

      # move the hand to view position and capture a point cloud
      cloud, viewPoints, viewPointIndices = rlAgent.GetDualCloud(
        viewCenter, viewKeepout, viewWorkspace)
      rlAgent.PlotCloud(cloud)

      # detect grasps in the sensory data
      graspsDetected = rlAgent.DetectGrasps(cloud, viewPoints, viewPointIndices,
        nGraspSamples, graspScoreThresh, nGraspInliers, graspDetectMode)
      grasps = rlAgent.FilterGraspsWithNoPoints(graspsDetected, objCloud)
      if len(graspsDetected) > len(grasps):
        print("Fitlered {} empty grasps.".format(len(graspsDetected)-len(grasps)))
      rlAgent.PlotGrasps(grasps)

      if showSteps:
        raw_input("Acquired grasps.")

      if len(grasps) == 0:
        print("No grasps found. Skipping iteration.")
        rlEnv.RemoveObjectSet(objHandles)
        rlAgent.UnplotGrasps()
        rlAgent.UnplotCloud()
        continue

      for graspIterationIdx in xrange(nGraspIterations):

        print("Episode {}.{}.{}.".format(valueIterationIdx, dataIterationIdx, graspIterationIdx))

        # perform pick action
        grasp = rlAgent.GetGrasp(grasps, pickEpsilon)
        s = rlEnv.GetState(rlAgent, grasp, None)
        rlAgent.PlotGrasps([grasp])

        if showSteps:
          print("Selected grasp.")

        # perform place action
        P = rlAgent.GetPlacePose(grasp, placeEpsilon)
        rlAgent.MoveHandToPose(P)
        ss = rlEnv.GetState(rlAgent, grasp, P)
        objHandle, objPose = rlEnv.GetObjectWithMaxGraspPoints(\
          grasp, objHandles, objCloud, objCloudIdxs)
        rlAgent.MoveObjectToHandAtGrasp(grasp, objHandle)
        r = rlEnv.RewardHeightExponential(
          objHandle, targetObjectAxis, maxAngleFromObjectAxis, maxObjectTableGap)
        print("The robot receives {} reward.".format(r))

        if showSteps:
          raw_input("Press [Enter] to continue...")

        # add experience to database
        experienceDatabase.append((s, ss, 0)) # grasp -> placement
        experienceDatabase.append((ss, None, r)) # placement -> end

        # record save data
        R.append(r)
        placeCounts += ss[1][len(s[1])-nPlaceOptions:]

        # cleanup this grasp iteration
        rlAgent.UnplotGrasps()
        rlEnv.MoveObjectToPose(objHandle, objPose)

      # cleanup this data iteration
      rlEnv.RemoveObjectSet(objHandles)
      rlAgent.UnplotCloud()

    # Compute value labels for data.
    experienceDatabase = rlAgent.PruneDatabase(experienceDatabase, maxExperiences)
    Dl = rlAgent.DownsampleAndLabelData(\
      experienceDatabase, trainingBatchSize)
    databaseSize.append(len(experienceDatabase))

    # Train network from replay database.
    trainLoss, testLoss = rlAgent.Train(Dl, recordLoss=recordLoss)
    trainLosses.append(trainLoss)
    testLosses.append(testLoss)

    pickEpsilon -= pickEpsilonDelta
    placeEpsilon -= placeEpsilonDelta
    pickEpsilon = max(minPickEpsilon, pickEpsilon)
    placeEpsilon = max(minPlaceEpsilon, placeEpsilon)

    # Save results
    averageReward.append(mean(R))
    placeActionCounts.append(placeCounts)
    iterationTime.append(time.time()-iterationStartTime)
    saveData = {"objectClass":objectClass, "nObjects":nObjects,
      "randomObjectScale":randomObjectScale, "maxAngleFromObjectAxis":maxAngleFromObjectAxis,
      "maxObjectTableGap":maxObjectTableGap, "graspDetectMode":graspDetectMode,
      "nGraspSamples":nGraspSamples, "graspScoreThresh":graspScoreThresh,
      "nValueIterations":nValueIterations, "nDataIterations":nDataIterations,
      "nGraspIterations":nGraspIterations, "pickEpsilon":pickEpsilon, "placeEpsilon":placeEpsilon,
      "minPickEpsilon":minPickEpsilon, "minPlaceEpsilon":minPlaceEpsilon,
      "pickEpsilonDelta":pickEpsilonDelta, "placeEpsilonDelta":placeEpsilonDelta,
      "maxExperiences":maxExperiences, "trainingBatchSize":trainingBatchSize,
      "averageReward":averageReward, "placeActionCounts":placeActionCounts, "trainLoss":trainLosses,
      "testLoss":testLosses,"databaseSize":databaseSize,"iterationTime":iterationTime,
      "placePoses":rlAgent.placePoses}
    savemat(saveFileName, saveData)

if __name__ == "__main__":
  '''Checks arguments and runs main function.'''

  saveFileSuffix = ""
  if len(sys.argv) > 1:
    saveFileSuffix = sys.argv[1]

  main(saveFileSuffix)
  exit()
