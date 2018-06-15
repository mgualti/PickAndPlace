#!/usr/bin/env python
'''Testing reinforcement learning with grasp images as part of the state.'''

# python
import sys
# scipy
from scipy.io import savemat
from numpy import array, pi
# self
from rl_agent import RlAgent
from three_d_net import ThreeDNet
from rl_environment import RlEnvironment

def main(objectClass, epsilon):
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================

  # objects
  nObjects = 7
  randomObjectScale = True
  targetObjectAxis = array([0,0,1])
  maxAngleFromObjectAxis = 20*(pi/180)
  maxObjectTableGap = 0.03

  # view
  viewCenter = array([0,0,0])
  viewKeepout = 0.50
  viewWorkspace = [(-1,1),(-1,1),(0.002,1)]
  objViewWorkspace = [(-1,1),(-1,1),(0.002,1)]

  # grasps
  graspDetectMode = 0 # 0=sample, 1=sample+label
  nGraspSamples = 150
  graspScoreThresh = 350

  # testing
  weightsFileName = "/home/mgualti/mgualti/PickAndPlace/simulation/caffe/image_iter_5000.caffemodel"
  nDataIterations = 300

  # visualization/saving
  saveFileName = "results-clutter-" + objectClass + "-epsilon" + str(epsilon) + ".mat"
  showViewer = False
  showSteps = False

  # INITIALIZATION =================================================================================

  threeDNet = ThreeDNet()
  rlEnv = RlEnvironment(showViewer)
  rlAgent = RlAgent(rlEnv)
  if epsilon < 1.0: rlAgent.LoadNetworkWeights(weightsFileName)
  Return = []

  # RUN TEST =======================================================================================

  for dataIterationIdx in xrange(nDataIterations):

    print("Iteration {}.".format(dataIterationIdx))

    # place clutter on table
    fullObjNames, objScales = threeDNet.GetRandomObjectSet(
      objectClass, nObjects, randomObjectScale)
    objHandles, objPoses = rlEnv.PlaceObjectSet(fullObjNames, objScales)

    objCloud, objCloudIdxs = rlEnv.AssignPointsToObjects(
        rlAgent, objHandles, viewCenter, viewKeepout, objViewWorkspace)

    if showSteps:
      raw_input("Objects placed.")

    # move the hand to view position and capture a point cloud
    cloud, viewPoints, viewPointIndices = rlAgent.GetDualCloud(
        viewCenter, viewKeepout, viewWorkspace)
    rlAgent.PlotCloud(cloud)

    if showSteps:
      raw_input("Point cloud.")

    # detect grasps in the sensory data
    graspsDetected = rlAgent.DetectGrasps(cloud, viewPoints, viewPointIndices,
        nGraspSamples, graspScoreThresh, graspDetectMode)
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

    # perform pick action
    grasp = rlAgent.GetGrasp(grasps, epsilon)
    rlAgent.PlotGrasps([grasp])

    if showSteps:
      raw_input("Selected grasp.")

    # perform place action
    P = rlAgent.GetPlacePose(grasp, epsilon)
    rlAgent.MoveHandToPose(P)
    objHandle, objPose = rlEnv.GetObjectWithMaxGraspPoints(
      grasp, objHandles, objCloud, objCloudIdxs)
    rlAgent.MoveObjectToHandAtGrasp(grasp, objHandle)
    r = rlEnv.RewardBinary(
      objHandle, targetObjectAxis, maxAngleFromObjectAxis, maxObjectTableGap)
    print("The robot receives {} reward.".format(r))
    Return.append(r)

    if showSteps:
      raw_input("Press [Enter] to continue...")

    # cleanup this data iteration
    rlEnv.RemoveObjectSet(objHandles)
    rlAgent.UnplotGrasps()
    rlAgent.UnplotCloud()

    # Save results
    saveData = {"nObjects":nObjects, "randomObjectScale":randomObjectScale,
      "targetObjectAxis":targetObjectAxis, "maxAngleFromObjectAxis":maxAngleFromObjectAxis,
      "maxObjectTableGap":maxObjectTableGap, "graspDetectMode":graspDetectMode,
      "nGraspSamples":nGraspSamples, "graspScoreThresh":graspScoreThresh,
      "weightsFileName":weightsFileName, "nDataIterations":nDataIterations,
      "epsilon":epsilon, "Return":Return}
    savemat(saveFileName, saveData)

if __name__ == "__main__":
  '''Entrypoint to python module.'''

  try:
    objectClass = sys.argv[1]
    epsilon = float(sys.argv[2])
  except:
    print("Usage: python2/test_clutter.py objectClass epsilon")
    exit()

  main(objectClass, epsilon)
  exit()

