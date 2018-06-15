#!/usr/bin/env python
'''The first module for testing reinforcement learning with a simple neural network. Images are not
   yet part of the state.'''

# python
import sys
# scipy
from scipy.io import savemat
from numpy import array, pi
# self
from rl_agent import RlAgent
from three_d_net import ThreeDNet
from rl_environment import RlEnvironment

def main(objectClass):
  '''Entrypoint to the program.
    - Input objectClass: Folder in 3D Net database.
  '''

  # PARAMETERS =====================================================================================

  # objects
  nObjects = 7
  randomScale = True
  targetObjectAxis = array([0,0,1])
  maxAngleFromObjectAxis = 20*(pi/180)
  maxObjectTableGap = 0.03

  # view
  viewCenter = array([0,0,0])
  viewKeepout = 0.50
  viewWorkspace = [(-1,1),(-1,1),(0.002,1)]

  # grasps
  graspDetectMode = 0 # 0=sample, 1=sample+label
  nGraspSamples = 150
  graspScoreThresh = 350

  # testing
  nDataIterations = 300

  # visualization/saving
  showViewer = False
  showEveryStep = False
  saveFileName = "results-clutter-" + objectClass + "-cylinder.mat"

  # INITIALIZATION =================================================================================

  threeDNet = ThreeDNet()
  rlEnv = RlEnvironment(showViewer)
  rlAgent = RlAgent(rlEnv)
  Return = []

  # RUN TEST =======================================================================================

  # Collect data for this training iteration.

  for dataIterationIdx in xrange(nDataIterations):

    print("Iteration {}.".format(dataIterationIdx))

    # place clutter on table
    fullObjNames, objScales = threeDNet.GetRandomObjectSet(
      objectClass, nObjects, randomScale)
    objHandles, objPoses = rlEnv.PlaceObjectSet(fullObjNames, objScales)

    objCloud, objCloudIdxs = rlEnv.AssignPointsToObjects(
        rlAgent, objHandles, viewCenter, viewKeepout, viewWorkspace)

    # move the hand to view position and capture a point cloud
    cloud, viewPoints, viewPointIndices = rlAgent.GetDualCloud(
        viewCenter, viewKeepout, viewWorkspace)
    rlAgent.PlotCloud(cloud)

    # move the hand to view position and capture a point cloud
    cloud, viewPoints, viewPointIndices = rlAgent.GetDualCloud(
      viewCenter, viewKeepout, viewWorkspace)
    rlAgent.PlotCloud(cloud)

    # detect grasps in the sensory data
    graspsDetected = rlAgent.DetectGrasps(cloud, viewPoints, viewPointIndices,
        nGraspSamples, graspScoreThresh, graspDetectMode)
    grasps = rlAgent.FilterGraspsWithNoPoints(graspsDetected, objCloud)
    if len(graspsDetected) > len(grasps):
      print("Fitlered {} empty grasps.".format(len(graspsDetected)-len(grasps)))
    rlAgent.PlotGrasps(grasps)

    if len(grasps) == 0:
      print("No grasps found. Skipping iteration.")
      rlEnv.RemoveObjectSet(objHandles)
      rlAgent.UnplotGrasps()
      rlAgent.UnplotCloud()
      continue

    if showEveryStep:
      raw_input("Press [Enter] to continue...")

    # perform pick action
    grasp, cylinder = rlAgent.GetGraspCylinder(
      grasps, cloud, viewPoints, viewPointIndices, nObjects, maxAngleFromObjectAxis)
    rlAgent.PlotGrasps([grasp])

    if showEveryStep:
      raw_input("Press [Enter] to continue...")

    # perform place action
    P = rlAgent.GetPlacePoseCylinder(grasp, cylinder, maxObjectTableGap)
    rlAgent.MoveHandToPose(P)
    objHandle, objPose = rlEnv.GetObjectWithMaxGraspPoints(
      grasp, objHandles, objCloud, objCloudIdxs)
    rlAgent.MoveObjectToHandAtGrasp(grasp, objHandle)
    r = rlEnv.RewardBinary(
      objHandle, targetObjectAxis, maxAngleFromObjectAxis, maxObjectTableGap)
    print("The robot receives {} reward.".format(r))
    Return.append(r)

    if showEveryStep:
      raw_input("Press [Enter] to continue...")

    # cleanup this data iteration
    rlEnv.RemoveObjectSet(objHandles)
    rlAgent.UnplotGrasps()
    rlAgent.UnplotCloud()

    saveData = {"nObjects":nObjects, "randomScale":randomScale, "targetObjectAxis":targetObjectAxis,
      "maxAngleFromObjectAxis":maxAngleFromObjectAxis, "maxObjectTableGap":maxObjectTableGap,
      "viewCenter":viewCenter, "viewKeepout":viewKeepout, "viewWorkspace":viewWorkspace,
      "graspDetectMode":graspDetectMode, "nGraspSamples":nGraspSamples,
      "graspScoreThresh":graspScoreThresh, "nDataIterations":nDataIterations, "Return":Return}
    savemat(saveFileName, saveData)

if __name__ == "__main__":
  '''Module entrypoint.'''

  try:
    objectClass = sys.argv[1]
  except:
    print("Usage: python2/test_single.py objectClass epsilon")
    exit()

  main(objectClass)
  exit()
