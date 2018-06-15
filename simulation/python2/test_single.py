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

def main(objectClass, epsilon):
  '''Entrypoint to the program.
    - Input objectClass: Folder in 3D Net database.
  '''

  # PARAMETERS =====================================================================================

  # objects
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
  nGraspSamples = 100
  graspScoreThresh = 350

  # learning
  weightsFileName = "/home/mgualti/mgualti/PickAndPlace/simulation/caffe/image_iter_5000.caffemodel"
  nDataIterations = 300

  # visualization/saving
  showViewer = False
  showEveryStep = False
  saveFileName = "results-single-" + objectClass + "-epsilon" + str(epsilon) + ".mat"

  # INITIALIZATION =================================================================================

  threeDNet = ThreeDNet()
  rlEnv = RlEnvironment(showViewer)
  rlAgent = RlAgent(rlEnv)
  if epsilon < 1.0: rlAgent.LoadNetworkWeights(weightsFileName)
  Return = []

  # RUN TEST =======================================================================================

  # Collect data for this training iteration.

  for dataIterationIdx in xrange(nDataIterations):

    print("Iteration {}.".format(dataIterationIdx))

    # place object in random orientation on table
    fullObjName, objScale = threeDNet.GetRandomObjectFromClass(objectClass, randomScale)
    objHandle, objRandPose = rlEnv.PlaceObjectRandomOrientation(fullObjName, objScale)

    # move the hand to view position and capture a point cloud
    cloud, viewPoints, viewPointIndices = rlAgent.GetDualCloud(
      viewCenter, viewKeepout, viewWorkspace)
    rlAgent.PlotCloud(cloud)

    # detect grasps in the sensory data
    grasps = rlAgent.DetectGrasps(cloud, viewPoints, viewPointIndices,
      nGraspSamples, graspScoreThresh, graspDetectMode)
    rlAgent.PlotGrasps(grasps)

    if len(grasps) == 0:
      print("No grasps found. Skipping iteration.")
      rlEnv.RemoveObject(objHandle)
      rlAgent.UnplotCloud()
      continue

    if showEveryStep:
      raw_input("Press [Enter] to continue...")

    # perform pick action
    grasp = rlAgent.GetGrasp(grasps, epsilon)
    rlAgent.PlotGrasps([grasp])

    if showEveryStep:
      raw_input("Press [Enter] to continue...")

    # perform place action
    P = rlAgent.GetPlacePose(grasp, epsilon)
    rlAgent.MoveHandToPose(P)
    rlAgent.MoveObjectToHandAtGrasp(grasp, objHandle)
    r = rlEnv.RewardBinary(
      objHandle, targetObjectAxis, maxAngleFromObjectAxis, maxObjectTableGap)
    print("The robot receives {} reward.".format(r))
    Return.append(r)

    if showEveryStep:
      raw_input("Press [Enter] to continue...")

    # cleanup this data iteration
    rlEnv.RemoveObject(objHandle)
    rlAgent.UnplotGrasps()
    rlAgent.UnplotCloud()

    saveData = {"randomScale":randomScale, "targetObjectAxis":targetObjectAxis,
      "maxAngleFromObjectAxis":maxAngleFromObjectAxis, "maxObjectTableGap":maxObjectTableGap,
      "graspDetectMode":graspDetectMode, "nGraspSamples":nGraspSamples,
      "graspScoreThresh":graspScoreThresh, "weightsFileName":weightsFileName,
      "nDataIterations":nDataIterations, "epsilon":epsilon, "Return":Return}
    savemat(saveFileName, saveData)

if __name__ == "__main__":
  '''Module entrypoint.'''

  try:
    objectClass = sys.argv[1]
    epsilon = float(sys.argv[2])
  except:
    print("Usage: python2/test_single.py objectClass epsilon")
    exit()

  main(objectClass, epsilon)
  exit()
