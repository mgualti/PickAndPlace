#!/usr/bin/env python
'''The first module for testing reinforcement learning with a simple neural network. Images are not
   yet part of the state.'''

# python
import sys
# scipy
#from scipy.io import savemat
#from numpy import array, pi
# self
#from rl_agent import RlAgent
from three_d_net import ThreeDNet
from rl_environment import RlEnvironment

def main(objectClass):
  '''Entrypoint to the program.
    - Input objectClass: Folder in 3D Net database.
  '''

  # PARAMETERS =====================================================================================

  # INITIALIZATION =================================================================================

  threeDNet = ThreeDNet()
  rlEnv = RlEnvironment(True)

  # RUN TEST =======================================================================================

  objectNames = threeDNet.GetObjectNames(objectClass)
  
  for objectIdx, objectName in enumerate(objectNames):
    
    print("Showing object {}: ".format(objectIdx+1) + objectName)
    
    fullObjName, objScale = threeDNet.GetObjectByName(objectClass, objectName)
    objHandle, objRandPose = rlEnv.PlaceObjectRandomOrientation(fullObjName, objScale)
    
    raw_input("Press [Enter] to continue...")
    rlEnv.RemoveObject(objHandle)
    
  print("Finished.")

if __name__ == "__main__":
  '''Usage: ./test_learning_result.py objectClass'''

  objectClass = sys.argv[1]
  main(objectClass)
  exit()