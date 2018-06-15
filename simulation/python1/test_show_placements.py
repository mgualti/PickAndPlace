#!/usr/bin/env python
'''The first module for testing reinforcement learning with a simple neural network. Images are not
   yet part of the state.'''

# python
# scipy
#from scipy.io import savemat
#from numpy import array, pi
# self
from rl_agent import RlAgent
from rl_environment import RlEnvironment

def main():
  '''Entrypoint to the program.
    - Input objectClass: Folder in 3D Net database.
  '''

  # PARAMETERS =====================================================================================

  # INITIALIZATION =================================================================================

  rlEnv = RlEnvironment(True)
  rlAgent = RlAgent(rlEnv)

  # RUN TEST =======================================================================================

  for placeIdx, placePose in enumerate(rlAgent.placePoses):

    print("Showing pose {}: ".format(placeIdx))
    print placePose

    rlAgent.MoveHandToPose(placePose)

    raw_input("Press [Enter] to continue...")

  print("Finished.")

if __name__ == "__main__":
  '''Program entrypoint.'''

  main()
  exit()