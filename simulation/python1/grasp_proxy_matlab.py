'''Provides an interface to the Matlab grasp detector.'''

# python
# scipy
from numpy import array, ascontiguousarray, fromstring, reshape, rollaxis
# matplotlib
from matplotlib import pyplot
# matlab
import matlab
import matlab.engine
# self
from grasp import Grasp

class GraspProxyMatlab:
  '''A class for interfacing with grasp detection.'''


  def __init__(self):
    '''Starts Matlab engine.'''

    print("Starting Matlab...")
    self.eng = matlab.engine.start_matlab()

    # add all of the required directories to the MATLAB path
    self.eng.addpath("/home/mgualti/Programs/caffe/matlab")
    self.eng.addpath("/home/mgualti/mgualti/PickAndPlace/simulation/matlab")
    self.eng.addpath("/home/mgualti/mgualti/PickAndPlace/simulation/matlab/gpd2")
    self.eng.parpool()

  def DetectGrasps(self, cloud, viewPoints, viewPointIndices, nSamples, scoreThresh, minWidth,
    maxWidth, tablePosition, tableUpAxis, tableFingerLength, offsets):
    '''Calls the DetectGrasps Matlab script.'''

    # short circuit the process if there are no points in the cloud
    if cloud.shape[0] == 0:
      return []

    mCloud = matlab.double(cloud.T.tolist())
    mViewPoints = matlab.double(viewPoints.T.tolist())
    mViewPointIndices = matlab.int32(viewPointIndices.tolist(), size=(len(viewPointIndices), 1))
    mTablePosition = matlab.double(tablePosition.tolist(), size=(3,1))
    plotBitmap = matlab.logical([False, False, False, False, False]);

    mGrasps = self.eng.DetectGrasps(mCloud, mViewPoints, mViewPointIndices, nSamples, scoreThresh,
      minWidth, maxWidth, mTablePosition, tableUpAxis, tableFingerLength, plotBitmap)

    return self.UnpackGrasps(mGrasps, offsets)

  def SampleGrasps(self, cloud, viewPoints, viewPointIndices, nSamples, minWidth, maxWidth,
    tablePosition, tableUpAxis, tableFingerLength, offsets):
    '''Calls the SampleGrasps Matlab script.'''

    # short circuit the process if there are no points in the cloud
    if cloud.shape[0] == 0:
      return []

    mCloud = matlab.double(cloud.T.tolist())
    mViewPoints = matlab.double(viewPoints.T.tolist())
    mViewPointIndices = matlab.int32(viewPointIndices.tolist(), size=(len(viewPointIndices), 1))
    mTablePosition = matlab.double(tablePosition.tolist(), size=(3,1))
    plotBitmap = matlab.logical([False, False, False, False]);

    mGrasps = self.eng.SampleGrasps(mCloud, mViewPoints, mViewPointIndices, nSamples, minWidth,
      maxWidth, mTablePosition, tableUpAxis, tableFingerLength, plotBitmap)

    return self.UnpackGrasps(mGrasps, offsets)

  def UnpackGrasps(self, mGrasps, offsets):
    '''Extracts the list of grasps in Matlab format and returns a list in Python format.'''

    grasps = []
    for mGrasp in mGrasps:

      top = array(mGrasp["top"]).flatten()
      bottom = array(mGrasp["bottom"]).flatten()
      axis = array(mGrasp["axis"]).flatten()
      approach = array(mGrasp["approach"]).flatten()
      binormal = array(mGrasp["binormal"]).flatten()
      height = mGrasp["height"]
      width = mGrasp["width"]
      score = mGrasp["score"]
      imageSize = array(mGrasp["imageSize"], dtype="int32").flatten()

      # decoding necessary, much faster to send two ASCII strings
      imageH = fromstring(mGrasp["imageH"], dtype="uint8")
      imageL = fromstring(mGrasp["imageL"], dtype="uint8")
      image = imageL + 128*imageH
      image = reshape(image, imageSize, order='F')
      image = ascontiguousarray(image, dtype='float32')
      #pyplot.imshow(image[:,:,0:3])
      #pyplot.show(block=True)
      image = rollaxis(image, 2) # make first dimension channel
      image = image / 255.0 # normalize

      # create grasp object
      grasp = Grasp(approach, axis, top, bottom, height, width, offsets,
        -binormal, score, image)
      grasps.append(grasp)

    return grasps