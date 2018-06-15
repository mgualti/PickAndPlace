'''A module with utilities for manipulating a point cloud (nx3 numpy array).'''

# Python
from copy import copy
# scipy
import numpy
import scipy.io
from numpy.linalg import norm
from matplotlib import pyplot
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D
from numpy import array, ascontiguousarray, concatenate, dot, logical_and, ones, zeros
# ROS
from sensor_msgs import point_cloud2
# pcl
import pcl

def ComputeNormals(cloud, viewPoint=None):
  '''TODO'''

  cloud = ascontiguousarray(cloud) # assumed by cloudprocpy
  n = cloud.shape[0]
  augment = zeros((n, 1))
  bigCloud = concatenate((cloud, augment), axis=1)
  cloudTrajopt = cloudprocpy.CloudXYZ()
  cloudTrajopt.from2dArray(bigCloud)
  normals = cloudprocpy.normalEstimation(cloudTrajopt).to2dArray()[:,4:7]

  # reverse any normals going in the wrong direction
  if viewPoint is not None:
    for i in xrange(len(normals)):
      d = viewPoint-cloud[i]
      if dot(normals[i], d/norm(d)) < 0:
        normals[i] = -normals[i]

  return normals


def FilterNans(cloud):
  '''TODO'''

  mask = numpy.logical_not(numpy.isnan(cloud).any(axis=1))
  cloud = cloud[mask]
  return cloud


def FilterWorkspace(cloud, workspace):
  '''TODO'''

  mask = ((cloud[:,0] >= workspace[0][0]) & (cloud[:,0] <= workspace[0][1]) & \
          (cloud[:,1] >= workspace[1][0]) & (cloud[:,1] <= workspace[1][1]) & \
          (cloud[:,2] >= workspace[2][0]) & (cloud[:,2] <= workspace[2][1]))
  cloud = cloud[mask, :]

  return cloud


def FilterNearAndFarPoints(cloud, minDistFromSensor, maxDistFromSensor):
  '''TODO'''

  mask = logical_and(cloud[:, 2] >= minDistFromSensor, cloud[:, 2] <= maxDistFromSensor)
  cloud = cloud[mask, :]
  return cloud


def FilterNearPoints(cloud, distFromSensor):
  '''TODO'''

  mask = cloud[:,2] >= distFromSensor
  cloud = cloud[mask, :]
  return cloud


def FilterStatisticalOutliers(cloud, kNeighbors=50, std=1.0):
  '''TODO'''

  cloud = ascontiguousarray(cloud) # cloudprocpy assumes this
  n = cloud.shape[0]
  augment = zeros((n, 1))
  bigCloud = concatenate((cloud, augment), axis=1)
  cloudTrajopt = cloudprocpy.CloudXYZ()
  cloudTrajopt.from2dArray(bigCloud)
  dsCloud = cloudprocpy.statisticalOutlierRemoval(cloudTrajopt, kNeighbors, std)
  cloud = dsCloud.to2dArray()[:,:3]
  return cloud


def FromMessage(msg):
  '''TODO'''

  cloud = array(list(point_cloud2.read_points(msg)))[:,:3]
  return cloud


def Icp(cloud1, cloud2, tol=1e-5, maxIterations=100, doPrint=True):
  '''The original Iterative Closest Point (ICP) algorithm, by Besl and McKay.
  See "A Method for Registration of 3D Shapes", 1992.

  - Input cloud1: Reference PointCloud.
  - Input cloud2: PointCloud we wish to match to cloud1.
  - Input tol: Iterate until the change in distance error is less than this.
  - Input maxIterations: Give up after this many iterations.
  - Input doPrint: True if some some information should be printed on each iteration.
  - Returns T: Transformation needed to place cloud2 onto cloud1.
  '''

  prevDist = float('inf'); iteration = 0
  T = numpy.eye(4)

  XTree = cKDTree(cloud1)
  X = copy(cloud1)
  P = copy(cloud2)

  while iteration < maxIterations:

    # 1. Compute points in X closest to P.

    dists, idxs = XTree.query(P)
    meanDist = dists.mean()
    diffDist = numpy.fabs(prevDist - meanDist)

    yList = []
    for idx in idxs:
      yList.append(X[idx])
    Y = numpy.array(yList)

    # 2. Check exit condition.
    if doPrint: print("it={}, meanDist={}, diff={}".format(iteration, meanDist, diffDist))
    if diffDist < tol: break
    prevDist = meanDist
    iteration += 1

    # 2. Create cross-covariance matrix.

    up = P.mean(0)
    ux = Y.mean(0)
    Spx = numpy.zeros((3,3))

    for i in xrange(len(Y)):
      Spx += numpy.dot((P[i,:] - up).reshape((3,1)), (Y[i,:] - ux).reshape((1,3)))

    # 3. Solve eigenvalue problem and get rotation.

    A = Spx - Spx.T
    d = numpy.array([A[1,2],A[2,0],A[0,1]]).reshape(1,3)

    tSpx = numpy.array(Spx.trace()).reshape((1,1))
    Q = numpy.zeros((4,4))
    Q[0,:] = numpy.concatenate((tSpx, d), 1)
    Q[1:,:] = numpy.concatenate((d.T, Spx + Spx.T - tSpx*numpy.eye(3)), 1)

    eigVals, eigVects = numpy.linalg.eig(Q)
    qr = eigVects[:, numpy.fabs(eigVals).argmax()]

    R = numpy.array([\
      [qr[0]**2+qr[1]**2-qr[2]**2-qr[3]**2, 2*(qr[1]*qr[2]-qr[0]*qr[3]), 2*(qr[1]*qr[3]+qr[0]*qr[2])],\
      [2*(qr[1]*qr[2]+qr[0]*qr[3]), qr[0]**2+qr[2]**2-qr[1]**2-qr[3]**2, 2*(qr[2]*qr[3]-qr[0]*qr[1])],\
      [2*(qr[1]*qr[3]-qr[0]*qr[2]), 2*(qr[2]*qr[3]+qr[0]*qr[1]), qr[0]**2+qr[3]**2-qr[1]**2-qr[2]**2]])

    # 4. Get translation.

    qt = ux.reshape(3,1) - R.dot(up.reshape(3,1))

    # 5. Apply transform to P and update T.

    TT = numpy.eye(4)
    TT[0:3,3] = qt.flatten()
    TT[0:3,0:3] = R

    PP = numpy.concatenate((P, numpy.ones((len(P),1))), 1).T

    for i in xrange(len(P)):
      P[i,:] = TT.dot(PP[:,i])[0:3]

    T = TT.dot(T)

  return T


def LoadFromPcdFile(pointsFileName):
  '''TODO'''

  fileHandle = open(pointsFileName, 'r')

  points = []
  for i, line in enumerate(fileHandle):
    if i <= 10: continue # header
    point = tuple([float(x) for x in line.split()])
    points.append(point)

  return array(points)


def LoadFromBinaryPcdFile(pointsFileName):
  '''TODO'''

  fileId = open(pointsFileName, 'rb')

  while True:
    line = fileId.readline().strip()
    if line.startswith('POINTS'):
      nPoints = int(line[7:])
    if line.startswith('DATA'): break

  buf = fileId.read(nPoints*16) # x,y,z, rgb
  cloud = numpy.fromstring(buf, dtype=numpy.float32)
  cloud = numpy.reshape(cloud, (nPoints,4))
  fileId.close()
  return cloud


def Plot(cloud, normals=None, nthNormal=0):

  fig = pyplot.figure()
  ax = fig.add_subplot(111, projection="3d", aspect="equal")

  # points
  x = []; y = []; z = []
  for point in cloud:
    x.append(point[0])
    y.append(point[1])
    z.append(point[2])

  ax.scatter(x, y, z, c='k', s=5, depthshade=False)
  extents = UpdatePlotExtents(x,y,z)

  # normals
  if normals is not None and nthNormal > 0:
    xx=[0,0]; yy=[0,0]; zz=[0,0]
    for i in xrange(len(cloud)):
      if i % nthNormal != 0: continue
      xx[0] = x[i]; xx[1] = x[i] + 0.02 * normals[i][0]
      yy[0] = y[i]; yy[1] = y[i] + 0.02 * normals[i][1]
      zz[0] = z[i]; zz[1] = z[i] + 0.02 * normals[i][2]
      ax.plot(xx, yy, tuple(zz), 'g')

  # bounding cube
  l = (extents[1]-extents[0], extents[3]-extents[2], extents[5]-extents[4])
  c = (extents[0]+l[0]/2.0, extents[2]+l[1]/2.0, extents[4]+l[2]/2.0)
  d = 1.10*max(l) / 2.0

  ax.plot((c[0]+d, c[0]+d, c[0]+d, c[0]+d, c[0]-d, c[0]-d, c[0]-d, c[0]-d), \
          (c[1]+d, c[1]+d, c[1]-d, c[1]-d, c[1]+d, c[1]+d, c[1]-d, c[1]-d), \
          (c[2]+d, c[2]-d, c[2]+d, c[2]-d, c[2]+d, c[2]-d, c[2]+d, c[2]-d), \
           c='k', linewidth=0)

  # labels
  ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
  ax.set_title("Point cloud with {} points.".format(len(cloud)))

  pyplot.show(block=True)


def RemoveOutliers(cloud, outlierBall=0.025, outlierCount=50):
  '''TODO'''

  fullCloud = cloud; cloud = []
  tree = cKDTree(fullCloud)
  indicesList = tree.query_ball_point(fullCloud, outlierBall)
  for i, indicies in enumerate(indicesList):
    if len(indicies) >= outlierCount:
      cloud.append(fullCloud[i])
  return array(cloud)


def SaveAsMat(normals, filename):
  '''
  Saves point cloud data to a mat file.

  @type data: nx3 numpy array
  @param data: Array 3d of points.
  @type fileName: string
  @param fileName: Full name of new file to create and save data to.
  '''

  d = {'normals': normals}
  scipy.io.savemat(filename, d)


def SaveAsPcd(data, fileName):
  '''
  Saves point cloud data to a pcd file.

  @type data: nx3 numpy array
  @param data: Array 3d of points.
  @type fileName: string
  @param fileName: Full name of new file to create and save data to.
  '''

  pcdFileHandle = open(fileName, 'w')

  pcdFileHandle.write('# .PCD v0.7 - Point Cloud Data file format\n')
  pcdFileHandle.write('VERSION 0.7\n')
  pcdFileHandle.write('FIELDS x y z\n')
  pcdFileHandle.write('SIZE 4 4 4\n')
  pcdFileHandle.write('TYPE F F F\n')
  pcdFileHandle.write('COUNT 1 1 1\n')
  pcdFileHandle.write('WIDTH ' + str(len(data)) + '\n')
  pcdFileHandle.write('HEIGHT 1\n')
  pcdFileHandle.write('POINTS ' + str(len(data)) + '\n')
  pcdFileHandle.write('DATA ascii\n')

  for point in data:
    for value in point:
      pcdFileHandle.write(str(value) + ' ')
    pcdFileHandle.write('\n')

  pcdFileHandle.close()


def SaveAsPcdWithNormals(points, normals, fileName):
  '''
  Saves point cloud data to a pcd file.

  @type data: nx3 numpy array
  @param data: Array 3d of points.
  @type fileName: string
  @param fileName: Full name of new file to create and save data to.
  '''

  pcdFileHandle = open(fileName, 'w')

  pcdFileHandle.write('# .PCD v0.7 - Point Cloud Data file format\n')
  pcdFileHandle.write('VERSION 0.7\n')
  pcdFileHandle.write('FIELDS x y z normal_x normal_y normal_z\n')
  pcdFileHandle.write('SIZE 4 4 4 4 4 4\n')
  pcdFileHandle.write('TYPE F F F F F F\n')
  pcdFileHandle.write('COUNT 1 1 1 1 1 1\n')
  pcdFileHandle.write('WIDTH ' + str(len(points)) + '\n')
  pcdFileHandle.write('HEIGHT 1\n')
  pcdFileHandle.write('POINTS ' + str(len(points)) + '\n')
  pcdFileHandle.write('DATA ascii\n')

  for i, point in enumerate(points):
    for value in point:
      pcdFileHandle.write(str(value) + ' ')
    for value in normals[i]:
      pcdFileHandle.write(str(value) + ' ')
    pcdFileHandle.write('\n')

  pcdFileHandle.close()


def TrimCloudToBall(ballCenter, ballRadius, cloud, normals=None):
  '''TODO'''

  cloudTree = cKDTree(cloud)
  ballCloudIndices = cloudTree.query_ball_point(ballCenter, ballRadius)
  if normals is None: return cloud[ballCloudIndices]
  return cloud[ballCloudIndices], normals[ballCloudIndices]


def TrimCloudToBallIndices(cloud, ballCenter, ballRadius):
  '''TODO'''

  cloudTree = cKDTree(cloud)
  ballCloudIndices = cloudTree.query_ball_point(ballCenter, ballRadius)
  return ballCloudIndices


def Transform(cloud, T, isPosition=True):
  '''Apply the homogeneous transform T to the point cloud. Use isPosition=False if transforming unit vectors.'''

  n = cloud.shape[0]
  cloud = cloud.T
  augment = ones((1, n)) if isPosition else zeros((1, n))
  cloud = concatenate((cloud, augment), axis=0)
  cloud = dot(T, cloud)
  cloud = cloud[0:3, :].T
  return cloud


def UpdatePlotExtents(x, y, z, extents=None):
  '''Extends the current extents in a plot by the given values.

  - Input x: List of x-coordiantes.
  - Input y: List of y-coordinates.
  - Input z: Lizt of z-coor  view_planning_proportional.SaveReachablePoints("december-reachable-points.mat")dinates.
  - Input extents: Extents of all other points in the plot as (minX, maxX, minY, maxY, minZ, maxZ).
  - Returns newExtents: The max/min of existing extents with the input coordinates.
  '''

  x = copy(x); y = copy(y); z = copy(z)

  if type(x) == type(numpy.array([])):
    x = x.flatten().tolist()
    y = y.flatten().tolist()
    z = z.flatten().tolist()

  if extents != None:
    x.append(extents[0]); x.append(extents[1])
    y.append(extents[2]); y.append(extents[3])
    z.append(extents[4]); z.append(extents[5])

  extents = (min(x),max(x), min(y),max(y), min(z),max(z))

  return extents


def Voxelize(cloud, voxelSize=0.003):
  '''TODO'''

  cloud = ascontiguousarray(cloud)
  cloud = cloud.astype('float32')

  pclCloud = pcl.PointCloud()
  pclCloud.from_array(cloud)
  voxFilter = pclCloud.make_voxel_grid_filter()
  voxFilter.set_leaf_size(voxelSize, voxelSize, voxelSize)
  pclCloud = voxFilter.filter()

  cloud = pclCloud.to_array()

  return cloud