function grasps = TestDetectGrasps()
    
    %% PARAMETERS
    addpath('/home/mgualti/Programs/caffe/matlab');
    addpath('/home/mgualti/mgualti/PickAndPlace/simulation/matlab')
    addpath('/home/mgualti/mgualti/PickAndPlace/simulation/matlab/gpd2');
    
    cloudFile = 'test/bottle6.mat';
    nSamples = 150;
    scoreThresh = 300;
    minWidth = 0.002;
    maxWidth = 0.085;
    tableUpAxis = 3;
    tablePosition = [0,0,0.002];
    tableFingerLength = 0.05;
    plotBitmap = [true, false, false, false, false];
    
    %% RUN TEST
    close('all');
    cloudData = load(cloudFile);
    cloud = cloudData.cloud;
    viewPoints = cloudData.viewPoints;
    viewPointIndices = cloudData.viewPointIndices;
    grasps = DetectGrasps(cloud, viewPoints, viewPointIndices, ...
        nSamples,scoreThresh, minWidth, maxWidth, tablePosition, ...
        tableUpAxis, tableFingerLength, plotBitmap);

end