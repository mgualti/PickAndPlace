function grasps = TestSampleGrasps()
    
    %% PARAMETERS
    addpath('/home/mgualti/mgualti/PickAndPlace/simulation/matlab')
    addpath('/home/mgualti/mgualti/PickAndPlace/simulation/matlab/gpd2');
    
    cloudFile = 'test/bottle1.mat';
    nSamples = 10;
    minWidth = 0.002;
    maxWidth = 0.10;
    tableUpAxis = 3;
    tablePosition = [0,0,0.002];
    tableFingerLength = 0.05;
    plotBitmap = [true, true, true, true];
    
    %% RUN TEST
    close('all');
    cloudData = load(cloudFile);
    cloud = cloudData.cloud;
    viewPoints = cloudData.viewPoints;
    viewPointIndices = cloudData.viewPointIndices;
    grasps = SampleGrasps(cloud, viewPoints, viewPointIndices, nSamples, ...
        minWidth, maxWidth, tablePosition, tableUpAxis, tableFingerLength, ...
        plotBitmap);

end