function cylinder = TestFitCylinder()
    
    %% PARAMETERS
    addpath('/home/mgualti/mgualti/PickAndPlace/simulation/matlab')
    addpath('/home/mgualti/mgualti/PickAndPlace/simulation/matlab/gpd2');
    
    k = 7;
    cloudFile = 'test/bottle_clutter_1.mat';
    plotBitmap = [true, false, true, true];
    
    %% RUN TEST
    close('all');
    cloudData = load(cloudFile);
    cloud = cloudData.cloud';
    viewPoints = cloudData.viewPoints';
    viewPointIndices = cloudData.viewPointIndices;
    cylinder = FitCylinder(cloud, viewPoints, viewPointIndices, k, ...
        plotBitmap);

end