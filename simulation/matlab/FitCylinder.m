% Samples and filters grasps in a point cloud.
%  Input cloud: nx3 point cloud for cylinder fitting.
%  Input viewPoints: mx3 points for each m viewpoints the cloud was taken
%    from.
%  Input viewPointIndices: Index into the point cloud where each of the
%    view points starts.
%  Input k: Number of clusters to partition data into before attempting to
%    segment to most isolated partition.
%  Input plotMode: 1x4 bitmap for which plots to show.
%  Output cylinder: Data structure with cylinder pose, radius, and height.
function cylinder = FitCylinder(cloud, viewPoints, viewPointIndices, ...
    k, plotBitmap)
    
    % Setup.
    
    n = size(cloud, 1);
    m = size(viewPoints, 1);
    
    if plotBitmap(1)
        tic;
        disp(['Fitting clyinder in cloud with ' num2str(n) ' points.']);
    end
    
    % Parameters.
    
    normalsNeighbors = 30;
    normalLength = 0.02;
    maxPointToCylinderDist = 0.01;
    
    % Point cloud processing.
    
    normals = ComputeNormals(cloud, viewPoints, viewPointIndices, normalsNeighbors);
    
    if plotBitmap(2)
        figure; hold('on');
        plot3(cloud(:,1), cloud(:,2), cloud(:,3), 'k.');
        for idx = 1:size(cloud,1)
            plot3([cloud(idx,1);cloud(idx,1)+normalLength*normals(idx,1)], ...
                  [cloud(idx,2);cloud(idx,2)+normalLength*normals(idx,2)], ...
                  [cloud(idx,3);cloud(idx,3)+normalLength*normals(idx,3)], ...
                 'b-');
        end
        xlabel('X'); ylabel('Y'); zlabel('Z');
        grid('on'); axis('equal');
    end
    
    % Cluster points.
    
    if k > 1
        [clusterIdxs, centroids] = kmeans(cloud, k);
        
        maxDist = -Inf; maxDistIdx = 0;
        for idx=1:size(centroids, 1)
            
            minDist = Inf;
            for jdx=1:size(centroids, 1)
                if idx==jdx, continue; end
                d = norm(centroids(idx, :)-centroids(jdx, :));
                if d < minDist, minDist = d; end
            end
            
            if minDist > maxDist
                maxDist = minDist;
                maxDistIdx = idx;
            end    
        end
        
        wantIdxs = clusterIdxs==maxDistIdx;
    else
        centroids = mean(cloud);
        wantIdxs = true(size(cloud, 1), 1);
    end
    
    if plotBitmap(3)
        figure; hold('on');
        plot3(cloud(:,1), cloud(:,2), cloud(:,3), 'k.');
        plot3(centroids(:,1), centroids(:,2), centroids(:,3), ...
            'rx', 'MarkerSize', 20);
        xlabel('X'); ylabel('Y'); zlabel('Z');
        grid('on'); axis('equal');
    end
    
    cloud = cloud(wantIdxs,:);
    normals = normals(wantIdxs,:);
    
    % Fit cylinder.
    
    p = pointCloud(cloud, 'Normal', normals);
    
    %roi = [-Inf,Inf;-Inf,Inf;0.002,Inf];
    %sampleIndices = p.findPointsInROI(roi);
    %c = pcfitcylinder(p, maxPointToCylinderDist, 'SampleIndices', sampleIndices);
    
    c = pcfitcylinder(p, maxPointToCylinderDist);
    
    if plotBitmap(4)
        figure; hold('on');
        plot3(cloud(:,1), cloud(:,2), cloud(:,3), 'k.');
        ctr = c.Center;
        axs = c.Orientation / norm(c.Orientation);
        len = c.Height / 2;
        plot3([ctr(1);ctr(1)+len*axs(1)], ...
              [ctr(2);ctr(2)+len*axs(2)], ...
              [ctr(3);ctr(3)+len*axs(3)], 'b-');
        c.plot()
        xlabel('X'); ylabel('Y'); zlabel('Z');
        grid('on'); axis('equal');
    end
    
    % Package result.
    
    cylinder = struct();
    cylinder.center = c.Center;
    cylinder.axis = c.Orientation / norm(c.Orientation);
    cylinder.radius = c.Radius;
    cylinder.height = c.Height;
    
    if plotBitmap(1)
        toc
    end
end

% Computes normals for point cloud and reverses them towards view point.
function normals = ComputeNormals(cloud, viewPoints, viewPointIndices, kNeighbors)
    
    n = size(cloud,1);
    m = length(viewPointIndices);
    pc = pointCloud(cloud);
    normals = pcnormals(pc, kNeighbors);

    % reverse direction of normals that are not pointing toward at least one camera
    normalsToReverse = ones(size(cloud, 1), 1); % this will be binary
    viewPointIndices = [viewPointIndices, n+1];
    for idx=1:m

        % get pts and normals associated with this cam
        idxThisCam = viewPointIndices(idx):viewPointIndices(idx+1)-1;
        ptsThisCam = cloud(idxThisCam, :);
        normalsThisCam = normals(idxThisCam, :);

        % get vectors from pts back to cam
        posThisCam = viewPoints(idx,:);
        numThisCam = size(ptsThisCam,1);
        pts2Cam = repmat(posThisCam,numThisCam,1) - ptsThisCam;

        % figure out which normals are pointing toward cam
        dotPtsNormals = dot(pts2Cam, normalsThisCam, 2);
        pointing2ThisCam = dotPtsNormals > 0;

        % don't reverse those normals
        normalsToReverse(idxThisCam(pointing2ThisCam)) = 0;
    end

    normals = normals .* repmat((1-2*normalsToReverse),1,3);
end