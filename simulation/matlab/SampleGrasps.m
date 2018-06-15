% Samples and filters grasps in a point cloud.
%  Input cloud: 3xn point cloud for grasp detection.
%  Input viewPoints: 3xm points for each m viewpoints the cloud was taken
%    from.
%  Input viewPointIndices: Index into the point cloud where each of the
%    view points starts.
%  Input nSamples: Scalar integer number of points in cloud to sample.
%  Input minWidth: Filters grasps with width less than this.
%  Input maxWidth: Filters grasps with width greater than this.
%  Input tablePosition: Filters grasps approaching from under the table and
%    grasps that are partially under a table specified by this position.
%    Also does not sample points on or below this table.
%  Input tableUpAxis: The axis in {1,2,3,-1,-2,-3} (i.e. {x,y,z,-x,-y,-z})
%    that indicates the table up direction.
%  Input tableFingerLength: When filtering grasps below the table, it may
%    be desirable to use a different finger length than the finger length
%    used for sampling the grasps. This is a scalar floating point number
%    indicating this length.
%  Input plotMode: 1x4 bitmap for which plots to show.
%  Output: grasps: Data structure with grasp poses and grasp images.
function grasps = SampleGrasps(cloud, viewPoints, viewPointIndices, ...
    nSamples, minWidth, maxWidth, tablePosition, tableUpAxis, ...
    tableFingerLength, plotBitmap)
    
    % 1. Setup.
    
    if plotBitmap(1)
        tic;
        disp(['Sampling grasps with ' num2str(nSamples) ...
        ' samples in cloud with ' num2str(size(cloud,2)) ' points.']);
    end
    
    if numel(viewPointIndices) > 1 && size(viewPointIndices, 2) == 1
        viewPointIndices = viewPointIndices';
    end
    
    % 2. Parameters.
    
    curveAxisProb = 0.5;
    [handparams, imageparams] = getParams();
    
    % adjust parameters for larger image used in distinguishing object pose
    imageparams.imageOD = 0.10;
    imageparams.imageDepth = 0.10;
    imageparams.imageHeight = 0.10;
%     imageparams.imageOD = 0.20;
%     imageparams.imageDepth = 0.20;
%     imageparams.imageHeight = 0.20;
    imageparams.imageSize = 60;
    
    % 3. Point cloud processing.
    
    p = clsPts();
    p = p.addCamPos(viewPoints);
    camsource = false(size(viewPoints, 2), size(cloud, 2));
    viewPointIndices = [viewPointIndices, size(cloud, 2)+1];
    for idx=1:size(viewPoints, 2)
        camsource(idx, viewPointIndices(idx):viewPointIndices(idx+1)-1) = true;
    end
    p = p.addPts(cloud, camsource);
    p = p.voxelize(0.002);
    
    % 4. Get grasp candidates.
    
    objUID = 'DetectGraspsObj';
    hands = clsPtsHands(p, handparams, objUID, (0), 0);
    hands = hands.subSampleTable(nSamples, tablePosition, tableUpAxis);
    hands = hands.getGraspCandidates(p, curveAxisProb);
    
    % plot grasp candidates
    if plotBitmap(2)
        figure; hold('on');
        p.plot('k');
        hands.plotHandList(hands.getAllHands());
        lightangle(-45, 30);
        title('Grasp Candidates');
    end
    
    % 5. Filter grasps outside the width limits and grasps below the table.
    
    handIdxs = hands.getAllHands();
    filterByWidth = false(size(handIdxs,1), 1);
    filterByTable = false(size(handIdxs,1), 1);
    
    for idx=1:size(handIdxs, 1)
        
        % get this hand
        handIdx = handIdxs(idx, 1);
        orientIdx = handIdxs(idx, 2);
        hs = hands.handSetList{handIdx};
        hand = hs.Hands{orientIdx};
        
        % filter by width
        width = hand.width;
        if hand.width < minWidth || hand.width > maxWidth
            filterByWidth(idx) = true;
            continue;
        end
        
        % filter below table
        up = abs(tableUpAxis);
        [~, bottom, ~] = hand.getHandParameters();
        approach = hand.F(:,1); binormal = hand.F(:,2);
        aboveGrasp = bottom - tableFingerLength*approach;
        leftFingerBottom = bottom - 0.5*width*binormal;
        rightFingerBottom = bottom + 0.5*width*binormal;
        leftFingerTop = leftFingerBottom + tableFingerLength*approach;
        rightFingerTop = rightFingerBottom + tableFingerLength*approach;
        positions = [leftFingerTop(up), leftFingerBottom(up), ...
            rightFingerTop(up), rightFingerBottom(up), aboveGrasp(up)];
        if tableUpAxis > 0
            filterByTable(idx) = min(positions) < tablePosition(up);
        else
            filterByTable(idx) = max(positions) > tablePosition(up);
        end
    end
    
    keepHand = ~filterByWidth & ~filterByTable;
    hands = hands.importAntipodalLabels(handIdxs, keepHand);
    handIdxs = hands.getAntipodalHandList();
    hands = hands.pruneHandList(handIdxs);
    handIdxs = hands.getAllHands();
    
    if plotBitmap(1)
        nOriginalGrasps = num2str(length(keepHand));
        nFilteredByWidth = num2str(sum(filterByWidth));
        nFilteredByTable = num2str(sum(filterByTable));
        nGrasps = num2str(size(handIdxs,1));
        nDoubledGrasps = num2str(2*size(handIdxs,1));
        disp(['Originally ' nOriginalGrasps ', filtered ' ...
            nFilteredByWidth ' by width and ' nFilteredByTable ...
            ' below table.']);
        disp(['Now have ' nGrasps ' and will double to ' nDoubledGrasps '.']); 
    end
    
    % plot grasps remaining after filter
    if plotBitmap(3)
        figure; hold('on'); p.plot('k');
        hands.plotHandList(handIdxs);
        lightangle(-45, 30);
        title('Filtered Grasps');
    end
    
    if size(handIdxs,1) == 0
        grasps = {};
        return;
    end
    
    % 6. Compute larger grasp images.
    
    hands = hands.setImageParams(imageparams);
    hands = hands.calculateImages(p);
    handsFlipped = FlipHands(hands);
    handsFlipped = handsFlipped.calculateImages(p);
    
    % plot grasps with images
    if plotBitmap(4)
        for idx=1:size(handIdxs,1)
            handIdx = handIdxs(idx, 1);
            orientIdx = handIdxs(idx, 2);
            hs = hands.handSetList{handIdx};
            hsf = handsFlipped.handSetList{handIdx};
            figure; hold('on'); title(['Hand ' num2str(idx)]);
            p.plot('k'); hs.plotHand('b',orientIdx,0);
            PlotImages(hs.images{orientIdx}, ['Hand ' num2str(idx)]); 
            PlotImages(hsf.images{orientIdx}, ['Hand ' num2str(idx) ' Flipped']);
        end
    end
    
    % 7. Package result.
    
    imageIndices = [1,2,3,4,6,7,8,9,11,12,13,14]; % remove occlusion channels
    grasps = cell(1,2*size(handIdxs,1));
    for idx=1:size(handIdxs, 1)
        
        handIdx = handIdxs(idx, 1);
        orientIdx = handIdxs(idx, 2);
        hs = hands.handSetList{handIdx};
        hsf = handsFlipped.handSetList{handIdx};
        hand = hs.Hands{orientIdx};
        handf = hsf.Hands{orientIdx};
        
        grasps{2*idx-1} = PackageHand(hand, 0, ...
            hs.images{orientIdx}(:,:,imageIndices));
        grasps{2*idx} = PackageHand(handf, 0, ...
            hsf.images{orientIdx}(:,:,imageIndices));
    end
    
    if plotBitmap(1)
        toc
    end
end

% Gets a clsPtsHands that are flipped 180 degrees about the hand axis.
function hands = FlipHands(hands)
    for idx=1:length(hands.F)
        F = hands.F{idx};
        hands.F{idx} = [F(:,1),-F(:,2),-F(:,3)];
        F = hands.handSetList{idx}.F;
        hands.handSetList{idx}.F = [F(:,1),-F(:,2),-F(:,3)];
    end
    handList = hands.getAllHands();
    for idx=1:size(handList,1)
        handIdx = handList(idx, 1);
        orientIdx = handList(idx, 2);
        F = hands.handSetList{handIdx}.Hands{orientIdx}.F;
        center = hands.handSetList{handIdx}.Hands{orientIdx}.center;
        hands.handSetList{handIdx}.Hands{orientIdx}.F = [F(:,1),-F(:,2),-F(:,3)];
        hands.handSetList{handIdx}.Hands{orientIdx}.center = -center;
    end
end

% Puts hand into the structure that the calling program will recognize.
function grasp = PackageHand(hand, score, image)
    grasp = struct();
    [grasp.center, grasp.bottom, grasp.top] = hand.getHandParameters();
    grasp.approach = hand.F(:,1);
    grasp.axis = hand.F(:,3);
    grasp.binormal = hand.F(:,2);
    grasp.height = hand.handparams.handHeight;
    grasp.width = hand.width;
    grasp.score = score;
    % faster if image is returned as ASCII string
    grasp.imageSize = size(image);
    imageL = image(:);
    gt = imageL > 127;
    imageL(gt) = imageL(gt) - 128;
    grasp.imageH = char(uint8(gt));
    grasp.imageL = char(imageL);
end

% Plot the 12 grasp images.
function PlotImages(img, titleText)
    figure; suptitle(titleText);
    subplot(2,3,1); imshow(img(:,:,1:3));
    subplot(2,3,2); imshow(img(:,:,6:8));
    subplot(2,3,3); imshow(img(:,:,11:13));
    subplot(2,3,4); imshow(img(:,:,4));
    subplot(2,3,5); imshow(img(:,:,9));
    subplot(2,3,6); imshow(img(:,:,14));
end