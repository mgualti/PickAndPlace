function PlotAverageLearningCurves()

    %% Parameters
    
    dirPrefixes = {'../../notebook/results/2017-05-22-mug_single/', ...
                   '../../notebook/results/2017-05-22-mug_clutter/'};
    filePrefix = 'results';
    
    scenarioId = 'Mugs';
    nTrainingRounds = 70;
    nSimulations = 10;
    
    resultFileNames = cell(1, 2*nSimulations);
    for idx = 0 : 1
        dirPrefix = dirPrefixes{idx+1};
        for jdx = 0 : nSimulations-1
            resultFileNames{idx*nSimulations+jdx+1} = ...
                [dirPrefix filePrefix num2str(jdx) '.mat'];
        end
    end

    %% Load

    close('all');
    
    resultData = cell(1, 2*nSimulations);
    for idx=1:length(resultFileNames)
        data = load(resultFileNames{idx});
        resultData{idx} = data;
    end

    %% Plot Average Reward
    
    GS = zeros(nSimulations, nTrainingRounds);
    for idx=1:nSimulations
        GS(idx, :) = resultData{idx}.averageReward;
    end
    
    GC = zeros(nSimulations, nTrainingRounds);
    for idx=1:nSimulations
        GC(idx, :) = resultData{nSimulations+idx}.averageReward;
    end
    
    
    figure; hold('on');
    set(gca, 'fontsize', 10, 'fontweight', 'bold');
    mu = mean(GS,1); sigma = std(GS,1);
    h1 = plot(mu,  'b', 'linewidth', 4);
    h2 = plot(mu+sigma, 'c', 'linewidth', 1);
    h3 = plot(mu-sigma, 'c', 'linewidth', 1);
    mu = mean(GC,1); sigma = std(GC,1);
    h4 = plot(mu, 'r', 'linewidth', 4)
    h5 = plot(mu+sigma, 'm', 'linewidth', 1);
    h6 = plot(mu-sigma, 'm', 'linewidth', 1);
    grid('on');
    xlabel('Training Round', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Average Correct Placements', 'FontSize', 12, 'FontWeight', 'bold');
    legendHandle = legend([h1,h2,h4,h5], ...
        '\mu (single)', '\mu \pm \sigma (single)', ...
        '\mu (clutter)', '\mu \pm \sigma (clutter)');
    set(legendHandle, 'FontSize', 12, 'FontWeight', 'bold', 'Location', 'northwest');
    title(scenarioId);
    

end