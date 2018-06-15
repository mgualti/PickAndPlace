function PlotPaperResults()

    %% Parameters
    
    resultsFileName = '../../notebook/results/2017-05-02-attempt1-results.mat';
    scenarioId = 'Mugs Single';

    %% Load

    close('all');

    if exist(resultsFileName, 'file')
        load(resultsFileName);
    end

    %% Display Test Results
    
    if ~exist(resultsFileName, 'file')
        return
    end

    %% Plot Average Reward
    
    if exist('averageReward', 'var')
        figure;
        plot(averageReward, '-x', 'linewidth', 2);
        grid('on');
        xlabel('Training Round', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Average Correct Placements', 'FontSize', 12, 'FontWeight', 'bold');
        title(scenarioId);
    end
    
    %% Plot Average Return and Average Correct

    figure; hold('on');
    avgGoodTempPlaceCount = avgGoodTempPlaceCount(1:100);
    avgBadTempPlaceCount = avgBadTempPlaceCount(1:100);
    avgGoodFinalPlaceCount = avgGoodFinalPlaceCount(1:100);
    avgBadFinalPlaceCount = avgBadFinalPlaceCount(1:100);
    plot(avgGoodTempPlaceCount ./ (avgGoodTempPlaceCount + ...
        avgBadTempPlaceCount), 'linewidth', 2);
    plot(avgGoodFinalPlaceCount ./ (avgGoodFinalPlaceCount + ...
        avgBadFinalPlaceCount), 'linewidth', 2);
    grid('on');
    xlabel('Training Round', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Average Correct', 'FontSize', 12, 'FontWeight', 'bold');
    legend('Temp', 'Final'); title('Proportion of Correct Placements');

    %% Plot Episode Length

    figure; hold('on');
    plot(avgGoodTempPlaceCount + avgBadTempPlaceCount + ...
        avgGoodFinalPlaceCount + avgBadFinalPlaceCount, 'linewidth', 2);
    plot(avgGoodTempPlaceCount + avgBadTempPlaceCount, 'linewidth', 2);
    plot(avgGoodFinalPlaceCount + avgBadFinalPlaceCount, 'linewidth', 2);
    grid('on');
    xlabel('Training Round', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Average Count', 'FontSize', 12, 'FontWeight', 'bold');
    legend('Total Places', 'Temporary Places', 'Final Places');
    title('Average Number of Places per Episode');

    %% Plot Actions
    
    figure;
    rows = [4,8,16,32,64,128];
    bar(rows, placeHistograms(rows,:),'hist');
    set(gca,'XScale','log')
    title('Place Action Counts');
    xlabel('Training Round', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Action Count', 'FontSize', 12, 'FontWeight', 'bold');
    xlim([0, 160]);

end