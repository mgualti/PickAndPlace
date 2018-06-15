function PlotRegraspResults()

    %% Parameters
    
    resultsFileName = '../results.mat';
    trainResultsFileName = '../results-mug_train.mat';
    testResultsFileName = '../results-mug_test.mat';

    %% Load

    close('all');

    if exist(resultsFileName, 'file')
        load(resultsFileName);
    end

    if exist(trainResultsFileName, 'file')
        trainData = load(trainResultsFileName);
    end
    if exist(testResultsFileName, 'file')
        testData = load(testResultsFileName);
    end

    %% Display Test Results

    if exist(trainResultsFileName, 'file')
        disp('Train:');
        DisplayTestResults(trainData);
    end
    if exist(testResultsFileName, 'file')
        disp('Test:');
        DisplayTestResults(testData);
    end
    if ~exist(resultsFileName, 'file')
        return;
    end

    %% Plot Average Return and Average Correct

    figure;
    plot(avgReturn, '-x', 'linewidth', 2);
    grid('on');
    xlabel('Iteration'); ylabel('Average Return');
    title('Average Return');

    figure; hold('on');
    plot(avgGoodTempPlaceCount ./ (avgGoodTempPlaceCount + ...
        avgBadTempPlaceCount), '-x', 'linewidth', 2);
    plot(avgGoodFinalPlaceCount ./ (avgGoodFinalPlaceCount + ...
        avgBadFinalPlaceCount), '-x', 'linewidth', 2);
    grid('on'); xlabel('Iteration'); ylabel('Average Correct');
    legend('Temp', 'Final'); title('Proportion of Correct Placements');

    %% Plot Episode Length

    figure; hold('on');
    plot(avgGoodTempPlaceCount + avgBadTempPlaceCount + ...
        avgGoodFinalPlaceCount + avgBadFinalPlaceCount, '-x', 'linewidth', 2);
    plot(avgGoodTempPlaceCount + avgBadTempPlaceCount, '-x', 'linewidth', 2);
    plot(avgGoodFinalPlaceCount + avgBadFinalPlaceCount, '-x', 'linewidth', 2);
    grid('on');
    xlabel('Iteration'); ylabel('Average Count');
    legend('Total Places', 'Temporary Places', 'Final Places');
    title('Average Number of Places per Episode');

    %% Plot Actions

    N = size(placeHistograms,1);
    if N < 10
        rows = 1:N;
    else
        rows = 1:int32(N/10):N;
    end
    figure; hold('on');
    if N < 2
        bar(placeHistograms);
    else
        bar(rows, placeHistograms(rows,:));
    end
    title('Place Action Counts');
    xlabel('Iteration'); ylabel('Action Count');
    xlim([rows(1)-5, rows(end)+5]);

    figure; hold('on');
    plot(avgGraspsDetected, '-x', 'linewidth', 2);
    plot(ones(size(avgGraspsDetected))*mean(avgGraspsDetected), '--', 'linewidth', 2);
    avgTopGraspsDetected(isnan(avgTopGraspsDetected)) = 0;
    plot(avgTopGraspsDetected, '-x', 'linewidth', 2);
    plot(ones(size(avgTopGraspsDetected))*mean(avgTopGraspsDetected), '--', 'linewidth', 2);
    grid('on'); legend('grasps', '\mu grasps', 'top grasps', '\mu top grasps');
    xlabel('Iteration'); ylabel('Number of Grasps');
    title('Average Number of Grasps Detected');

    %% Plot Loss

    if ~isempty(testLoss)

%         N = size(trainLoss,1);
%         iteration = 1:size(trainLoss,2);
%         iteration = iteration*100; % depends on python code

%         for idx=1:10:N
%             figure; hold('on');
%             title(['Train and Test Loss, iteration=' num2str(idx)]);
%             plot(iteration, trainLoss(idx,:));
%             plot(iteration, testLoss(idx,:));
%             legend('Train', 'Test'); grid('on');
%             xlabel('Caffe Iteration'); ylabel('Loss');
%         end

        % figure; hold('on');
        % title('Train Losses');
        % plot(iteration, trainLoss');
        % grid('on');
        % xlabel('Caffe Iteration'); ylabel('Loss');

%         figure; hold('on');
%         title('Test Losses');
%         plot(iteration, testLoss');
%         grid('on');
%         xlabel('Caffe Iteration'); ylabel('Loss');

        figure; hold('on');
        title('Average Test Loss');
        plot(mean(testLoss,2), '-x', 'linewidth', 2);
        grid('on'); xlabel('Iteration'); ylabel('Loss');

    end

    %% Plot Run Time and Database 

    figure; hold('on');
    title('Training Run Time');
    plot(iterationTime, '-x', 'linewidth', 2);
    plot(ones(size(iterationTime))*mean(iterationTime), '--', 'linewidth', 2);
    grid('on'); xlabel('Iteration'); ylabel('Time (s)');

    figure; hold('on');
    title('Database Size');
    plot(databaseSize, '-x', 'linewidth', 2);
    grid('on'); xlabel('Iteration'); ylabel('Number of Entries');
    
end

function DisplayTestResults(data)
    disp(['correct=' num2str(mean(data.goodFinalPlaceCount)) ...
        ' return=' num2str(mean(data.Return)) ...
        ' nPlaces=' num2str(mean(data.goodTempPlaceCount + ...
        data.badTempPlaceCount + data.goodFinalPlaceCount + ...
        data.badFinalPlaceCount))]);
end