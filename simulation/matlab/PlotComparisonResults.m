function PlotComparisonResults()

    %% Parameters
    
    resultsFileName = '../results2.mat';
    trainResultsFileName = '../results-clutter-mug_train-epsilon0.0.mat';
    testResultsFileName = '../results-clutter-mug_test-epsilon0.0.mat';

    %% Load

    close('all'); clc;

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

    if exist('trainData', 'var')
        disp('Train:');
        DisplayTestResults(trainData);
    end
    if exist('testData', 'var')
        disp('Test:');
        DisplayTestResults(testData);
    end
    if ~exist(resultsFileName, 'file')
        return
    end

    %% Plot Average Reward

    figure;
    plot(averageReward, '-x', 'linewidth', 2);
    grid('on');
    xlabel('Iteration'); ylabel('Average Reward');
    title('Average Reward');

    %% Plot Actions

    N = size(placeActionCounts,1);
    if N < 10
        rows = 1:N;
    else
        rows = 1:int32(N/10):N;
    end
    figure; hold('on');
    if N < 2
        bar(placeActionCounts);
    else
        bar(rows, placeActionCounts(rows,:));
    end
    title('Place Action Counts');
    xlabel('Iteration'); ylabel('Action Count');
    xlim([rows(1)-5, rows(end)+5]);

    %% Plot Loss

    if ~isempty(testLoss)

        N = size(trainLoss,1);
        iteration = 1:size(trainLoss,2);
        iteration = iteration*100; % depends on python code

        for idx=1:10:N
            figure; hold('on');
            title(['Train and Test Loss, iteration=' num2str(idx)]);
            plot(iteration, trainLoss(idx,:));
            plot(iteration, testLoss(idx,:));
            legend('Train', 'Test'); grid('on');
            xlabel('Caffe Iteration'); ylabel('Loss');
        end

        % figure; hold('on');
        % title('Train Losses');
        % plot(iteration, trainLoss');
        % grid('on');
        % xlabel('Caffe Iteration'); ylabel('Loss');

        figure; hold('on');
        title('Test Losses');
        plot(iteration, testLoss');
        grid('on');
        xlabel('Caffe Iteration'); ylabel('Loss');

        figure; hold('on');
        title('Average Test Loss');
        plot(mean(testLoss,2), '-x', 'linewidth', 2);
        grid('on'); xlabel('Iteration'); ylabel('Loss');

    end

    %% Plot Run Time and Database 

    figure; hold('on');
    title('Training Run Time');
    plot(iterationTime, '-x', 'linewidth', 2);
    plot(ones(size(iterationTime))*mean(iterationTime), ...
        '--', 'linewidth', 2); grid('on');
    xlabel('Iteration'); ylabel('Time (s)');

    figure; hold('on');
    title('Database Size');
    plot(databaseSize, '-x', 'linewidth', 2);
    grid('on'); xlabel('Iteration'); ylabel('Number of Entries');

end

function DisplayTestResults(data)
    nAttempts = length(data.Return);
    totalReturn = sum(data.Return);
    disp(['  ' num2str(totalReturn) '/' num2str(nAttempts) '=']);
    disp(['  ' num2str(totalReturn/nAttempts)]);
end