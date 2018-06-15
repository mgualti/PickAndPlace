%% Parameters

clear;
resultsFileName = '../results.mat';
trainResultsFileName = '../results-bottle_train.mat';
testResultsFileName = '../results-bottle_test.mat';

%% Load

clc; close('all');

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
    disp(['correct=' num2str(mean(trainData.correct)) ...
          ' return=' num2str(mean(trainData.G)) ...
          ' nActions=' num2str(mean(trainData.graspCount+trainData.placeCount))]);
end

if exist('testData', 'var')
    disp('Test:');
    disp(['correct=' num2str(mean(testData.correct)) ...
          ' return=' num2str(mean(testData.G)) ...
          ' nActions=' num2str(mean(testData.graspCount+testData.placeCount))]);
end

if ~exist(resultsFileName, 'file')
    return;
end

%% Plot Average Return and Average Correct

figure;
plot(averageReturn, '-x', 'linewidth', 2);
grid('on');
xlabel('Iteration'); ylabel('Average Return');
title('Average Return');

figure;
plot(averageCorrect, '-x', 'linewidth', 2);
grid('on');
xlabel('Iteration'); ylabel('Average Correct');
title('Average Correct');

%% Plot Episode Length

figure; hold('on');
plot(graspCounts+placeCounts, '-x', 'linewidth', 2);
plot(graspCounts, '-x', 'linewidth', 2);
plot(placeCounts, '-x', 'linewidth', 2);
grid('on');
xlabel('Iteration'); ylabel('Average Count');
legend('Episode Length', 'Grasp Counts', 'Place Counts');
title('Average Number of Actions per Episode');

%% Plot Actions

N = size(placeHistograms,1);
if N > 10
    rows = 1:int32(N/10):N;
else
    rows = 1:N;
end
figure; hold('on');
bar(rows, placeHistograms(rows,:));
title('Place Action Counts');
xlabel('Iteration'); ylabel('Action Count');
xlim([rows(1)-5, rows(end)+5]);

figure; hold('on');
plot(averageGraspsDetected, '-x', 'linewidth', 2);
plot(ones(size(averageGraspsDetected))*mean(averageGraspsDetected), '--', 'linewidth', 2);
grid('on');
xlabel('Iteration'); ylabel('Number of Grasps');
title('Average Number of Grasps Detected');

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
plot(ones(size(iterationTime))*mean(iterationTime), '--', 'linewidth', 2);
grid('on'); xlabel('Iteration'); ylabel('Time (s)');

figure; hold('on');
title('Database Size');
plot(databaseSize, '-x', 'linewidth', 2);
grid('on'); xlabel('Iteration'); ylabel('Number of Entries');