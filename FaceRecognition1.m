function outputLabel=FaceRecognition1(trainPath, testPath)

%% Fetch the images and the labels of the image
% Train images will be the images that are compared to the test images
trainList=ls(trainPath);
labelImgSet=trainList(3:end,:); % the folder names are the labels

% Define cell size and block size for our HOG Features
cellSize = [32 32];
blockSize = [8 8];

% Extract an HOG example to know the relevant size for our arrays
imgName=ls([trainPath, trainList(3,:),'\*.jpg']);
tmpImg = imread([trainPath, trainList(3,:), '\', imgName]);
tmpImg = rgb2gray(tmpImg);
hogExam = extractHOGFeatures(tmpImg, 'CellSize', cellSize, 'BlockSize', blockSize);
hogFeatSize = length(hogExam);
trainHOGFeat = zeros(length(trainList)-2, hogFeatSize, 'single');

% Extract the HOG Features from our training images one by one
for i=3:length(trainList) - 2
    imgName=ls([trainPath, trainList(i,:),'\*.jpg']);
    tmpImg = imread([trainPath, trainList(i,:), '\', imgName]);
    tmpImg = rgb2gray(tmpImg);
    trainHOGFeat(i - 2, :) = extractHOGFeatures(tmpImg, 'CellSize', cellSize, 'BlockSize', blockSize);
end
%% Extract the HOG Features from our test images and perform the ED Calculation
outputLabel = [];

testList = ls([testPath, '*.jpg']);
testHOGFeat = zeros(length(testList), hogFeatSize, 'single');

for i = 1 : length(testList)
    edScore = []; % Temp array to score the relevant ED scores for our calculations, reset at every iteration
    tmpImg = imread([testPath, testList(i, :)]);
    testHOGFeat(i, :) = extractHOGFeatures(tmpImg, 'CellSize', cellSize, 'BlockSize', blockSize);
    % Inner for loop where we calculate the ED score straight after we've
    % extracted the HOG features, comparing our test image to all the
    % training images
    for j = 1 : length(labelImgSet)
        edScore = [edScore;sqrt(sum((trainHOGFeat(j, :) - testHOGFeat(i, :)) .^ 2))];
    end
    % Find the index of the lowest ED score, which will be the predicted
    % training image for our test image
    lowestEDIndx = find(edScore==min(edScore));
    outputLabel = [outputLabel;labelImgSet(lowestEDIndx(1), :)];
end
end