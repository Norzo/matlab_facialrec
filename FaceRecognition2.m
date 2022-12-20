function outputLabel=FaceRecognition2(trainPath, testPath)


%% Fetch images and image names
trainList=ls(trainPath);
labelImgSet=trainList(3:end,:); % the folder names are the label
% Extract HOG from one sample image to know how many features will be
% present
imgName=ls([trainPath, trainList(3,:),'\*.jpg']);
tmpImg = imread([trainPath, trainList(3,:), '\', imgName]);
tmpImg = rgb2gray(tmpImg);
cellSize = [128 128];
[hog_128x128, vis128x128] = extractHOGFeatures(tmpImg, 'CellSize', cellSize);
hogFeatSize = length(hog_128x128);
trainHOGFeats = zeros(length(labelImgSet), hogFeatSize, 'single');

%% Extract HOG Features from each training image
for i=3:length(trainList) - 2
    imgName=ls([trainPath, trainList(i,:),'\*.jpg']);
    tmpImg = imread([trainPath, trainList(i,:), '\', imgName]);
    tmpImg = rgb2gray(tmpImg);
    trainHOGFeats(i - 2, :) = extractHOGFeatures(tmpImg, 'CellSize', cellSize);
end
% Create a classifier dependent on the HOG Features and the labels for the
% training set
classMdl = fitcecoc(trainHOGFeats, labelImgSet, 'Coding', 'onevsall', 'Learners', 'discriminant', 'FitPosterior', 'on');

%% Extract the HOG Features from the test set
testList = ls([testPath, '*.jpg']);
testFeatures = zeros(length(testList), hogFeatSize, 'single');
for i = 1 : length(testList)
   tmpImg = imread([testPath, testList(i, :)]); 
   tmpImg = rgb2gray(tmpImg);
   testFeatures(i, :) = extractHOGFeatures(tmpImg, 'CellSize', cellSize);
end

% Fetch the predicted labels that the classifier is able to identify
predictedLabels = predict(classMdl, testFeatures);

outputLabel = predictedLabels;

end