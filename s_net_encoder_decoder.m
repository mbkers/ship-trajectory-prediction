% s_net_encoder_decoder.m
% This script creates, trains and tests a recurrent sequence-to-sequence
% encoder-decoder model with attention (by using functions rather than a
% MATLAB layer array, layerGraph or dlnetwork object) for ship trajectory
% prediction.

clear
clc

%% Load data
load("s_data_preprocessing_variables.mat")

%% Preprocess data
% Create arrayDatastore objects from the source and target data
dsXTrain = arrayDatastore(X_train,OutputType="same");
dsTTrain = arrayDatastore(T_train,OutputType="same");
dsXVal = arrayDatastore(X_val,OutputType="same");
dsTVal = arrayDatastore(T_val,OutputType="same");
dsXTest = arrayDatastore(X_test,OutputType="same");
dsTTest = arrayDatastore(T_test,OutputType="same");

% Combine them to create a CombinedDatastore object
dsTrain = combine(dsXTrain,dsTTrain);
dsVal = combine(dsXVal,dsTVal);
dsTest = combine(dsXTest,dsTTest);

%% Initialise model parameters
% Specify parameters for both the encoder and decoder
numFeatures = size(X_train{1},1);
numResponses = size(T_train{1},1);
numHiddenUnits = 32;
dropout = 0.20;

% Initialise encoder model parameters
    % Define the input size
    inputSize = numFeatures;

    % Initialise the learnable parameters for the encoder BiLSTM operation
    % and include them in a structure array:
    % 1) Initialise the input weights with the Glorot initialiser*
    % 2) Initialise the recurrent weights with the orthogonal initialiser*
    % 3) Initialise the bias with the unit forget gate initialiser*
    % *(see the 'Supporting functions' section at the end of the script)
    numOut = 8*numHiddenUnits;
    numIn = inputSize;
    sz = [numOut numIn];

    parameters.encoder.bilstm.InputWeights = initializeGlorot(sz,numOut,numIn);
    parameters.encoder.bilstm.RecurrentWeights = initializeOrthogonal([8*numHiddenUnits numHiddenUnits]);
    parameters.encoder.bilstm.Bias = initializeUnitForgetGate(2*numHiddenUnits);

% Initialise decoder model parameters
    % Define the output size
    outputSize = numResponses;

    % Initialise the weights of the attention mechanism with the Glorot
    % initialiser
    sz = [numHiddenUnits numHiddenUnits];
    numOut = numHiddenUnits;
    numIn = numHiddenUnits;
    parameters.decoder.attention.Weights = initializeGlorot(sz,numOut,numIn);

    % Initialise the learnable parameters for the decoder LSTM operation
    % (same initialisers as the encoder BiLSTM operation)
    sz = [4*numHiddenUnits numFeatures+numHiddenUnits];
    numOut = 4*numHiddenUnits;
    numIn = numFeatures + numHiddenUnits;

    parameters.decoder.lstm.InputWeights = initializeGlorot(sz,numOut,numIn);
    parameters.decoder.lstm.RecurrentWeights = initializeOrthogonal([4*numHiddenUnits numHiddenUnits]);
    parameters.decoder.lstm.Bias = initializeUnitForgetGate(numHiddenUnits);

    % Initialise the learnable parameters for the decoder fully connected
    % operation:
    % 1) Initialise the weights with the Glorot initialiser
    % 2) Initialise the bias with zeros using zeros initialisation*
    % *(see the 'Supporting functions' section at the end of the script)
    sz = [outputSize 2*numHiddenUnits];
    numOut = outputSize;
    numIn = 2*numHiddenUnits;

    parameters.decoder.fc.Weights = initializeGlorot(sz,numOut,numIn);
    parameters.decoder.fc.Bias = initializeZeros([outputSize 1]);

%% Define model function(s)
% The 'modelEncoder' function, provided in the 'Encoder model function'
% section of the script, takes the input data, model parameters, and the
% optional mask used to determine the correct outputs for training, and
% returns the model outputs and the BiLSTM hidden state.

% The 'modelDecoder' function, provided in the 'Decoder model function'
% section of the script, takes the input data, model parameters, the
% context vector, the LSTM initial hidden state, the outputs of the
% encoder, and the dropout probability, and returns the decoder output,
% the updated context vector, the updated LSTM state, and the attention
% scores.

%% Define model loss function
% The 'modelLoss' function, provided in the 'Model loss function' section
% of the script, takes the encoder and decoder model parameters, a
% mini-batch of input data and the padding masks corresponding to the
% input data as well as the dropout probability and returns the loss and
% the gradients of the loss with respect to the learnable parameters in
% the models.

%% Specify training options
% Learn rate (piecewise learning rate schedule)
learnRate = 0.1;
learnRateSchedule = "piecewise";
learnRateDropPeriod = 10; % 100
learnRateDropFactor = 0.5;

% Mini-batch size
miniBatchSize = 128;

% Number of epochs
maxEpochs = 100; % 1000

% Early stopping
validationPatience = Inf;

% L2 regularisation (TOCONSIDER)

% Gradient clipping (TOCONSIDER)

% Specify values for the ADAM optimisation algorithm
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;

%% Train model
% Train the model using a custom training loop.

% Prepare the training and validation data:
% The 'minibatchqueue' function processes and manages mini-batches of data
% for training. For each mini-batch, the 'minibatchqueue' object returns
% four output arguments: the source sequences, the target sequences, the
% lengths of all source sequences in the mini-batch, and the sequence mask
% of the target sequences.
numMiniBatchOutputs = 4;
mbq_train = minibatchqueue(dsTrain, ...
    numMiniBatchOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@(x,t) preprocessMiniBatch(x,t,inputSize,outputSize));

mbq_val = minibatchqueue(dsVal, ...
    numMiniBatchOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@(x,t) preprocessMiniBatch(x,t,inputSize,outputSize));

% Initialise the parameters for the 'adamupdate' function
trailingAvg = [];
trailingAvgSq = [];

% Calculate the total number of iterations for the training progress monitor
numObservationsTrain = numel(X_train);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
maxIterations = maxEpochs * numIterationsPerEpoch;

% Calculate the validation frequency (evenly distribute)
numObservationsVal = numel(X_val);
numIterationsPerEpochVal = ceil(numObservationsVal / miniBatchSize);
validationFrequency = ceil(numIterationsPerEpoch / numIterationsPerEpochVal);

% Initialise the variables for early stopping
earlyStop = false; % Initialise the early stopping flag
if isfinite(validationPatience)
    validationLosses = inf(1,validationPatience); % Losses to compare
end

% Initialise tracking for the best model
bestValidationLoss = inf;
bestModelParameters = parameters;

% Initialise and prepare the training progress monitor
monitor = trainingProgressMonitor;

monitor.Metrics = ["TrainingLoss","ValidationLoss"];

groupSubPlot(monitor,"Loss",["TrainingLoss","ValidationLoss"]);

monitor.Info = ["LearningRate","Epoch","Iteration","ExecutionEnvironment"];

monitor.XLabel = "Iteration";
monitor.Status = "Configuring";
monitor.Progress = 0;

% Select the execution environment
executionEnvironment = "auto";
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    updateInfo(monitor,ExecutionEnvironment="GPU");
else
    updateInfo(monitor,ExecutionEnvironment="CPU");
end

% Train the model using a custom training loop. For each epoch:
% 1) Shuffle the training data, reset the validation data and loop over
% mini-batches of training data. For each mini-batch:
    % 1.1) Evaluate the model loss and gradients
    % 1.2) Update the encoder and decoder model parameters using the
    % 'adamupdate' function
    % 1.3) Record and plot the training loss
    % 1.4) Update the training progress monitor
    % 1.5) Record and plot the validation loss
    % 1.6) Check for early stopping
    % 1.7) Update best model if current model is better
    % 1.8) Update the progress percentage
% 2) Determine the learning rate for the piecewise learning rate schedule
% 3) Stop training when the 'Stop' property of the training progress
% monitor is true

epoch = 0;
iteration = 0;

monitor.Status = "Running";

% Loop over epochs
while epoch < maxEpochs && ~monitor.Stop
    epoch = epoch + 1;

    % Shuffle training mini-batch queues
    shuffle(mbq_train);

    % Reset validation mini-batch queues
    reset(mbq_val);

    % Loop over mini-batches
    while hasdata(mbq_train) && ~earlyStop && ~monitor.Stop
        iteration = iteration + 1;

        % Read mini-batch of data
        [X,T,sequenceLengthsSource,maskTarget] = next(mbq_train);

        % Evaluate the model loss and gradients
        [lossTrain,gradients] = dlfeval(@modelLoss,parameters,X,T, ...
            sequenceLengthsSource,maskTarget,dropout);

        % Update the model parameters using the ADAM optimisation algorithm
        [parameters,trailingAvg,trailingAvgSq] = adamupdate( ...
            parameters,gradients,trailingAvg,trailingAvgSq,iteration, ...
            learnRate,gradientDecayFactor,squaredGradientDecayFactor);

        % Normalise the loss by sequence length
        lossTrain = lossTrain ./ size(T,3);

        % Record training loss
        recordMetrics(monitor,iteration,TrainingLoss=lossTrain);

        % Update the training progress monitor
        updateInfo(monitor, ...
            LearningRate=learnRate, ...
            Epoch=string(epoch) + " of " + string(maxEpochs), ...
            Iteration=string(iteration) + " of " + string(maxIterations));

        % Validation
        if iteration == 1 || mod(iteration,validationFrequency) == 0
            % Read mini-batch of data
            [X,T,~,~] = next(mbq_val);

            % Encode
            sequenceLengths = []; % No masking
            [Z,hiddenState] = modelEncoder(parameters.encoder,X,sequenceLengths);

            % Decoder predictions
            dropout = 0;
            doTeacherForcing = false;
            sequenceLength = size(X,3); % Sequence length to predict
            Y = decoderPredictions(parameters.decoder,Z,T,hiddenState, ...
                dropout,doTeacherForcing,sequenceLength);

            % Compute loss
            lossVal = huber(Y,T,DataFormat="CBT");

            % Normalise the loss by sequence length
            lossVal = lossVal ./ size(T,3);

            % Record validation loss
            recordMetrics(monitor,iteration,ValidationLoss=lossVal);

            % Check for early stopping
            if isfinite(validationPatience)
                validationLosses = [validationLosses lossVal];
                if min(validationLosses) == validationLosses(1)
                    earlyStop = true;
                else
                    validationLosses(1) = [];
                end
            end

            % Update best model if current model is better
            if extractdata(lossVal) < bestValidationLoss
                bestValidationLoss = extractdata(lossVal);
                bestModelParameters = parameters;
                disp(strcat("New best validation loss at epoch ",num2str(epoch),": ",num2str(bestValidationLoss)));
            end
        end

        % Update progress percentage
        monitor.Progress = 100 * iteration / maxIterations;
    end

    % Drop the learn rate if epoch is a multiple of 'learnRateDropPeriod'
    if learnRateSchedule == "piecewise" && mod(epoch,learnRateDropPeriod) == 0
        learnRate = learnRate * learnRateDropFactor;
    end
end

% Update the training status
if monitor.Stop == 1
    monitor.Status = "Training stopped";
else
    monitor.Status = "Training complete";
end

%% Test model
% Preprocess the data
mbq_test = minibatchqueue(dsTest, ...
    numMiniBatchOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@(x,t) preprocessMiniBatch(x,t,inputSize,outputSize));

% Initialise the outputs
Y_test = [];
Y_test_diff = [];

% Loop over mini-batches
while hasdata(mbq_test)
    % Read mini-batch of data
    [X,T,~,~] = next(mbq_test);

    % Encode
    sequenceLengths = []; % No masking
    [Z,hiddenState] = modelEncoder(parameters.encoder,X,sequenceLengths); % sequenceLengthsSource

    % Decoder predictions
    dropout = 0;
    doTeacherForcing = false;
    sequenceLength = size(X,3); % Sequence length to predict
    Y = decoderPredictions(parameters.decoder,Z,X(:,:,end),hiddenState, ...
        dropout,doTeacherForcing,sequenceLength);

    % Determine predictions
    Y = extractdata(gather(Y));
    Y_test = [Y_test Y];

    % Compare predicted and true values
    Y_test_diff_batch = Y - T;
    Y_test_diff = [Y_test_diff extractdata(gather(Y_test_diff_batch))];
end

% Inverse permute the second and third dimensions
Y_test = ipermute(Y_test,[1 3 2]);
Y_test_diff = ipermute(Y_test_diff,[1 3 2]);

% Evaluate the accuracy for each test sequence by calculating the RMSE
rmse = sqrt(mean(Y_test_diff.^2,[1 2]));

% Calculate the mean RMSE over all test data
rmse_all = sqrt(mean(Y_test_diff.^2,"all"));

% Plot the RMSE values
% figure
% histogram(squeeze(rmse))
% xlabel("RMSE")
% ylabel("Frequency")

% figure
% bar(squeeze(rmse))
% xlabel("Test sequence")
% ylabel("RMSE")

% Convert data back to geographic coordinates
    % Denormalise data
    Y_test_geo = cell(numel(X_test),1);
    for n = 1 : numel(X_test)
        Y_test_geo{n} = min_T + [(Y_test(:,:,n).'-l)./(u-l)].*(max_T-min_T); % X or T ?
    end

    % Inverse feature transformation
    for n = 1 : numel(X_test)
        Y_test_geo{n}(1,1:2) = ais_test{n,1}{end,{'lat' 'lon'}} + Y_test_geo{n}(1,1:2);
        for n_k = 2 : size(Y_test_geo{n},1)
            Y_test_geo{n}(n_k,1:2) = Y_test_geo{n}(n_k-1,1:2) + Y_test_geo{n}(n_k,1:2);
        end
    end

% Compute the mean and max great circle distance between predicted and
% target sequences on the test set
gc_dist_mean = zeros(1,numel(Y_test_geo));
gc_dist_max = zeros(1,numel(Y_test_geo));
for gc_dist_idx = 1 : numel(Y_test_geo)
    gc_dist_mean(gc_dist_idx) = mean(distance(Y_test_geo{gc_dist_idx}(:,1:2),...
        ais_test{gc_dist_idx,2}{:,{'lat' 'lon'}},wgs84Ellipsoid('km')));
    gc_dist_max(gc_dist_idx) = max(distance(Y_test_geo{gc_dist_idx}(:,1:2),...
        ais_test{gc_dist_idx,2}{:,{'lat' 'lon'}},wgs84Ellipsoid('km')));
end

% Calculate the mean of the mean gc distance (grand mean or pooled mean)
% and mean of the max gc distance
gc_dist_grand_mean = mean(gc_dist_mean);
gc_dist_max_mean = mean(gc_dist_max);

% Plot the mean and max great circle distance values
figure
histogram(gc_dist_mean,"BinWidth",1)
hold on
histogram(gc_dist_max,"BinWidth",1)
xline(gc_dist_grand_mean,"--",strcat(string(gc_dist_grand_mean)," ","km"))
xline(gc_dist_max_mean,"--",strcat(string(gc_dist_max_mean)," ","km"))
xlabel("Great circle distance (km)")
ylabel("Frequency")
legend("Mean","Max")

% figure
% bar(gc_dist_mean)
% xlabel("Test sequence")
% ylabel("Mean distance (km)")

%% Make predictions (example)
% Find k largest and smallest values and indices
[~,I_max] = maxk(gc_dist_mean,5);
[~,I_min] = mink(gc_dist_mean,5);
I = [I_max I_min];

% Loop over the selected test sequences
for i = 1 : numel(I)
    idx = I(i);
    [X,T,~,~] = preprocessMiniBatch( ...
        X_test(idx),T_test(idx),numFeatures,numResponses);

    % Reset the RNN state
    %

    % Encode
    sequenceLengths = [];
    [Z,hiddenState] = modelEncoder(parameters.encoder,X,sequenceLengths);

    % Decoder predictions
    dropout = 0;
    doTeacherForcing = false;
    sequenceLength = size(X,3);
    Y = decoderPredictions(parameters.decoder,Z,X(:,:,end),hiddenState, ...
        dropout,doTeacherForcing,sequenceLength);

    % Determine predictions
    Y = extractdata(gather(Y));

    % Remove dimensions of length 1
    Y = squeeze(Y);

    % Convert data back to geographic coordinates
        % Denormalise data
        Y_geo = zeros(size(Y));
        for n = 1 : size(Y,2)
            Y_geo(:,n) = min_T + [(Y(:,n).'-l)./(u-l)].*(max_T-min_T);
        end
    
        % Inverse feature transformation
        Y_geo(:,1) = ais_test{idx,1}{end,{'lat' 'lon'}}' + Y_geo(:,1);
        for n_k = 2 : size(Y_geo,2)
            Y_geo(:,n_k) = Y_geo(:,n_k-1) + Y_geo(:,n_k);
        end

    % Show results
    figure
    geoshow(ais_test{idx,1}.lat,ais_test{idx,1}.lon,'Color',[0 0.4470 0.7410], ...
        'Marker','o','MarkerFaceColor',[0 0.4470 0.7410],'MarkerSize',2)
    hold on
    geoshow(ais_test{idx,2}.lat,ais_test{idx,2}.lon,'Color',[0.8500 0.3250 0.0980], ...
        'Marker','o','MarkerFaceColor',[0.8500 0.3250 0.0980],'MarkerSize',2)
    geoshow(Y_geo(1,:),Y_geo(2,:),'Color',[0.9290 0.6940 0.1250], ...
        'Marker','o','MarkerFaceColor',[0.9290 0.6940 0.1250],'MarkerSize',2)
    legend('Input','Target','Prediction','Location','best')
    xlabel('longitude (deg)')
    ylabel('latitude (deg)')
end

%% Mini-batch preprocessing function
% For each mini-batch, the preprocessing function 'preprocessMiniBatch':
% 1) Returns the lengths of all sequences in the mini-batch and (if
% necessary) pads the sequences to the same length as the longest
% sequence, for the source and target sequences, respectively.
% 2) Permutes the second and third dimensions of the padded sequences.
% 3) Returns the mini-batch variables unformatted dlarray objects with
% underlying data type single. All other outputs are arrays of data type
% single.
% 4) Train on a GPU if one is available. Return all mini-batch variables
% on the GPU if one is available.

function [X,T,sequenceLengthsSource,maskTarget] = preprocessMiniBatch( ...
    X_train,T_train,inputSize,outputSize)

sequenceLengthsSource = cellfun(@(x) size(x,2),X_train);

X = padsequences(X_train,2,PaddingValue=inputSize);
X = permute(X,[1 3 2]);

[T,maskTarget] = padsequences(T_train,2,PaddingValue=outputSize);
T = permute(T,[1 3 2]);
maskTarget = permute(maskTarget,[1 3 2]);

end

%% Model loss function
% - modelLoss
%   - modelEncoder
%   - decoderPredictions
%       - modelDecoder

function [loss,gradients] = modelLoss(parameters,X,T, ...
    sequenceLengthsSource,maskTarget,dropout)

% Forward data through the model encoder
[Z,hiddenState] = modelEncoder(parameters.encoder,X,sequenceLengthsSource);

% Decoder output/predictions
doTeacherForcing = rand < 0.5;
sequenceLength = size(T,3);
Y = decoderPredictions(parameters.decoder,Z,T,hiddenState,dropout, ...
    doTeacherForcing,sequenceLength);

% Compute and update loss
loss = huber(Y,T,Mask=maskTarget,DataFormat="CBT");

% Compute and update gradients
gradients = dlgradient(loss,parameters);

end

%% Encoder model function (BiLSTM)
function [Z,hiddenState] = modelEncoder(parameters,X,sequenceLengths) % sequenceLengthsSource

% BiLSTM operation
inputWeights = parameters.bilstm.InputWeights;
recurrentWeights = parameters.bilstm.RecurrentWeights;
bias = parameters.bilstm.Bias;

% Initialise the BiLSTM hidden and cell state
numHiddenUnits = size(recurrentWeights,2);
sz = [2*numHiddenUnits 1];
initialHiddenState = initializeZeros(sz);
initialCellState = initializeZeros(sz);

Z = dlarray(X,"CBT");
[Z,hiddenState] = bilstm(Z,initialHiddenState,initialCellState, ...
    inputWeights,recurrentWeights,bias); % DataFormat="CBT"
Z = stripdims(Z,"CBT");

% Masking for training
if ~isempty(sequenceLengths)
    miniBatchSize = size(Z,2);
    for n = 1 : miniBatchSize
        hiddenState(:,n) = Z(:,n,sequenceLengths(n));
    end
end

end

%% Decoder model function
function [Y,context,hiddenState,attentionScores] = modelDecoder( ...
    parameters,X,context,hiddenState,Z,dropout)

X = dlarray(X); % needed?

% RNN input
sequenceLength = size(X,3);
Y = cat(1,X,repmat(context,[1 1 sequenceLength]));

% LSTM operation
inputWeights = parameters.lstm.InputWeights;
recurrentWeights = parameters.lstm.RecurrentWeights;
bias = parameters.lstm.Bias;

initialCellState = dlarray(zeros(size(hiddenState)));

[Y,hiddenState] = lstm(Y,hiddenState,initialCellState, ...
    inputWeights,recurrentWeights,bias,DataFormat="CBT");

% Dropout
mask = rand(size(Y),"like",Y) > dropout;
Y = Y.*mask;

% Attention
weights = parameters.attention.Weights;
[context,attentionScores] = luongAttention(hiddenState,Z,weights);

% Concatenate
Y = cat(1,Y,repmat(context,[1 1 sequenceLength]));

% Fully connect
weights = parameters.fc.Weights;
bias = parameters.fc.Bias;
Y = fullyconnect(Y,weights,bias,DataFormat="CBT");

end

%% Luong attention function
% The luongAttention function returns the context vector and attention
% scores according to the Luong "general" scoring.
function [context,attentionScores] = luongAttention(hiddenState,Z,weights)

numHeads = 1;
queries = hiddenState;
keys = pagemtimes(weights,Z);
values = Z;

[context,attentionScores] = attention(queries,keys,values,numHeads, ...
    Scale=1,DataFormat="CBT");

end

%% Decoder model predictions function
function Y = decoderPredictions(parameters,Z,T,hiddenState,dropout, ...
    doTeacherForcing,sequenceLength)

% Convert to dlarray
T = dlarray(T);

% Initialise context
miniBatchSize = size(T,2);
numHiddenUnits = size(Z,1);
context = zeros([numHiddenUnits miniBatchSize],"like",Z);

if doTeacherForcing
    % Forward through decoder
    Y = modelDecoder(parameters,T,context,hiddenState,Z,dropout);
else % Autoregressive decoding
    % Get first time step for decoder
    decoderInput = T(:,:,1);

    % Initialise output
    numClasses = numel(parameters.fc.Bias);
    Y = zeros([numClasses miniBatchSize sequenceLength],"like",decoderInput);

    % Loop over time steps
    for t = 1 : sequenceLength
        % Forward through decoder
        [Y(:,:,t),context,hiddenState] = modelDecoder(parameters, ...
            decoderInput,context,hiddenState,Z,dropout);

        % Update decoder input
        decoderInput = Y(:,:,t);
    end
end

end

%% BiLSTM function
function [Y,hiddenState,cellState] = bilstm(X,H0,C0,inputWeights,recurrentWeights,bias)

% Determine forward and backward parameter indices
numHiddenUnits = numel(bias)/8;
idxForward = 1:4*numHiddenUnits;
idxBackward = 4*numHiddenUnits+1:8*numHiddenUnits;

% Forward and backward states
H0Forward = H0(1:numHiddenUnits);
H0Backward = H0(numHiddenUnits+1:end);
C0Forward = C0(1:numHiddenUnits);
C0Backward = C0(numHiddenUnits+1:end);

% Forward and backward parameters
inputWeightsForward = inputWeights(idxForward,:);
inputWeightsBackward = inputWeights(idxBackward,:);
recurrentWeightsForward = recurrentWeights(idxForward,:);
recurrentWeightsBackward = recurrentWeights(idxBackward,:);
biasForward = bias(idxForward);
biasBackward = bias(idxBackward);

% Forward LSTM
[YForward,hiddenStateForward,cellStateForward] = lstm(X,H0Forward,C0Forward,inputWeightsForward, ...
    recurrentWeightsForward,biasForward);

% Backward LSTM
XBackward = X;
idx = finddim(X,"T");
if ~isempty(idx)
    XBackward = flip(XBackward,idx);
end

[YBackward,hiddenStateBackward,cellStateBackward] = lstm(XBackward,H0Backward,C0Backward,inputWeightsBackward, ...
    recurrentWeightsBackward,biasBackward);

if ~isempty(idx)
    YBackward = flip(YBackward,idx);
end

% Merge outputs (cat)
% Y = cat(1,YForward,YBackward);
% hiddenState = cat(1,hiddenStateForward,hiddenStateBackward);
% cellState = cat(1,cellStateForward,cellStateBackward);

% Merge outputs (sum)
Y = YForward + YBackward;
hiddenState = hiddenStateForward + hiddenStateBackward;
cellState = cellStateForward + cellStateBackward;

end

% Code source: https://uk.mathworks.com/help/deeplearning/ug/create-bilstm-function.html

%% Initialization functions
% Glorot Initialization
function weights = initializeGlorot(sz,numOut,numIn)

Z = 2*rand(sz,'single') - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights);

end

% Orthogonal Initialization
function parameter = initializeOrthogonal(sz)

Z = randn(sz,'single');
[Q,R] = qr(Z,0);

D = diag(R);
Q = Q * diag(D ./ abs(D));

parameter = dlarray(Q);

end

% Unit Forget Gate Initialization
function bias = initializeUnitForgetGate(numHiddenUnits)

bias = zeros(4*numHiddenUnits,1,'single');

idx = numHiddenUnits+1:2*numHiddenUnits;
bias(idx) = 1;

bias = dlarray(bias);

end

% Zeros Initialization
function parameter = initializeZeros(sz)

parameter = zeros(sz,'single');
parameter = dlarray(parameter);

end

% Code source: https://uk.mathworks.com/help/deeplearning/ug/initialize-learnable-parameters-for-custom-training-loop.html
