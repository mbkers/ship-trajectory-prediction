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
    % 1) Add the Mixture Density Network (MDN) parameters
    % 2) Initialise the weights with the Glorot initialiser
    % 3) Initialise the bias with zeros using zeros initialisation*
    % *(see the 'Supporting functions' section at the end of the script)

    % Specify the number of Gaussian components in the mixture
    numGaussians = 5;

    % Define the output size for the MDN
    outputSize = numGaussians*(2*numResponses+1);

    % Initialise the MDN parameters
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
% Learning rate
learnRateInitial = 0.1; % Initial learning rate
decayRate = 0.96; % Decay rate per epoch
decaySteps = 1; % Adjust the learning rate at every epoch

% Mini-batch size
miniBatchSize = 128;

% Number of epochs
maxEpochs = 100;

% Early stopping
validationPatience = Inf;

% L2 regularization
l2Regularization = 0.0001; % L2 regularization coefficient, lambda

% Gradient clipping (global-l2norm)
gradientThreshold = 10;

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
    MiniBatchFcn=@(x,t) preprocessMiniBatch(x,t,inputSize,numResponses));

mbq_val = minibatchqueue(dsVal, ...
    numMiniBatchOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@(x,t) preprocessMiniBatch(x,t,inputSize,numResponses));

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
bestValidationLoss = Inf;
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
    % 1.2) Apply L2 regularization to the weights
    % 1.3) Apply the gradient threshold operation (if needed)
    % 1.4) Determine the learning rate for the learning rate schedule
    % 1.5) Update the encoder and decoder model parameters using the
    % 'adamupdate' function
    % 1.6) Record and plot the training loss
    % 1.7) Update the training progress monitor
    % 1.8) Record and plot the validation loss
    % 1.9) Check for early stopping
    % 1.10) Update best model if current model is better
    % 1.11) Update the progress percentage
% 2) Stop training when the 'Stop' property of the training progress
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
        [lossTrain,gradients] = dlfeval(@modelLossWrapper,parameters,X,T, ...
            sequenceLengthsSource,maskTarget,dropout);

        % Apply L2 regularization to the gradients of the weight parameters
        gradients = applyL2Regularization(gradients,parameters,l2Regularization);

        % Apply the gradient threshold operation
        gradients = thresholdGlobalL2Norm(gradients,gradientThreshold);

        % Calculate the new learning rate with exponential decay
        learnRate = learnRateInitial * decayRate^((epoch - 1) / decaySteps);

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
            % dropout = 0;
            doTeacherForcing = false;
            sequenceLength = size(X,3); % Sequence length to predict
            [~,Y_pi,Y_mu,Y_sigma] = decoderPredictions(parameters.decoder,Z,T, ...
                hiddenState,0,doTeacherForcing,sequenceLength); % dropout,

            % Compute loss
            lossVal = mdnNegativeLogLikelihood(Y_pi,Y_mu,Y_sigma,T,[]);

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
                disp(strcat("New best validation loss at epoch ",num2str(epoch), ...
                    " (iteration ",num2str(iteration),"): ",num2str(bestValidationLoss)));
            end
        end

        % Update progress percentage
        monitor.Progress = 100 * iteration / maxIterations;
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
    MiniBatchFcn=@(x,t) preprocessMiniBatch(x,t,inputSize,numResponses));

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
    [Y,~,~,~] = decoderPredictions(parameters.decoder,Z,X(:,:,end), ...
        hiddenState,dropout,doTeacherForcing,sequenceLength);

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
    Y = decoderPredictions(parameters.decoder,Z,X(:,:,end), ...
        hiddenState,dropout,doTeacherForcing,sequenceLength);

    % Determine predictions
    Y = extractdata(Y); % extractdata(gather(Y));

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
    X_train,T_train,inputSize,numResponses)

sequenceLengthsSource = cellfun(@(x) size(x,2),X_train);

X = padsequences(X_train,2,PaddingValue=inputSize);
X = permute(X,[1 3 2]);

[T,maskTarget] = padsequences(T_train,2,PaddingValue=numResponses);
T = permute(T,[1 3 2]);
maskTarget = permute(maskTarget,[1 3 2]);

end

%% Model loss function
% - modelLoss
%   - modelEncoder
%   - decoderPredictions
%       - modelDecoder

% Custom function to wrap the modelLoss function
function [loss,gradients] = modelLossWrapper(parameters,X,T, ...
    sequenceLengthsSource,maskTarget,dropout)

% Compute the loss
loss = modelLoss(parameters,X,T,sequenceLengthsSource,maskTarget,dropout);

% Compute the gradients
gradients = dlgradient(loss,parameters);

end


function loss = modelLoss(parameters,X,T, ...
    sequenceLengthsSource,maskTarget,dropout) % [loss,gradients]

% Forward data through the model encoder
[Z,hiddenState] = modelEncoder(parameters.encoder,X,sequenceLengthsSource);

% Decoder predictions
doTeacherForcing = rand < 0.5;
sequenceLength = size(T,3);
[~,Y_pi,Y_mu,Y_sigma] = decoderPredictions(parameters.decoder,Z,T, ...
    hiddenState,dropout,doTeacherForcing,sequenceLength);

% Compute negative log-likelihood loss
loss = mdnNegativeLogLikelihood(Y_pi,Y_mu,Y_sigma,T,maskTarget);

% Compute gradients
% gradients = dlgradient(loss,parameters);

end


function nll = mdnNegativeLogLikelihood(mixingCoefficients,means,stdevs,target,mask)

% Compute the negative log-likelihood loss for the mixture density network
[numGaussians,numSamples,numTimeSteps] = size(mixingCoefficients);
numResponses = size(target,1);

% Reshape target to match the dimensions of means and stdevs
target = reshape(target,[numResponses 1 numSamples numTimeSteps]);

% Compute the Gaussian probability density for each component
diff = target - means;
exponent = -0.5 * (diff ./ stdevs).^2;
normalizer = sqrt(2 * pi) .* stdevs;
gaussianProbabilities = exp(exponent) ./ normalizer;
gaussianProbabilities = prod(gaussianProbabilities,1); % Joint pdf

% Reshape mixingCoefficients to match the dimensions of gaussianProbabilities
mixingCoefficients = reshape(mixingCoefficients,[1 numGaussians numSamples numTimeSteps]);

% Compute the weighted probabilities
weightedProbabilities = sum(mixingCoefficients .* gaussianProbabilities,2);

% Add a small epsilon value to weightedProbabilities for numerical stability
epsilon = 1e-8;
weightedProbabilities = weightedProbabilities + epsilon;

% Reshape mask to match the dimensions of weightedProbabilities
% mask = reshape(mask,[numResponses 1 numSamples numTimeSteps]);

% Apply the mask to the weighted probabilities
% weightedProbabilities = weightedProbabilities .* mask(1,:,:,:);

% Compute the negative log-likelihood loss
logProb = log(weightedProbabilities);
nll = -sum(logProb,"all");

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
function [Y_pi,Y_mu,Y_sigma,context,hiddenState,attentionScores] = modelDecoder( ...
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

% Fully connect/Mixture Density Network
weights = parameters.fc.Weights;
bias = parameters.fc.Bias;
Y_mdn = fullyconnect(Y,weights,bias,DataFormat="CBT");

% Split the MDN output into mixing coefficients, means, and standard deviations
numResponses = size(X,1);
numGaussians = size(weights,1) / (2*numResponses+1);
numSamples = size(X,2);
sequenceLength = size(X,3);

% Mixing coefficients
Y_pi = Y_mdn(1:numGaussians,:,:);
Y_pi = softmax(Y_pi,DataFormat="CBT"); % Ensure they sum to 1 for each time step
% Shape of Y_pi: [numGaussians numSamples sequenceLength]

% Means
Y_mu = Y_mdn(numGaussians+1:numGaussians+numResponses*numGaussians,:,:);
Y_mu = reshape(Y_mu,[numResponses numGaussians numSamples sequenceLength]);

% Standard deviations
Y_sigma = Y_mdn(numGaussians+numResponses*numGaussians+1:end,:,:);
Y_sigma = reshape(Y_sigma,[numResponses numGaussians numSamples sequenceLength]);
Y_sigma = exp(Y_sigma); % Ensure positivity

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
function [Y,Y_pi,Y_mu,Y_sigma] = decoderPredictions( ...
    parameters,Z,T,hiddenState,dropout,doTeacherForcing,sequenceLength)

% Convert to dlarray
T = dlarray(T);

% Initialise context
numHiddenUnits = size(Z,1);
miniBatchSize = size(T,2);
context = zeros([numHiddenUnits miniBatchSize],"like",Z);

numResponses = size(T,1);
numGaussians = size(parameters.fc.Weights,1) / (2*numResponses+1);

if doTeacherForcing
    % Forward through decoder with teacher forcing
    [Y_pi,Y_mu,Y_sigma] = modelDecoder(parameters,T,context,hiddenState,Z,dropout);

    % Sample from the mixture of Gaussians
    % Y = sampleFromMixture(Y_pi,Y_mu,Y_sigma);
    Y = [];
else
    % Autoregressive decoding
    % Get first time step for decoder
    decoderInput = T(:,:,1);

    % Initialise output (sampled predictions)
    Y = zeros([numResponses miniBatchSize sequenceLength],"like",decoderInput);

    % Initialise MDN outputs
    Y_pi = zeros([numGaussians miniBatchSize sequenceLength],"like",decoderInput);
    Y_mu = zeros([numResponses numGaussians miniBatchSize sequenceLength],"like",decoderInput);
    Y_sigma = zeros([numResponses numGaussians miniBatchSize sequenceLength],"like",decoderInput);

    % Loop over time steps
    for t = 1 : sequenceLength
        % Forward through decoder
        [Y_pi(:,:,t),Y_mu(:,:,:,t),Y_sigma(:,:,:,t),context,hiddenState] = modelDecoder( ...
            parameters,decoderInput,context,hiddenState,Z,dropout);

        % Sample from the mixture of Gaussians
        Y(:,:,t) = sampleFromMixture(Y_pi(:,:,t),Y_mu(:,:,:,t),Y_sigma(:,:,:,t));

        % Update decoder input
        decoderInput = Y(:,:,t);
    end
end

% Helper function to sample from the mixture of Gaussians
    function samples = sampleFromMixture(mixingCoefficients,means,stdevs)
        % Convert from dlarray to numeric array
        mixingCoefficients = extractdata(mixingCoefficients);

        % Compute cumulative sum of mixing coefficients
        cumulativeMixingCoefficients = cumsum(mixingCoefficients,1);

        % Generate random numbers for each sample in the mini-batch
        randomNumbers = rand(1,miniBatchSize);

        % Find the selected Gaussian component for each sample in the mini-batch
        [~,selectedIndices] = max(cumulativeMixingCoefficients > randomNumbers,[],1);

        % Sample from the selected Gaussian components for each sample in the mini-batch
        samples = zeros(numResponses,miniBatchSize,"like",means);

        % Convert selectedIndices to linear indices
        linearIndices = sub2ind(size(means,2:3),selectedIndices,1:miniBatchSize);

        % Gather the corresponding means and standard deviations using linear indexing
        selectedMeans = means(:,linearIndices);
        selectedStdevs = stdevs(:,linearIndices);

        % Generate random noise for each sample in the mini-batch
        noise = randn(numResponses,miniBatchSize,"like",means);

        % Compute the samples by adding the scaled noise to the selected means
        samples = selectedMeans + selectedStdevs .* noise;
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

%% L2 regularization
% This function iterates through the parameters and gradients structures,
% applying L2 regularization to the gradients of the weights.

function gradients = applyL2Regularization(gradients,parameters,l2Regularization)
% List of component names in network
componentNames = fieldnames(parameters);

for i = 1 : numel(componentNames)
    componentName = componentNames{i}; % e.g., 'encoder', 'decoder'

    % Get subcomponents, e.g., 'bilstm', 'attention', 'lstm', 'fc'
    subComponentNames = fieldnames(parameters.(componentName));

    for j = 1 : numel(subComponentNames)
        subComponentName = subComponentNames{j};

        % Get parameter types, e.g., 'InputWeights', 'RecurrentWeights', 'Bias'
        paramTypes = fieldnames(parameters.(componentName).(subComponentName));

        for k = 1 : numel(paramTypes)
            paramType = paramTypes{k};

            % Check if the current parameter is a weight (exclude biases)
            if contains(paramType,"Weights")
                % Apply L2 regularization to the gradient
                gradients.(componentName).(subComponentName).(paramType) = ...
                    gradients.(componentName).(subComponentName).(paramType) + ...
                    l2Regularization * parameters.(componentName).(subComponentName).(paramType);
            end
            % Bias gradients are not regularized, so no else clause is needed
        end
    end
end
end

%% Gradient clipping
% This function first computes the global L2 norm of all gradients in the
% structure by iterating through each field and subfield(s), summing the
% square of all gradient elements, and taking the square root of the total.
% If the global L2 norm exceeds the specified gradientThreshold, each
% gradient is scaled down by the same factor (normScale) to ensure the
% global L2 norm after scaling is equal to the threshold.

function gradients = thresholdGlobalL2Norm(gradients,gradientThreshold)

globalL2Norm = 0;

% Calculate global L2 norm
fieldNames = fieldnames(gradients);
for i = 1 : numel(fieldNames)
    subFieldNames = fieldnames(gradients.(fieldNames{i}));
    for j = 1 : numel(subFieldNames)
        subSubFieldNames = fieldnames(gradients.(fieldNames{i}).(subFieldNames{j}));
        for k = 1 : numel(subSubFieldNames)
            gradientValues = gradients.(fieldNames{i}).(subFieldNames{j}).(subSubFieldNames{k});
            globalL2Norm = globalL2Norm + sum(gradientValues(:).^2);
        end
    end
end
globalL2Norm = sqrt(globalL2Norm);

% Scale gradients if global L2 norm exceeds the threshold
if globalL2Norm > gradientThreshold
    normScale = gradientThreshold / globalL2Norm;
    for i = 1 : numel(fieldNames)
        subFieldNames = fieldnames(gradients.(fieldNames{i}));
        for j = 1 : numel(subFieldNames)
            subSubFieldNames = fieldnames(gradients.(fieldNames{i}).(subFieldNames{j}));
            for k = 1 : numel(subSubFieldNames)
                gradients.(fieldNames{i}).(subFieldNames{j}).(subSubFieldNames{k}) = ...
                    gradients.(fieldNames{i}).(subFieldNames{j}).(subSubFieldNames{k}) * normScale;
            end
        end
    end
end
end
