% s_data_preprocessing.m
% This script performs data preprocessing.

clear
clc

%% Import data
filename = "AIS_169884559859562343_5169-1698845599902.csv";
ais = f_import_mc_csv_v2023(filename);

% Convert VesselType codes to names
ais.vessel_type = convertVesselTypeCodes(ais.vessel_type);

%% Manage missing and invalid data
% in: ais [table]
% out: ais [table]

% Remove duplicates
[~,u_idx] = unique(ais(:,{'datetime','mmsi'}));
ais = ais(u_idx,:);

% Remove rows with invalid MMSI
mmsi_str = string(ais.mmsi);
mmsi_str_length = strlength(mmsi_str);
mmsi_invalid = ~(mmsi_str_length == 9);
ais(mmsi_invalid,:) = [];

% Keep rows with specified vessel_type - TEMP
ais = ais(ais.vessel_type == "Cargo",:);

% Plot AIS data
% figure
% worldmap([min(ais.lat)-0.1 max(ais.lat)+0.1], ...
%     [min(ais.lon)-0.1 max(ais.lon)+0.1]) % latlim,lonlim
% geoshow(ais.lat,ais.lon,'DisplayType','point', ...
%     'MarkerEdgeColor',[0.6350 0.0780 0.1840],'Marker','x')

% Sort the rows of the table
ais = sortrows(ais,{'mmsi','datetime'},'ascend');

%% Aggregate data into sequences
% in: ais [table]
% out: ais_sseqs [cell array of tables]

% Get unique values
u_mmsi = unique(ais.mmsi);

% Loop over unique MMSI and aggregate the data into sequences
ais_sseqs = {}; % Subsequences
for seq_idx = 1 : numel(u_mmsi)
    % Retrieve a sequence from the data
    ais_seq = ais(ais.mmsi == u_mmsi(seq_idx),:);

    % Pass to the next iteration if size(sequence) < 2
    if size(ais_seq,1) < 2
        continue
    end

    % Determine the implied speed (m/s)
    ais_seq.speed_implied = speedImplied(ais_seq.datetime,ais_seq.lat, ...
        ais_seq.lon,ais_seq.sog);

    % Determine the implied bearing (deg)
    ais_seq.bearing_implied = bearingImplied(ais_seq.lat,ais_seq.lon, ...
        ais_seq.cog(1));

    % Segment the sequence into subsequences by applying a time interval threshold
    time_threshold = hours(1);
    ais_sseq = segmentSeq(ais_seq,time_threshold);

    % Concatenate the subsequences while looping over each sequence
    ais_sseqs = cat(1,ais_sseqs,ais_sseq);
end

%% Resample subsequences
% in: ais_sseqs [cell array of tables]
% out: ais_sseqs [cell array of tables]

% Define time steps
dt = 5;
dt = minutes(dt);

% Resample the subsequences' datetimes to regular time intervals
% specified by dt (note: no extrapolation)
for sseq_idx = 1 : numel(ais_sseqs)
    % Retrieve a subsequence from the cell array
    ais_sseq = ais_sseqs{sseq_idx};

    % Convert table to timetable
    ais_sseq_tt = table2timetable(ais_sseq);

    % Resample data in timetable
    ais_sseq_tt_1 = retime(ais_sseq_tt(:,{'lat','lon'}), ...
        'regular','linear','TimeStep',dt,'EndValues',NaN); % 'spline'
    ais_sseq_tt_2 = retime(ais_sseq_tt(:,'speed_implied'), ...
        'regular','linear','TimeStep',dt,'EndValues',NaN); % 'makima'
    ais_sseq_tt_3 = retime(ais_sseq_tt(:,'bearing_implied'), ...
        'regular','nearest','TimeStep',dt,'EndValues',NaN); % TODO: implement circular interpolation

    % Concatenate values
    ais_sseq_tt = [ais_sseq_tt_1 ais_sseq_tt_2 ais_sseq_tt_3];

    % Add back vessel type column
    ais_sseq_tt.vessel_type = repmat(ais_sseq.vessel_type(1),[size(ais_sseq_tt,1) 1]);

    % Remove rows with NaN values
    ais_sseq_tt = rmmissing(ais_sseq_tt);

    % Update the table in the cell array
    ais_sseqs{sseq_idx} = timetable2table(ais_sseq_tt);
end

%% Transform features
for sseq_idx = 1 : numel(ais_sseqs)
    % Pass to the next iteration if size(subsequence) < 2
    if size(ais_sseqs{sseq_idx},1) < 2
        continue
    end

    % Compute the difference between consecutive observations
    ais_sseqs{sseq_idx}.lat_diff = [0; diff(ais_sseqs{sseq_idx}.lat)];
    ais_sseqs{sseq_idx}.lon_diff = [0; diff(ais_sseqs{sseq_idx}.lon)];
    ais_sseqs{sseq_idx}.speed_implied_diff = [0; diff(ais_sseqs{sseq_idx}.speed_implied)];
    ais_sseqs{sseq_idx}.bearing_implied_diff = [0; abs(diff(ais_sseqs{sseq_idx}.bearing_implied))];
    ais_sseqs{sseq_idx,1}(1,:) = [];
end

% Remove empty subsequences
% ais_sseqs_empty = cellfun(@isempty,ais_sseqs);
% ais_sseqs(ais_sseqs_empty) = [];

%% Exclude empty and single-entry subsequences
% in: ais_sseqs [cell array of tables]
% out: ais_sseqs [cell array of tables]

% Identify subsequences with fewer than two entries
is_empty_or_single = cellfun(@(x) istable(x) && height(x) < 2,ais_sseqs);

% Remove subsequences with fewer than two entries from the cell array
ais_sseqs = ais_sseqs(~is_empty_or_single);

%% Exclude outliers
% in: ais_sseqs [cell array of tables]
% out: ais_sseqs [cell array of tables]

% Concatenate features from all subsequences
    % Preallocate variables to hold concatenated data and indices
    total_rows = sum(cellfun(@(x) size(x,1),ais_sseqs));
    total_cols = 2; % For 'lat_diff' and 'lon_diff'
    data_global = zeros(total_rows,total_cols);
    sseq_indices = zeros(total_rows,1); % To track the origin subsequence

    current_idx = 1;
    for sseq_idx = 1 : numel(ais_sseqs)
        % Retrieve a subsequence from the cell array
        ais_sseq = ais_sseqs{sseq_idx};
        data_subset = table2array(ais_sseq(:,["lat_diff" "lon_diff"]));
        num_rows = size(data_subset,1);

        % Concatenate the data
        data_global(current_idx:(current_idx + num_rows - 1),:) = data_subset;

        % Record the subsequence index for each row
        sseq_indices(current_idx:(current_idx + num_rows - 1)) = sseq_idx;

        % Update 'current_idx' for the next iteration
        current_idx = current_idx + num_rows;
    end

% Apply isolation forest
    % Specify the fraction of outliers in the data
    contamination_fraction = 0.05;

    % Detect outliers using an isolation forest
    rng("default") % For reproducibility
    [forest,tf_forest,s_forest] = iforest(data_global, ...
        ContaminationFraction=contamination_fraction);

    % Plot a histogram of the score values
    figure
    histogram(s_forest,Normalization="probability")
    xline(forest.ScoreThreshold,"k-", ...
        join(["Threshold =" forest.ScoreThreshold]))
    title("Histogram of Anomaly Scores for Isolation Forest")

% Trace outliers back to original subsequences
    % Find which subsequences contain outliers
    outlier_indices = unique(sseq_indices(tf_forest));

    % Remove subsequences containing outliers
    ais_sseqs(outlier_indices) = [];

%% Visualise subsequences
% Plot the subsequences
figure
worldmap([min(ais.lat)-0.1 max(ais.lat)+0.1],[min(ais.lon)-0.1 max(ais.lon)+0.1])
for sseq_idx = 1 : numel(ais_sseqs)
    geoshow(ais_sseqs{sseq_idx}.lat,ais_sseqs{sseq_idx}.lon)
    hold on
end

% Display a frequency table of the subsequences' vessel types
vessel_type = cellfun(@(tbl) tbl.vessel_type(1),ais_sseqs,'UniformOutput',false);
vessel_type = cat(1,vessel_type{:});
tabulate(vessel_type)

%% Apply sliding window
% in: ais_sseqs [cell array of tables]
% out: input_seq, response_seq [cell array of tables]

% Parameters for sliding window
input_window_length = hours(2.5) / dt; % Number of time steps in window
response_window_length = hours(2.5) / dt;
step_size = 1; % Time step difference between consecutive windows

% Apply sliding window
[input_seq,response_seq] = slidingWindow(ais_sseqs,input_window_length, ...
    response_window_length,step_size);

%% Prepare training, validation and test data splits
% Define the input features
% Available input features: 'lat','lon','speed_implied','bearing_implied',
% 'lat_diff','lon_diff','speed_implied_diff','bearing_implied_diff'
input_features = {'lat_diff','lon_diff'};

% Define the response features
response_features = {'lat_diff','lon_diff'};

% Get indices for training (80%), validation (10%) and test (10%) split
n_sseq = numel(input_seq); % Number of subsequences
rng("default")
[idx_train,idx_val,idx_test] = trainingPartitions(n_sseq,[0.8 0.1 0.1]);
ais_train = [input_seq(idx_train) response_seq(idx_train)];
ais_val = [input_seq(idx_val) response_seq(idx_val)];
ais_test = [input_seq(idx_test) response_seq(idx_test)];

%%% Training %%%
% Specify the predictors (X_train) and the responses (T_train)
X_train = cell(size(ais_train,1),1);
T_train = cell(size(ais_train,1),1);
for ais_train_idx = 1 : size(ais_train,1)
    X = ais_train{ais_train_idx,1};
    T = ais_train{ais_train_idx,2};
    X_train{ais_train_idx} = X{:,input_features};
    T_train{ais_train_idx} = T{:,response_features};
end

% Rescale the data to the range [l,u] (Note: when making predictions, also
% normalise the test data using the same statistics as the training data)
l = -1; % lower
u = 1; % upper
X_train_cat = cat(1,X_train{:}); % Calculate statistics over all sequences
T_train_cat = cat(1,T_train{:});
min_X = min(X_train_cat);
max_X = max(X_train_cat);
min_T = min(T_train_cat);
max_T = max(T_train_cat);
for n = 1 : numel(X_train)
    X_train{n} = l + [(X_train{n}-min_X)./(max_X-min_X)].*(u-l);
    T_train{n} = l + [(T_train{n}-min_T)./(max_T-min_T)].*(u-l);
end

% Reformat the training data for the RNN
X_train = cellfun(@transpose,X_train,"UniformOutput",false);
T_train = cellfun(@transpose,T_train,"UniformOutput",false);

%%% Validation %%%
% Specify the predictors (X_val) and the responses (T_val)
X_val = cell(size(ais_val,1),1);
T_val = cell(size(ais_val,1),1);
for ais_val_idx = 1 : size(ais_val,1)
    X = ais_val{ais_val_idx,1};
    T = ais_val{ais_val_idx,2};
    X_val{ais_val_idx} = X{:,input_features};
    T_val{ais_val_idx} = T{:,response_features};
end

% Rescale the data to the range [l,u] (use the same statistics as the
% training data)
for n = 1 : numel(X_val)
    X_val{n} = l + [(X_val{n}-min_X)./(max_X-min_X)].*(u-l);
    T_val{n} = l + [(T_val{n}-min_T)./(max_T-min_T)].*(u-l);
end

% Reformat the test data
X_val = cellfun(@transpose,X_val,"UniformOutput",false);
T_val = cellfun(@transpose,T_val,"UniformOutput",false);

%%% Test %%%
% Prepare the test data for prediction using the same steps as for the
% training data

% Specify the predictors (X_test) and the responses (T_test)
X_test = cell(size(ais_test,1),1);
T_test = cell(size(ais_test,1),1);
for ais_test_idx = 1 : size(ais_test,1)
    X = ais_test{ais_test_idx,1};
    T = ais_test{ais_test_idx,2};
    X_test{ais_test_idx} = X{:,input_features};
    T_test{ais_test_idx} = T{:,response_features};
end

% Rescale the data to the range [l,u] (use the same statistics as the
% training data)
for n = 1 : numel(X_test)
    X_test{n} = l + [(X_test{n}-min_X)./(max_X-min_X)].*(u-l);
    T_test{n} = l + [(T_test{n}-min_T)./(max_T-min_T)].*(u-l);
end

% Reformat the test data
X_test = cellfun(@transpose,X_test,"UniformOutput",false);
T_test = cellfun(@transpose,T_test,"UniformOutput",false);

%% Save all variables from workspace
save("s_data_preprocessing_workspace.mat")

%% Save only required variables from workspace
save("s_data_preprocessing_variables.mat", ...
    "ais_test","X_train","T_train","X_val","T_val","X_test","T_test", ...
    "l","u","min_X","max_X","min_T","max_T") % "ais_train","ais_val",

%% Supporting local functions
function vesselTypeNames = convertVesselTypeCodes(vesselTypeCodes)
%convertVesselTypeCodes Convert Marine Cadastre VesselType codes to
% corresponding VesselType names (e.g. '1024' == Tanker).
%
%   VESSELTYPENAMES = CONVERTVESSELTYPECODES(VESSELTYPECODES)
%       Input: a categorical array VESSELTYPECODES of vessel type codes
%       Output: a categorical array VESSELTYPENAMES of vessel type names
%
%   Example: ais.VesselType = convertVesselTypeCodes(ais.VesselType);

% Define VesselType codes and names as a cell array of character vectors
vesselTypeCodesMaster = {
    cellstr(string([70:1:79 1003 1004 1016])) ...
    cellstr(string([30 1001 1002])) ...
    cellstr(string([35 1021])) ...
    cellstr(string(0)) ...
    cellstr(string([1:1:19 20 23:1:29 33 34 38:1:51 53:1:59 90:1:99 100:1:199 ...
    200:1:255 256:1:999 1005:1:1011 1018 1020 1022])) ...
    cellstr(string([60:1:69 1012:1:1015])) ...
    cellstr(string([36 37 1019])) ...
    cellstr(string([80:1:89 1017 1024])) ...
    cellstr(string([21 22 31 32 52 1023 1025]))};

vesselTypeNamesMaster = {
    'Cargo', ...
    'Fishing', ...
    'Military', ...
    'NA', ...
    'Other', ...
    'Passenger', ...
    'Pleasure', ...
    'Tanker', ...
    'Tug'};

% Convert VesselType codes to names
for k = 1 : numel(vesselTypeNamesMaster)
    vesselTypeNames = mergecats(vesselTypeCodes,vesselTypeCodesMaster{k},vesselTypeNamesMaster{k});
    vesselTypeCodes = vesselTypeNames;
end

end


function speed_implied = speedImplied(t,lat,lon,sog)
%speedImplied Calculate the implied speed (m/s) from lat and lon.
%
%   SPEED_IMPLIED = SPEEDIMPLIED(T,LAT,LON,SOG)
%       Inputs: T [datetime], LAT [double], LON [double] and SOG [double]
%       Output: SPEED_IMPLIED [double]
%
%   Example: ais_seq.speed_implied = speedImplied(ais_seq.datetime,ais_seq.lat,ais_seq.lon,ais_seq.sog);

if length(t) < 2 % Check if group has fewer than two points
    speed_implied = sog / 1.9438444924574; % kts to m/s
else
    % Convert the datetime (t) to a duration in seconds
    t = seconds(t - t(1));

    % Calculate the time differences between consecutive rows in seconds
    t_diff = diff(t);

    % Calculate the distances between consecutive rows in metres
    dist = distance(lat(1:end-1),lon(1:end-1),lat(2:end),lon(2:end),wgs84Ellipsoid);

    % Include the first entry from the SOG column*
    sog_1 = sog(1) / 1.9438444924574; % knots to metres per second

    % Calculate the speeds by dividing the distances by the time differences
    speed_implied = {[sog_1; dist ./ t_diff]};

    % Convert the cell array to a numeric array
    speed_implied = cell2mat(speed_implied);
end

end

%*needed for the case when interp_time is between the first and second
% point and interpolation is carried out


function bearing_implied = bearingImplied(lat,lon,cog)
%bearingImplied Calculate the implied bearing (deg) from lat and lon.
%   The bearing (azimuth) is the angle a line makes with a meridian (line
%   of longitude), measured on a sphere in degrees clockwise from north
%   (ranging from 0 to 360). See the MATLAB functions 'distance' and 'azimuth'.
%
%   BEARING_IMPLIED = BEARINGIMPLIED(LAT,LON,COG)
%       Inputs: LAT [double], LON [double] and COG [double]
%       Output: BEARING_IMPLIED measured clockwise from True North [double]
%
%   Example: ais_seq.bearing_implied = bearingImplied(ais_seq.lat,ais_seq.lon,ais_seq.cog(1));

n_points = length(lat);
bearing_implied = zeros(1, n_points - 1);

for i = 1 : (n_points - 1)
    lat1 = lat(i);
    lon1 = lon(i);
    lat2 = lat(i + 1);
    lon2 = lon(i + 1);

    % Convert latitude and longitude from degrees to radians
    lat1 = deg2rad(lat1);
    lon1 = deg2rad(lon1);
    lat2 = deg2rad(lat2);
    lon2 = deg2rad(lon2);

    % Calculate the bearing using the Haversine formula
    dLon = lon2 - lon1;
    y = sin(dLon) * cos(lat2);
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon);

    % Convert the bearing from radians to degrees
    bearing = atan2(y,x);
    bearing = rad2deg(bearing);

    % Normalise the bearing to be in the range [0, 360] degrees
    bearing = mod(bearing + 360,360);

    % Store the bearing in the results array
    bearing_implied(i) = bearing;
end

% Include the first entry from the COG column
bearing_implied = [cog bearing_implied]';

end


function ais_sseq = segmentSeq(ais_seq,time_threshold)
%segmentSeq Segment sequence into subsequences based on a time threshold.
%
%   AIS_SSEQ = SEGMENTSEQ(AIS_SEQ,TIME_THRESHOLD)
%       Input: the table AIS_SSEQ and duration TIME_THRESHOLD
%       Output: the cell array AIS_SSEQ made up of subsequences
%
%   Example: ais_sseq = segmentSeq(ais_seq,time_threshold);

% Compute the time intervals between subsequent rows
time_diff = [0; diff(ais_seq.datetime)];

% Define a threshold
% time_threshold = hours(time_threshold);

% Create a binary vector
new_sseq = time_diff > time_threshold;
new_sseq_indices = find(new_sseq);

% Find the indices
idx_start = [1; new_sseq_indices];
idx_end = [new_sseq_indices-1; numel(new_sseq)];

% Split the sequence into subsequences according to the threshold
ais_sseq = cell(numel(idx_start),1);
for t = 1 : numel(idx_start)
    ais_sseq{t} = ais_seq(idx_start(t):idx_end(t),:);
end

end


function varargout = trainingPartitions(numObservations,splits)
%TRAININGPARTITONS Random indices for splitting training data
%   [idx1,...,idxN] = trainingPartitions(numObservations,splits) returns
%   random vectors of indices to help split a data set with the specified
%   number of observations, where SPLITS is a vector of length N of
%   partition sizes that sum to one.
%
%   % Example: Get indices for 50%-50% training-test split of 500
%   % observations.
%   [idxTrain,idxTest] = trainingPartitions(500,[0.5 0.5])
%
%   % Example: Get indices for 80%-10%-10% training, validation, test split
%   % of 500 observations. 
%   [idxTrain,idxValidation,idxTest] = trainingPartitions(500,[0.8 0.1 0.1])

arguments
    numObservations (1,1) {mustBePositive}
    splits {mustBeVector,mustBeInRange(splits,0,1,"exclusive"),mustSumToOne}
end

numPartitions = numel(splits);
varargout = cell(1,numPartitions);

idx = randperm(numObservations);

idxEnd = 0;

for i = 1:numPartitions-1
    idxStart = idxEnd + 1;
    idxEnd = idxStart + floor(splits(i)*numObservations) - 1;

    varargout{i} = idx(idxStart:idxEnd);
end

% Last partition.
varargout{end} = idx(idxEnd+1:end);

end

function mustSumToOne(v)
% Validate that value sums to one.

if sum(v,"all") ~= 1
    error("Value must sum to one.")
end

end

% Code source: openExample('nnet/TrainNeuralNetworkWithTabularDataExample')
% TrainNetworkWithTabularDataExample.mlx


function [input_seq,response_seq] = slidingWindow(ais_sseqs,input_window_length,response_window_length,step_size)
%slidingWindow Apply sliding window to a cell array of subsequences.
%
%   [INPUT_SEQ,RESPONSE_SEQ] = SLIDINGWINDOW(AIS_SSEQS,INPUT_WINDOW_LENGTH,RESPONSE_WINDOW_LENGTH,STEP_SIZE)
%       Inputs: AIS_SSEQS [table], INPUT_WINDOW_LENGTH [positive
%       integer], RESPONSE_WINDOW_LENGTH [positive integer] and STEP_SIZE
%       [positive integer]
%       Outputs: INPUT_SEQ [cell array of tables] and RESPONSE_SEQ [cell array of tables]
%
%   Example: [input_seq,response_seq] = slidingWindow(ais_sseqs,input_window_length,response_window_length,step_size);

input_seq = {};
response_seq = {};

for sseq_idx = 1 : numel(ais_sseqs)
    % Retrieve a subsequence from the cell array
    ais_sseq = ais_sseqs{sseq_idx};

    % Get the sequence length
    seq_length = size(ais_sseq,1);

    % If the sequence is too short, either skip or pad it
    if seq_length < (input_window_length + response_window_length)
        continue % Skip this sequence
    end

    % Apply sliding window
    for i = 1 : step_size : (seq_length - input_window_length - response_window_length + 1)
        input_window = ais_sseq(i:(i + input_window_length - 1), :);
        response_window = ais_sseq((i + input_window_length):(i + input_window_length + response_window_length - 1), :);

        input_seq{end + 1} = input_window;
        response_seq{end + 1} = response_window;
    end
end

% Transpose the resulting cell arrays
input_seq = input_seq';
response_seq = response_seq';

% Remove empty subsequences
input_seq_empty = cellfun(@isempty,input_seq);
input_seq(input_seq_empty) = [];
response_seq(input_seq_empty) = [];

end
