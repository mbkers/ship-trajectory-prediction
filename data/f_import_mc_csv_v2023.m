function ais = f_import_mc_csv_v2023(filename, dataLines)
%F_IMPORT_MC_CSV_V2023 Import data from a text file
%  AIS = F_IMPORT_MC_CSV_V2023(FILENAME) reads data from text file FILENAME
%  for the default selection. Returns the data as a table.
%
%  AIS = F_IMPORT_MC_CSV_V2023(FILE, DATALINES) reads data for the
%  specified row interval(s) of text file FILENAME. Specify DATALINES as a
%  positive scalar integer or a N-by-2 array of positive scalar integers
%  for dis-contiguous row intervals.
%
%  Example:
%  ais = f_import_mc_csv_v2023("C:\Users\mkers\Desktop\AIS_169643192171262183_6652-1696431926223.csv", [2, Inf]);
%
%  See also READTABLE.

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 17);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["mmsi", "datetime", "lat", "lon", "sog", "cog", "Var7", "Var8", "Var9", "Var10", "vessel_type", "Var12", "Var13", "Var14", "Var15", "Var16", "Var17"];
opts.SelectedVariableNames = ["mmsi", "datetime", "lat", "lon", "sog", "cog", "vessel_type"];
opts.VariableTypes = ["categorical", "datetime", "double", "double", "double", "double", "string", "string", "string", "string", "categorical", "string", "string", "string", "string", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Var7", "Var8", "Var9", "Var10", "Var12", "Var13", "Var14", "Var15", "Var16", "Var17"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["mmsi", "Var7", "Var8", "Var9", "Var10", "vessel_type", "Var12", "Var13", "Var14", "Var15", "Var16", "Var17"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, "datetime", "InputFormat", "yyyy-MM-dd'T'HH:mm:ss");

% Import the data
ais = readtable(filename, opts);

end