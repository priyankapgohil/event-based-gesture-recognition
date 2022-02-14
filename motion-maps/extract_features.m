%% clear workspace %%
clear;
close;
clc;


%% get the current folder and cd to it %%
currentFolder = regexprep(mfilename('fullpath'), mfilename(), '');
cd(currentFolder);


%% dataset we want to extract features for %%
datasetName = 'IITM_DVS_10';
% path to the dataset %
dataset_path = fullfile(currentFolder, '/data/', datasetName, '/original_data/');
data_dir = dir(dataset_path);
% path to the features extracted %
features_path = fullfile(currentFolder, '/data/', datasetName, '/features_extracted/');
features_dir = dir(features_path);

% remove hidden folders and files from it %
inds = hidden_indices(data_dir);
data_dir(inds) = [];
inds = hidden_indices(features_dir);
features_dir(inds) = [];


%% extract motion maps for each video %%
tic
fprintf('Generating motion maps for %s dataset. This might take a while.\n', datasetName);
generate_motion_maps(dataset_path, features_path);
toc
