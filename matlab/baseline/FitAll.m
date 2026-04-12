%% FitAll.m — Batch fit all spectra in a set with both MATLAB and EasySpin
%
% Usage:
%   FitAll              — reads set_name from config/inference.json
%   FitAll('set-1')     — explicit set name
%
% For each .DTA file in data/<set_name>/:
%   1) Runs FitMatlab(setName, id)
%   2) Runs FitEasyspin(setName, id)
%
% Saves summary CSVs:
%   results/<set_name>/matlab/summary.csv
%   results/<set_name>/easyspin/summary.csv
%
% Authors: A.V. Uriadov, D.V. Savchenko
% National Technical University of Ukraine
% "Igor Sikorsky Kyiv Polytechnic Institute"
% -----------------------------------------------------------------

function FitAll(setName)

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(thisDir));
addpath(fullfile(repoRoot, 'matlab'));  % ensure load_config is discoverable
addpath(thisDir);                       % ensure FitMatlab/FitEasyspin are on path

%% Resolve set name
if nargin < 1 || isempty(setName)
    cfgInf  = load_config('inference');
    if isfield(cfgInf, 'set_name')
        setName = char(cfgInf.set_name);
    else
        error('FitAll:NoSetName', ...
              'No set_name argument and no set_name field in config/inference.json.');
    end
end
setName = char(setName);

%% Find all DTA files in data/<setName>/
dataDir  = fullfile(repoRoot, 'data', setName);
assert(isfolder(dataDir), 'Data folder not found: %s', dataDir);

dtaFiles = dir(fullfile(dataDir, '*.DTA'));
nFiles   = length(dtaFiles);
fprintf('=== FitAll: %s — %d spectra ===\n\n', setName, nFiles);

if nFiles == 0
    warning('No .DTA files found in %s', dataDir);
    return;
end

%% Allocate result tables
matlabResults  = cell(nFiles, 1);
easyspinResults = cell(nFiles, 1);

%% Process each spectrum
for i = 1:nFiles
    filename = dtaFiles(i).name;
    specId   = erase(filename, '.DTA');
    fprintf('\n[%d/%d] ---- %s ----\n', i, nFiles, filename);

    % MATLAB optimizer fit
    try
        matlabResults{i} = FitMatlab(setName, specId);
    catch ME
        fprintf('  [FitMatlab] FAILED: %s\n', ME.message);
        matlabResults{i} = makeFailedResult(specId, setName, 'matlab_lsqnonlin', ME.message);
    end

    % EasySpin fit
    try
        easyspinResults{i} = FitEasyspin(setName, specId);
    catch ME
        fprintf('  [FitEasyspin] FAILED: %s\n', ME.message);
        easyspinResults{i} = makeFailedResult(specId, setName, 'easyspin_levmar', ME.message);
    end
end

%% Write summary CSVs
writeSummary(fullfile(repoRoot, 'results', setName, 'matlab', 'summary.csv'), matlabResults);
writeSummary(fullfile(repoRoot, 'results', setName, 'easyspin', 'summary.csv'), easyspinResults);

fprintf('\n=== FitAll: %s — DONE ===\n', setName);
end

%% ========================= Helper functions =========================

function r = makeFailedResult(specId, setName, method, errMsg)
    r = struct();
    r.spectrum_id = specId;
    r.set_name    = setName;
    r.method      = method;
    r.B0_G        = NaN;
    r.dB_G        = NaN;
    r.p           = NaN;
    r.A           = NaN;
    r.g_factor    = NaN;
    r.mwFreq_GHz  = NaN;
    r.rmsd        = NaN;
    r.error       = errMsg;
end

function writeSummary(csvFile, resultsCellArray)
    % Convert cell array of structs into a table and write CSV
    n = numel(resultsCellArray);
    ids      = cell(n,1);
    B0s      = nan(n,1);
    dBs      = nan(n,1);
    ps       = nan(n,1);
    gs       = nan(n,1);
    freqs    = nan(n,1);
    rmsds    = nan(n,1);
    statuses = cell(n,1);

    for i = 1:n
        r = resultsCellArray{i};
        if isempty(r); continue; end
        ids{i}   = r.spectrum_id;
        B0s(i)   = r.B0_G;
        dBs(i)   = r.dB_G;
        ps(i)    = r.p;
        gs(i)    = r.g_factor;
        freqs(i) = r.mwFreq_GHz;
        rmsds(i) = r.rmsd;
        if isfield(r, 'error')
            statuses{i} = 'FAILED';
        else
            statuses{i} = 'OK';
        end
    end

    T = table(ids, B0s, dBs, ps, gs, freqs, rmsds, statuses, ...
              'VariableNames', {'spectrum_id','B0_G','dB_G','p','g_factor','mwFreq_GHz','rmsd','status'});
    writetable(T, csvFile);
    fprintf('[FitAll] Summary saved: %s\n', csvFile);
end
