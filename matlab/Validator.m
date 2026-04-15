%% ========================================================================
%  Validator.m
%  Reconstruct a Dysonian spectrum from CNN-predicted parameters and
%  overlay it on the experimental data for visual quality assessment
%
%  Authors:  A.V. Uriadov, D.V. Savchenko
%  National Technical University of Ukraine
%  "Igor Sikorsky Kyiv Polytechnic Institute"
%
% =========================================================================
%
%  PURPOSE
%    Load CNN-predicted parameters (B0, dB, p) and the original Bruker
%    spectrum, reconstruct the Dysonian line shape, scale it to the
%    experimental data via least-squares (a*sim + b), and save an overlay
%    plot for visual validation.
%
%  CONFIGURATION
%    Reads from config/inference.json:
%      set_name, runName, spectrum_basename
%    Reads per-set dataset config (config/sets/<set_name>.json or
%    config/dataset.json) for dB_thr_G (narrow/wide mode threshold).
%
%  INPUT
%    data/<set_name>/<basename>.DTA                                  — raw Bruker spectrum
%    <Drive>/results/<set_name>/cnn/runs/<runName>/B_axis.csv        — CNN magnetic field axis
%    <Drive>/results/<set_name>/cnn/runs/<runName>/model_meta.json   — training metadata
%    <Drive>/results/<set_name>/cnn/<basename>_predicted.json        — CNN output
%
%  OUTPUT
%    <Drive>/results/<set_name>/cnn/<basename>_cnn_fit.png — overlay plot
%      Title: set-N/id | B0=... G, dB=... G, p=...
%      Green = experimental, Red = CNN reconstruction
%
%  PHYSICS MODEL
%    Feher-Kip Dysonian derivative with full two-term A/D coefficients
%    (Holiatkina et al., J. Appl. Phys. 134, 145702, 2023).
%    Narrow mode (dB < dB_thr): standard Dyson.
%    Wide mode (dB > dB_thr): modified Dyson/Joshi.
%
%  REQUIREMENTS
%    EasySpin (eprload), MATLAB R2025b
%
% =========================================================================

clear; clc;

%% -------- Load config --------
% runName / baseName / dB_thr_G come from config/*.json so MATLAB and
% Python read a single source of truth.
thisDir   = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(thisDir);
addpath(thisDir);  % ensure load_config is discoverable

cfgPaths = load_config('paths');
cfgInf   = load_config('inference');

% Per-set support: inference.json now has an optional set_name field.
if isfield(cfgInf, 'set_name') && ~isempty(cfgInf.set_name)
    setName = char(cfgInf.set_name);
    cfgDS = load_config(['sets/' setName]);
else
    setName = '';
    cfgDS = load_config('dataset');
end

runName  = char(cfgInf.runName);
baseName = char(cfgInf.spectrum_basename);
if strcmp(runName, 'TO_BE_SET_AFTER_FIRST_TRAINING')
    error('Validator:RunNotSet', ...
          'config/inference.json runName is still the placeholder. Train a model and update runName.');
end
dBThrDefault = double(cfgDS.dB_thr_G);  % authoritative narrow/wide threshold

%% -------- Resolve paths --------
% Convention in the current architecture:
%   - raw Bruker .DTA/.DSC live in the LOCAL repo data/<set>/ (script-relative)
%   - trained runs live on Drive under
%     <project>/results/<set>/cnn/<runs_subdir>/<runName>/
%   - predicted params JSON is written by dyson_cnn.infer.predict_for_set to
%     <project>/results/<set>/cnn/<baseName>_predicted.json
%   - the fit overlay PNG is written to
%     <project>/results/<set>/cnn/<baseName>_cnn_fit.png
driveProjectDir = fullfile(cfgPaths.drive_root_mac, cfgPaths.project_subdir);
if ~isempty(setName)
    driveSetDir = fullfile(driveProjectDir, 'results', setName, 'cnn');
    dataDir     = fullfile(repoRoot, 'data', setName);
else
    driveSetDir = driveProjectDir;
    dataDir     = fullfile(repoRoot, 'data');
end
runDir = fullfile(driveSetDir, char(cfgPaths.runs_subdir), runName);

dtaFile  = fullfile(dataDir, [baseName '.DTA']);

predJson = fullfile(driveSetDir, strcat(baseName, '_predicted.json'));
metaJson = fullfile(runDir, 'model_meta.json');

bCsv = fullfile(runDir, 'B_axis.csv');
bNpy = fullfile(runDir, 'B_axis.npy');

outPng = fullfile(driveSetDir, [baseName '_cnn_fit.png']);

%% --- Sanity checks ---
if ~isfile(dtaFile);  error('DTA not found: %s', dtaFile); end
if ~isfile(predJson); error('Pred JSON not found: %s', predJson); end
if ~isfile(metaJson); error('Meta JSON not found: %s', metaJson); end
if ~isfile(bCsv) && ~isfile(bNpy)
    error('B_axis not found. Need either %s or %s', bCsv, bNpy);
end

%% 1) --- Load CNN B-axis ---  
if isfile(bCsv)
    % Most robust for single-column CSVs and nonstandard headers
    B = readmatrix(bCsv);
else
    B = readNPY(bNpy);
end
B = double(B(:));
if isempty(B) || all(~isfinite(B))
    error('B_axis is empty or invalid.');
end

%% 2) --- Read experimental spectrum via EasySpin eprload --- 
% eprload automatically reads DSC and related metadata if available
try
    [B_real, spc_real] = eprload(dtaFile);
catch ME
    error('eprload() failed. Перевір, що EasySpin додано до MATLAB path.\n%s', ME.message);
end

B_real   = double(B_real(:));
spc_real = double(spc_real(:));

% Remove NaN / Inf values
good = isfinite(B_real) & isfinite(spc_real);
B_real   = B_real(good);
spc_real = spc_real(good);

% Ensure monotonic B axis for interpolation
if any(diff(B_real) <= 0)
    [B_real, ord] = sort(B_real, 'ascend');
    spc_real = spc_real(ord);
end

%% 3) --- Unit consistency check (pitfall: mT vs G) --- 
if max(B) > 2000 && max(B_real) < 2000
    % підозра на mT
    B_real = 10 * B_real;
end

%% 4) --- Interpolate experimental spectrum onto CNN B-axis ---
% yExp now has the same length as B and is free from raw-byte artifacts.
% Zero-fill outside the Bruker sweep window rather than linearly
% extrapolating — a narrow ~200 G sweep extended over 4500 G of CNN axis
% via linear extrapolation generates ±10^6 scale tails that dominate the
% least-squares a*sim + b fit below and produce a completely flat red
% reconstructed curve (see PrepareOneSpectrumForCNN.m for the same fix).
yExp = interp1(B_real, spc_real, B, 'linear', 0);

% Remove DC offset (stabilizes a*sim + b scaling)
yExp = yExp - median(yExp);

%% 5) --- (Optional) visible-only region (B > 0) --- 
% Important: applied AFTER interpolation to preserve matching lengths
idxVis = (B > 0);
if any(idxVis)
    B    = B(idxVis);
    yExp = yExp(idxVis);
end

%% 6)  --- Read predicted parameters --- 
S = jsondecode(fileread(predJson));
if isfield(S, 'y_pred_physical')
    B0 = S.y_pred_physical.B0;
    dB = S.y_pred_physical.dB;
    p3 = S.y_pred_physical.p3;
else
    error('JSON must contain y_pred_physical.{B0,dB,p3}');
end

%% 7) --- dB threshold from config/dataset.json ---
% dBThrDefault was loaded at the top from cfgDS.dB_thr_G, which is the
% single source of truth. We still cross-check against model_meta.json's
% embedded generator_meta so a mismatched run directory (e.g. dataset
% regenerated with a different threshold after training) is flagged.
dBThr = dBThrDefault;
if isfile(metaJson)
    try
        M = jsondecode(fileread(metaJson));
        if isfield(M, 'generator_meta') && isfield(M.generator_meta, 'dB_thr_G')
            metaThr = double(M.generator_meta.dB_thr_G);
            if abs(metaThr - dBThrDefault) > 1e-9
                warning('Validator:ThresholdMismatch', ...
                        ['model_meta.json dB_thr_G=%.6g differs from ' ...
                         'config/dataset.json dB_thr_G=%.6g. Using config value.'], ...
                        metaThr, dBThrDefault);
            end
        end
    catch ME
        warning('Validator:MetaReadFailed', 'Could not read %s: %s', metaJson, ME.message);
    end
end

%% 8) --- Reconstruct spectrum using DysonGeneratorMix logic (narrow/wide)
isNarrow = (dB < dBThr);

if isNarrow
    ySim = dysonNarrow_dIdB(B, B0, dB, p3);
    modeStr = 'narrow (Dyson)';
else
    % wide/Joshi on visible axis via uniform axis + interpolation
    B2 = max(B);
    Npoints = numel(B);
    Buni = linspace(0, B2, Npoints).';
    yUni = dysonWideJoshi_visible_dIdB(B2, Npoints, B0, dB, p3);
    ySim = interp1(Buni, yUni, B, 'linear', 'extrap');
    modeStr = 'wide (modified Dyson/Joshi)';
end

% Polarity fix like generator
ySim = polarityFix(ySim);

%% 9) --- Scale reconstructed to experimental (a*sim + b) --- 
A = [ySim(:), ones(numel(ySim),1)];
coef = A \ yExp(:);
a = coef(1);
b = coef(2);
yFit = a*ySim + b;

%% 10) --- Compute g-factor ---
muB = 9.2740100783e-24;
h   = 6.62607015e-34;

% Read mwFreq from DSC
mwFreqGHz = 9.86;
got_mwFreqGHz = false;
dscPath = fullfile(dataDir, [baseName '.DSC']);
if isfile(dscPath)
    txt = fileread(dscPath);
    freqPattern = regexp(txt, 'FrequencyMon\s+([\d\.Ee\+\-]+)\s*GHz', 'tokens', 'once');
    if ~isempty(freqPattern)
        mwFreqGHz = str2double(freqPattern{1});
        got_mwFreqGHz = true;
    end
end
g_est = (h*(mwFreqGHz*1e9)) / (muB*(B0*1e-4));

%% 11) --- Plot + save (same format as FitMatlab/FitEasyspin) ---
fig = figure('Visible','off','Color','w','Position',[140 180 1000 540]);
ax = axes(fig);
ax.Color = 'w';
ax.XColor = 'k';
ax.YColor = 'k';
ax.GridColor = [0.8 0.8 0.8];
grid(ax,'on'); hold(ax,'on');

plot(B, yExp, 'g-', 'LineWidth', 1.0);
plot(B, yFit, 'r-', 'LineWidth', 1.2);
xlabel('Magnetic field B (G)');
ylabel('dI/dB (a.u.)');

if ~isempty(setName)
    prefix = [setName '/' baseName];
else
    prefix = baseName;
end

titleStr = sprintf('%s | B0=%.2f G, dB=%.2f G, p=%.3f', ...
    prefix, B0, dB, p3);

t = title(ax, titleStr, 'FontWeight','bold', 'Interpreter','none');
t.Color = 'k';

lgd = legend(ax, 'Experimental', 'CNN fit', 'Location','best');
lgd.Color = 'w';
lgd.TextColor = 'k';
lgd.EdgeColor = 'k';

saveas(fig, outPng);
close(fig);
fprintf('Saved: %s\n', outPng);

%% ========================= Local functions =========================

function sig = dysonNarrow_dIdB(B, B0, dBpp, p)
dBpp = max(dBpp, eps);
x = 2*(B - B0) ./ (sqrt(3)*dBpp);

[Acoef, Dcoef] = dysonAD_from_p(p);

den = (1 + x.^2).^2;
termAbs  = (2*x) ./ den;
termDisp = (1 - x.^2) ./ den;

sig = (-Acoef .* termAbs) + (Dcoef .* termDisp);
end

function [Acoef, Dcoef] = dysonAD_from_p(p)
% Compute Dyson mixing coefficients A(p), D(p).
% Full formula with two terms (Holiatkina et al., J. Appl. Phys. 134, 125306 (2023))

p = max(p, 1e-9); % avoid division by zero
ch = cosh(p); sh = sinh(p);
c  = cos(p);  s  = sin(p);
den = (ch + c);

% Full formula with two terms (as used in DysonFitMatlab.m and DysonFitEasyspin_Simplex.m)
denom1 = 2*p.*den;
denom2 = den.^2;

Acoef = (sh + s)./denom1 + (1 + ch.*c)./denom2;
Dcoef = (sh - s)./denom1 + (sh.*s)./denom2;
end

function sigVis = dysonWideJoshi_visible_dIdB(B2, Npoints, B0, dB, alpha)
alpha = max(alpha, 0);
dB = max(dB, eps);

Bfull = linspace(-B2, +B2, 2*Npoints);
dBstep = (Bfull(end)-Bfull(1))/(numel(Bfull)-1);

num1 = (dB + alpha*(Bfull - B0));
den1 = (4*(Bfull - B0).^2 + dB^2);

num2 = (dB - alpha*(Bfull + B0));
den2 = (4*(Bfull + B0).^2 + dB^2);

P = (num1 ./ den1) + (num2 ./ den2);
dP = gradient(P, dBstep);

idx = (Bfull > 0);
sigVis = dP(idx);

if numel(sigVis) ~= Npoints
    Bvis = linspace(0, B2, Npoints);
    sigVis = interp1(Bfull(idx), sigVis, Bvis, 'linear', 'extrap');
end

sigVis = sigVis(:);
end

function sig = polarityFix(sig)
[~, iMax] = max(sig);
[~, iMin] = min(sig);
if iMin < iMax
    sig = -sig;
end
end

function data = readNPY(filename)
% Minimal NPY reader for 1D numeric arrays (little-endian).
fid = fopen(filename, 'r');
if fid < 0, error('Cannot open NPY file: %s', filename); end

magic = fread(fid, 6, 'uint8')';
if ~isequal(magic, [147 78 85 77 80 89])
    fclose(fid);
    error('Not a valid NPY file: %s', filename);
end

fread(fid, 2, 'uint8'); % version
hlen = fread(fid, 1, 'uint16');
header = char(fread(fid, hlen, 'uint8')');

dtypeTok = regexp(header, '''descr'':\s*''([^'']+)''', 'tokens', 'once');
shapeTok = regexp(header, '''shape'':\s*\(([^)]*)\)', 'tokens', 'once');

if isempty(dtypeTok) || isempty(shapeTok)
    fclose(fid);
    error('Cannot parse NPY header: %s', filename);
end

dtype = dtypeTok{1};
shapeStr = strtrim(shapeTok{1});
dims = sscanf(strrep(shapeStr, ',', ' '), '%d')';
count = prod(dims);

switch dtype
    case {'<f8','|f8'}; freadType = 'double';
    case {'<f4','|f4'}; freadType = 'single';
    case {'<i8','|i8'}; freadType = 'int64';
    case {'<i4','|i4'}; freadType = 'int32';
    otherwise
        fclose(fid);
        error('Unsupported dtype in NPY: %s', dtype);
end

data = fread(fid, count, [freadType '=>double']);
fclose(fid);

data = reshape(data, dims);
end