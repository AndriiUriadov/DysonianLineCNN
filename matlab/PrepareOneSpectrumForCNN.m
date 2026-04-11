%% -------------- PrepareOneSpectrumForCNN.m --------------
%
% Prepares ONE real spectrum (from .DTA/.DSC) for input into a CNN in Colab.
%
% ---------------------------------------------------------
%
% Authors: A.V.Uriadov, D.V.Savchenko
% National Technical University of Ukraine
% "Igor Sikorsky Kyiv Polytechnic Institute"
%
% ---------------------------------------------------------
% 
% Output: spectrum.csv in the project root (where this .m file is located),
%         plus a PNG preview.
%
% Requirements:
% 1) The project root contains the folder runs/<runName>/B_axis.csv;
% 2) The real spectrum is located in the data folder (under project root);
% 3) Bruker reading function is required: eprload (EasySpin).
% ---------------------------------------------------------

clear; clc; close all;

%% === 0) PROJECT ROOT ===
scriptPath  = fileparts(mfilename('fullpath'));
projectRoot = scriptPath;
fprintf("[INFO] Project root: %s\n", projectRoot);

%% === 1) USER SETTINGS ===
% 1.1) Manual selection of run (model)
                                % <<<<<<<<<<<<<<<<<<<<<<<<<<
runName = "20260125_140421";    % <<< SPECIFY MODEL RUN
                                % <<<<<<<<<<<<<<<<<<<<<<<<<<
                            
runDir  = fullfile(projectRoot, "runs", runName);
assert(isfolder(runDir), "[WARN] Run directory not found: %s", runDir);

% Base name (without .DTA/.DSC)
                                % <<<<<<<<<<<<<<<<<<<<<<<<<<
realBaseName = "1";             % <<< SPECIFY .DTA BASE NAME
                                % <<<<<<<<<<<<<<<<<<<<<<<<<<

% 1.2) B_axis.csv (from selected run)
BaxisFile = fullfile(runDir, "B_axis.csv");
assert(isfile(BaxisFile), "[WARN] B_axis.csv not found in run: %s", BaxisFile);

% 1.3) Real spectrum in the data folder (base name without extension)
dataDir = fullfile(projectRoot, "data");
assert(isfolder(dataDir), "[WARN] Data folder not found: %s", dataDir);

% 1.4) Output files in project root
outCsv = fullfile(projectRoot, strcat(realBaseName, '_spectrum.csv'));
outPng = fullfile(projectRoot, strcat(realBaseName, '_spectrum_preview.png'));
% 

%% === 2) LOAD B_axis (CNN axis) ===
B_axis = readmatrix(BaxisFile);
B_axis = double(B_axis(:));
assert(numel(B_axis) == 4096, "[WARN] B_axis must be 4096 points, got %d", numel(B_axis));
fprintf("[INFO] Loaded B_axis: %d points\n", numel(B_axis));

%% === 3) LOAD REAL SPECTRUM (Bruker DTA/DSC) ===
% EasySpin is required (eprload). If EasySpin is not in the path, add it:
% addpath('C:\path\to\easyspin');

% Build path to DTA file
realDTA = fullfile(dataDir, realBaseName + ".DTA");
assert(isfile(realDTA), "[WARN] DTA file not found: %s", realDTA);

% Read spectrum
% eprload returns: [B, spc, par] or [spc, par] depending on the version.
try
    [B_real, spc_real, par] = eprload(realDTA);
catch
    % fallback (some versions return only spc, par)
    [spc_real, par] = eprload(realDTA);
    % then try to obtain B from par (if available) or fail
    if isfield(par, "Field") || isfield(par, "B")
        error("[WARN] Unable to reliably reconstruct B_axis from par in this version. Please update EasySpin or use [B,spc,par] output.");
    else
        error("[WARN] eprload did not return B_real. Check EasySpin installation or file format.");
    end
end

% Spectrum may be row/column/matrix — enforce (N,1)
spc_real = double(spc_real(:));
B_real   = double(B_real(:));

fprintf("[INFO] OK. Loaded real spectrum: %d points\n", numel(spc_real));

%% === 4) BASIC CLEANUP (minimal, no aggressive processing) ===
% Remove NaN/Inf values (if present)
bad = ~isfinite(spc_real) | ~isfinite(B_real);
if any(bad)
    spc_real(bad) = [];
    B_real(bad)   = [];
    fprintf("[WARN] Removed %d non-finite points\n", nnz(bad));
end

% Sign correction (optional): detector inversion
% Uncomment if needed:
% if abs(min(spc_real)) > abs(max(spc_real))
%     spc_real = -spc_real;
% end

% Detrending (baseline shift): subtract median (more robust than mean)
spc_real = spc_real - median(spc_real);

%% === 5) RESAMPLE REAL SPECTRUM ONTO CNN B_axis (4096 points) ===
% CNN expects exactly the same B_axis as used during generation/training.
% Therefore, interpolate the real spectrum onto B_axis.
%
% IMPORTANT: B_real must be monotonic (increasing). If not — sort it.
if any(diff(B_real) < 0)
    [B_real, idx] = sort(B_real, "ascend");
    spc_real = spc_real(idx);
end

% Check range overlap
if (min(B_axis) < min(B_real)) || (max(B_axis) > max(B_real))
    fprintf("[WARN] B_axis extends beyond the range of real B_real.\n");
    fprintf("       B_axis: [%.3f .. %.3f]\n", min(B_axis), max(B_axis));
    fprintf("       B_real: [%.3f .. %.3f]\n", min(B_real), max(B_real));
    fprintf("       Linear extrapolation will be used.\n");
end

% Interpolation (linear + extrapolation)
spc_on_axis = interp1(B_real, spc_real, B_axis, "linear", "extrap");
spc_on_axis = double(spc_on_axis(:));

%% === 6) (OPTIONAL) LIGHT SMOOTHING (very mild) ===
% If real data are noisy, light smoothing may help.
% Do NOT overdo it: the CNN was trained on a specific line shape.
doSmooth = false;    % <<< TRUE if needed
if doSmooth
    win = 9;         % small odd window
    spc_on_axis = movmean(spc_on_axis, win);
end

%% === 7) SAVE OUTPUT FOR COLAB ===
% Save SINGLE-CHANNEL signal (4096x1) to CSV.
% In Colab you will convert it to (1,4096,3) using your channels:
% ch0=standardize, ch1=ptp, ch2=pos(B_axis) or your chosen scheme.
writematrix(spc_on_axis, outCsv);
fprintf("[INFO] Saved spectrum to: %s\n", outCsv);

%% === 8) PREVIEW PLOT ===
fig = figure('Color','w','Position',[100 100 900 350]);
plot(B_axis, spc_on_axis, 'LineWidth', 1.1);
grid on;
xlabel('B (Gauss)');
ylabel('Signal (a.u.)');
title(sprintf("Prepared spectrum on CNN axis (%s)", realBaseName), 'Interpreter','none');
exportgraphics(fig, outPng, 'Resolution', 150);
fprintf("[INFO] OK. Saved preview to:  %s\n", outPng);

disp("[INFO] DONE.");