%% Validator.m
% ------------------------------------------------------------
%
% Authors: A.V.Uriadov, D.V.Savchenko
% National Technical University of Ukraine
% "Igor Sikorsky Kyiv Polytechnic Institute"
%
% ------------------------------------------------------------
% PURPOSE:
%   Validation of CNN-predicted Dyson line parameters
%   by reconstruction and comparison with experimental EPR spectra
%
% DATA SOURCES:
%   - Exp data:   data/SPECTRUM_ID.DTA  
%   - B axis:     run/<runName>/B_axis.csv or B_axis.npy
%   - CNN params: run/<runName>/real_predicted_params.json
%   - Meta:       run/<runName>/model_meta.json (for dB_thr_G)
%
% OUTPUT FILES 
%   - <SPECTRUM_ID>_cnn_fit.png
% ------------------------------------------------------------

clear; clc;

%% -------- Settings --------
runName  = '20260125_140421'; % <- % folder inside ./runs/
baseName = '1';          % <- experimental spectrum ID (DTA)
dBThrDefault = 1000;          % fallback threshold if meta is missing

%% -------- Paths relative to this script --------
thisDir    = fileparts(mfilename('fullpath'));
projectDir = thisDir; % if the script is located in the project root;
                      % otherwise: projectDir = fileparts(thisDir);

dataDir = fullfile(projectDir, 'data');
runDir  = fullfile(projectDir, 'runs', runName);

dtaFile  = fullfile(dataDir, [baseName '.DTA']);

predJson = fullfile(thisDir, strcat(baseName, '_real_predicted_params.json'));
metaJson = fullfile(runDir, 'model_meta.json');

bCsv = fullfile(runDir, 'B_axis.csv');
bNpy = fullfile(runDir, 'B_axis.npy');

outPng = fullfile(projectDir, [baseName '_cnn_fit.png']);

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
% yExp now has the same length as B and is free from raw-byte artifacts
yExp = interp1(B_real, spc_real, B, 'linear', 'extrap');

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

%% 7) --- Read dB threshold from model_meta.json --- 
M = jsondecode(fileread(metaJson));
dBThr = dBThrDefault;
if isfield(M, 'generator_meta') && isfield(M.generator_meta, 'dB_thr_G')
    dBThr = M.generator_meta.dB_thr_G;
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

%% 10) --- Plot + save --- 
fig = figure('Color','w','Name','Experimental vs CNN reconstructed');
ax = axes(fig);
ax.Color = 'w';          
ax.XColor = 'k';         
ax.YColor = 'k';
ax.GridColor = [0.8 0.8 0.8]; 
grid(ax,'on'); hold(ax,'on');


plot(B, yExp, 'g', 'LineWidth', 1); hold on;
plot(B, yFit, 'r', 'LineWidth', 1);
grid on;

title(sprintf('%s | B0=%.2f G, dB=%.2f G, p=%.4f | %s', baseName, B0, dB, p3, modeStr), ...
      'Interpreter','none');
xlabel('Magnetic field B (G)');
ylabel('dI/dB (a.u.)');

lgd.Color = 'w';       
lgd.TextColor = 'k';   
lgd.EdgeColor = 'k';   
legend('Experimental', 'Reconstructed from CNN params', ...
       'Location','southwest');

exportgraphics(gcf, outPng, 'Resolution', 200);
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