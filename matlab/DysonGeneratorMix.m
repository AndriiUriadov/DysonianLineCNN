%% ========================================================================
%  DysonGeneratorMix.m
%  Synthetic EPR Dysonian / Modified-Dyson (Joshi) Dataset Generator
%
%  Authors:  A.V. Uriadov, D.V. Savchenko
%  National Technical University of Ukraine
%  "Igor Sikorsky Kyiv Polytechnic Institute"
%
% =========================================================================
%
%  PURPOSE
%    Generate synthetic first-derivative EPR spectra for training a CNN
%    to regress Dysonian line parameters (B0, dB, p). Automatically
%    selects narrow (Feher-Kip Dyson, parameter p) or wide (Joshi,
%    parameter alpha) model based on dBRange vs dB_thr_G (default 1000 G).
%
%  USAGE
%    % Legacy single-model mode:
%    DysonGeneratorMix
%
%    % Per-set mode (reads config/sets/<SetName>.json):
%    SetName = 'set-1'; DysonGeneratorMix
%
%  CONFIGURATION
%    All parameters come from JSON config files via load_config():
%      - Per-set mode:  config/sets/<SetName>.json
%      - Legacy mode:   config/dataset.json
%    Output directory resolved from config/paths.json.
%    Per-set output goes to <DriveProjectDir>/results/<SetName>/cnn/dataset/.
%
%  PHYSICS MODEL
%    Geometry-aware Dysonian first-derivative lineshape dispatched via
%    dysonNarrow(B, B0, dB, p, geometry). Geometry is read from
%    cfgDS.geometry (per-set config) and stored in meta.geometry:
%      - 'plate'  (default) — Feher-Kip flat-plate, full two-term A/D
%                              coefficients
%                              (Holiatkina et al., 2023). Used for sets 1..5.
%      - 'sphere'            — spherical powder-grain geometry with
%                              additional 1/p^3 and 1/p^4 terms
%                              (Savchenko et al., 2022). Used for set-6
%                               (CDs@SiO2 nanocomposites).
%
%  AUGMENTATION PIPELINE
%    physics -> polarity fix -> modulation smoothing (probabilistic) ->
%    field scale/offset warp + resample -> baseline -> noise -> spikes ->
%    peak-to-peak normalization
%
%  OUTPUT FILES
%    X_dyson_mix_<Prefix>.npy   [N x Npoints] float32 — spectra
%    y_dyson_mix_<Prefix>.npy   [N x 3]       float32 — labels [B0, dB, p3]
%    B_axis_mix_<Prefix>.npy    [Npoints]      float32 — magnetic field axis
%    meta_mix_<Prefix>.json     struct         — generation metadata
%
%  LABELS
%    y(:,3) = p in narrow mode (skin-effect parameter)
%    y(:,3) = alpha in wide mode (Joshi asymmetry, alpha >= 0)
%
%  REQUIREMENTS
%    MATLAB R2025b, npy-matlab (writeNPY) or Python+NumPy for .npy output
%
% =========================================================================

%% --------------------- Load config ---------------------
% All tunable defaults come from config JSON files so that MATLAB and
% Python share a single source of truth. When SetName is provided (e.g.
% 'set-1'), reads config/sets/<SetName>.json instead of config/dataset.json
% and saves output to <DriveProjectDir>/<SetName>/ instead of the root.
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);  % ensure load_config is discoverable

% Optional set name: set by caller before running, or empty for legacy mode
if ~exist('SetName','var') || isempty(SetName)
    SetName = '';
end

if ~isempty(SetName)
    cfgDS = load_config(['sets/' SetName]);
else
    cfgDS = load_config('dataset');
end
cfgPaths = load_config('paths');

%% --------------------- Debug settings ---------------------
DoVisualCheck = logical(cfgDS.DoVisualCheck);  % show random spectra for visual sanity check
Nvisual       = double(cfgDS.Nvisual);         % number of spectra to plot

%% --------------------- Parse inputs ---------------------
% Parsing controls parameters data type 
p = inputParser;
% To define, validate, and document configurable parameters.

p.FunctionName = mfilename;
% Assign the current file name to the parser for clearer
% error messages and diagnostics.

% --------------------- Core dataset parameters ---------------------

% All defaults below come from config/dataset.json via cfgDS. Hardcoded
% fallbacks removed — change them in dataset.json, not here.

addParameter(p,'N',         double(cfgDS.N),          @(x)isnumeric(x)&&isscalar(x)&&x>0);
% Total number of synthetic spectra to generate.

addParameter(p,'B0Range',   double(cfgDS.B0Range_G(:)'), @(x)isnumeric(x)&&numel(x)==2&&x(1)<x(2));
% Range of resonance magnetic field B0 (in Gauss).

addParameter(p,'dBRange',   double(cfgDS.dBRange_G(:)'), @(x)isnumeric(x)&&numel(x)==2&&x(1)<x(2));
% Range of peak-to-peak linewidth dB (ΔB_pp) in Gauss.

addParameter(p,'pRange',    double(cfgDS.pRange(:)'),  @(x)isnumeric(x)&&numel(x)==2&&x(1)<=x(2));
% Range of asymmetry parameter:
%   - narrow mode: physical Dyson parameter p
%   - wide mode: effective asymmetry parameter alpha

addParameter(p,'BWindow',   double(cfgDS.BWindow_G(:)'), @(x)isnumeric(x)&&numel(x)==2&&x(1)<x(2));
% Magnetic field window [Bmin, Bmax] used for the visible spectrum (in Gauss).

addParameter(p,'Npoints',   double(cfgDS.Npoints),    @(x)isnumeric(x)&&isscalar(x)&&x>=128);
% Number of points in the magnetic field axis (visible part, B > 0).

addParameter(p,'dBThr',     double(cfgDS.dB_thr_G),   @(x)isnumeric(x)&&isscalar(x)&&x>0);
% Threshold linewidth (in Gauss) separating narrow and wide Dyson regimes.

addParameter(p,'OutDir',    '',                       @(x)ischar(x)||isstring(x));
% Output directory for generated files.
% If empty, resolved from config/paths.json:
%   fullfile(drive_root_mac, project_subdir).

addParameter(p,'Prefix',    char(cfgDS.Prefix),       @(x)ischar(x)||isstring(x));
% Filename prefix for all generated output files.

addParameter(p,'Seed',      double(cfgDS.Seed),       @(x)isnumeric(x)&&isscalar(x));
% Random number generator seed (for reproducibility).

addParameter(p,'DoAugment', logical(cfgDS.DoAugment), @(x)islogical(x)&&isscalar(x));
% Enable or disable spectral augmentations (noise, baseline, distortions).

% --------------------- Augmentation parameters ---------------------

addParameter(p,'SNRdBRange',         double(cfgDS.SNRdBRange(:)'), @(x)isnumeric(x)&&numel(x)==2&&x(1)<=x(2));
% Signal-to-noise ratio range (in dB) used for additive Gaussian noise.

addParameter(p,'BaselineOffsetRange',double(cfgDS.BaselineOffsetRange(:)'), @(x)isnumeric(x)&&numel(x)==2);
% Constant baseline offset added to the spectrum (relative units).

addParameter(p,'BaselineSlopeRange', double(cfgDS.BaselineSlopeRange(:)'),  @(x)isnumeric(x)&&numel(x)==2);
% Linear baseline slope range across the magnetic field window.

addParameter(p,'BaselineQuadRange',  double(cfgDS.BaselineQuadRange(:)'),   @(x)isnumeric(x)&&numel(x)==2);
% Quadratic baseline curvature range (second-order field dependence).

addParameter(p,'BScaleRange',        double(cfgDS.BScaleRange(:)'),         @(x)isnumeric(x)&&numel(x)==2&&x(1)<=x(2));
% Global magnetic field scaling factor (simulates calibration errors).

addParameter(p,'BOffsetRange',       double(cfgDS.BOffsetRange_G(:)'),      @(x)isnumeric(x)&&numel(x)==2);
% Global magnetic field offset (in Gauss).

addParameter(p,'SpikeProb',          double(cfgDS.SpikeProb),               @(x)isnumeric(x)&&isscalar(x)&&x>=0&&x<=1);
% Probability that a spectrum contains a spike artifact.

addParameter(p,'SpikeCountRange',    double(cfgDS.SpikeCountRange(:)'),     @(x)isnumeric(x)&&numel(x)==2);
% Range of the number of spike artifacts per spectrum.

addParameter(p,'SpikeAmpRange',      double(cfgDS.SpikeAmpRange(:)'),       @(x)isnumeric(x)&&numel(x)==2&&x(1)<=x(2));
% Relative amplitude range of spike artifacts (fraction of signal scale).

addParameter(p,'ModSmoothProb',      double(cfgDS.ModSmoothProb),           @(x)isnumeric(x)&&isscalar(x)&&x>=0&&x<=1);
% Probability of applying modulation-induced smoothing
% (simulates lock-in detection effects).

addParameter(p,'ModSmoothSigmaPts',  double(cfgDS.ModSmoothSigmaPts(:)'),   @(x)isnumeric(x)&&numel(x)==2&&x(1)<=x(2));
% Gaussian smoothing width (sigma) in units of data points.

parse(p);
cfg = p.Results;

cfg.DoVisualCheck = DoVisualCheck;
cfg.Nvisual = Nvisual;

outDir = char(cfg.OutDir);
prefix = char(cfg.Prefix);

rng(cfg.Seed);

N       = cfg.N;
B0min   = cfg.B0Range(1); B0max = cfg.B0Range(2);
dBmin   = cfg.dBRange(1); dBmax = cfg.dBRange(2);
p3min   = cfg.pRange(1);  p3max = cfg.pRange(2);
B1      = cfg.BWindow(1); B2 = cfg.BWindow(2);
Npoints = cfg.Npoints;
dBthr   = cfg.dBThr;

% Optional geometry field ('plate' default, 'sphere' for nanocomposites)
if isfield(cfgDS, 'geometry') && ~isempty(cfgDS.geometry)
    geometry = char(cfgDS.geometry);
else
    geometry = 'plate';
end

%% --------------------- Decide mode ---------------------
isNarrow = (dBmax < dBthr);
isWide   = (dBmin > dBthr);

if ~(isNarrow || isWide)
    error(['dBRange intersects the threshold %.1f G. ' ...
           'Choose dBRange fully < %.1f G (narrow) OR fully > %.1f G (wide).'], dBthr, dBthr);
end
mode = ternary(isNarrow,'narrow','wide');

% In wide mode, the 3rd label represents alpha and must be non-negative.
if isWide
    if p3max < 0
        error('Wide mode expects alpha>=0 but pRange upper bound is < 0.');
    end
    p3min = max(0, p3min);
end

%% --------------------- Visible axis ---------------------
% Visible axis is always B>0 with exactly Npoints samples.
Bvis = linspace(max(0,B1), B2, Npoints);
dBstepVis = (Bvis(end)-Bvis(1))/(numel(Bvis)-1); 

%% --------------------- Allocate outputs ---------------------
X = zeros(N, Npoints, 'single');
y = zeros(N, 3, 'single');

%% --------------------- Output paths ---------------------
% Output directory handling:
% If OutDir is empty (the default), resolve from config/paths.json to the
% Drive project folder. When SetName is provided, output goes to the
% per-set dataset directory: <DriveProjectDir>/results/<SetName>/cnn/dataset/.
DriveProjectDir = fullfile(cfgPaths.drive_root_mac, cfgPaths.project_subdir);
if isempty(outDir)
    if ~isempty(SetName)
        outDir = fullfile(DriveProjectDir, 'results', SetName, 'cnn', 'dataset');
    else
        outDir = DriveProjectDir;
    end
elseif ~exist(outDir,'dir')
    warning('outDir does not exist: %s\nSaving to Drive project folder: %s', ...
            outDir, DriveProjectDir);
    if ~isempty(SetName)
        outDir = fullfile(DriveProjectDir, 'results', SetName, 'cnn', 'dataset');
    else
        outDir = DriveProjectDir;
    end
end
if ~exist(outDir,'dir')
    mkdir(outDir);
end

%% --------------------- Main loop ---------------------
for i = 1:N
    % Sample parameters (uniform)
    B0 = randUniform(B0min, B0max);
    dB = randUniform(dBmin, dBmax);
    p3 = randUniform(p3min, p3max);  % p (narrow) OR alpha (wide)

    % 1) Physics: clean derivative signal on visible axis
    if isNarrow
        sig = dysonNarrow(Bvis, B0, dB, p3, geometry);
    else
        sig = dysonWideJoshi_visible_dIdB(B2, Npoints, B0, dB, p3);
    end

    % 2) Polarity fix (unify sign convention): left lobe positive
    sig = polarityFix(sig);

    % 3) Augmentations (after physics)
    if cfg.DoAugment
        sig = applyAugmentations(sig, Bvis, cfg);
    end

    % 4) Robust normalization
    sig = normalizeP2P(sig);

    % Store
    X(i,:) = single(sig);
    y(i,:) = single([B0, dB, p3]);
end

%% --------------------- Visual sanity check ---------------------
if cfg.DoVisualCheck
    rng('shuffle');  % ensure random choice each run

    Nshow = min(cfg.Nvisual, size(X,1));
    idx = randperm(size(X,1), Nshow);

    figure('Name','DysonGeneratorMix – Visual Check','Color','w');
    tiledlayout(ceil(Nshow/2), 2, 'Padding','compact','TileSpacing','compact');

    for k = 1:Nshow
        i = idx(k);
        nexttile;
        plot(Bvis, X(i,:), 'b', 'LineWidth', 1);
        grid on;

        title(sprintf('#%d | B0=%.1f G, dB=%.1f G, p3=%.3f', ...
            i, y(i,1), y(i,2), y(i,3)), ...
            'FontSize', 9);

        xlabel('B (G)');
        ylabel('dI/dB (arb. units)');
    end

    sgtitle(sprintf('Mode: %s | Random %d spectra (visible B>0)', ...
        mode, Nshow), 'FontWeight','bold');
end

%% --------------------- Save outputs ---------------------
meta = struct();
meta.created_utc = char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
if ~isempty(SetName); meta.set_name = SetName; end
meta.mode = mode;
meta.units = 'G';
meta.N = N;
meta.Npoints = Npoints;
meta.B_window_G = [B1, B2];
meta.B_visible_G = [Bvis(1), Bvis(end)];
meta.dB_thr_G = dBthr;
meta.B0_range_G = [B0min, B0max];
meta.dB_range_G = [dBmin, dBmax];
meta.third_param_range = [p3min, p3max];
meta.third_param_meaning = ternary(isNarrow,'p','alpha');
meta.geometry = geometry;
meta.visible_only = true;
meta.derivative = 'first';
meta.augment_enabled = cfg.DoAugment;

xFile = fullfile(outDir, sprintf('X_dyson_mix_%s', prefix));
yFile = fullfile(outDir, sprintf('y_dyson_mix_%s', prefix));
bFile = fullfile(outDir, sprintf('B_axis_mix_%s', prefix));
jFile = fullfile(outDir, sprintf('meta_mix_%s.json', prefix));

savedAs = saveOutputs(xFile, yFile, bFile, jFile, X, y, single(Bvis), meta);

fprintf('Done. Mode: %s. Saved as: %s\n', mode, savedAs);
fprintf('Saved files to: %s\n', outDir);
fprintf('X: %s.npy\n', xFile);
fprintf('y: %s.npy\n', yFile);
fprintf('B: %s.npy\n', bFile);
fprintf('meta: %s\n', jFile);

%% ========================= Functions ===========================

function out = ternary(cond, a, b)
% Simple inline ternary operator.
if cond; out = a; else; out = b; end
end

function v = randUniform(a,b)
% Uniform random scalar in [a,b].
v = a + (b-a)*rand();
end

function sigVis = dysonWideJoshi_visible_dIdB(B2, Npoints, B0, dB, alpha)
% Modified Dyson (Joshi) for wide lines.
%
% Computes absorption-like function P(B) on a symmetric field axis:
%   B_full = linspace(-B2, +B2, 2*Npoints)
% then returns the derivative dP/dB restricted to the visible half B>0,
% with exactly Npoints samples.
%
% Joshi absorption model:
%   P ~ [ (dB + alpha*(B-B0)) / (4(B-B0)^2 + dB^2) ] +
%       [ (dB - alpha*(B+B0)) / (4(B+B0)^2 + dB^2) ]
% The sign flip in the second term makes P(B) even around B=0.

alpha = max(alpha, 0);
dB = max(dB, eps);

Bfull = linspace(-B2, +B2, 2*Npoints);
dBstep = (Bfull(end)-Bfull(1))/(numel(Bfull)-1);

num1 = (dB + alpha*(Bfull - B0));
den1 = (4*(Bfull - B0).^2 + dB^2);

num2 = (dB - alpha*(Bfull + B0));
den2 = (4*(Bfull + B0).^2 + dB^2);

P = (num1 ./ den1) + (num2 ./ den2);

% EPR measures the first derivative
dP = gradient(P, dBstep);

% Visible half (B > 0)
idx = Bfull > 0;
sigVis = dP(idx);

% Safety: enforce exact length
if numel(sigVis) ~= Npoints
    Bvis = linspace(0, B2, Npoints);
    sigVis = interp1(Bfull(idx), sigVis, Bvis, 'linear', 'extrap');
end
end

function sig = polarityFix(sig)
% Enforce a consistent lobe sign convention.
% Convention: the first extremum (in index order) should be a maximum.

[~, iMax] = max(sig);
[~, iMin] = min(sig);

% If minimum comes first -> flip sign
if iMin < iMax
    sig = -sig;
end
end

function sig = applyAugmentations(sig, B, cfg)
% Apply realistic distortions to mimic experimental EPR.
N = numel(sig);

% (A) Modulation/lock-in smoothing (probabilistic)
if rand() < cfg.ModSmoothProb
    sigmaPts = randUniform(cfg.ModSmoothSigmaPts(1), cfg.ModSmoothSigmaPts(2));
    sig = gaussSmooth1D(sig, sigmaPts);
end

% (B) Field calibration warp: scale + offset, then resample back
Bscale  = randUniform(cfg.BScaleRange(1), cfg.BScaleRange(2));
Boffset = randUniform(cfg.BOffsetRange(1), cfg.BOffsetRange(2));
Bwarp = Bscale .* B + Boffset;
sig = interp1(Bwarp, sig, B, 'linear', 'extrap');

% (C) Baseline: offset + slope + quadratic curvature (on normalized t)
b0 = randUniform(cfg.BaselineOffsetRange(1), cfg.BaselineOffsetRange(2));
b1 = randUniform(cfg.BaselineSlopeRange(1),  cfg.BaselineSlopeRange(2));
b2 = randUniform(cfg.BaselineQuadRange(1),   cfg.BaselineQuadRange(2));
t = linspace(-1, 1, N);
baseline = b0 + b1*t + b2*(t.^2);
sig = sig + baseline;

% (D) Additive white noise using target SNR (dB)
snrDb = randUniform(cfg.SNRdBRange(1), cfg.SNRdBRange(2));
sigPower = mean(sig.^2);
noisePower = sigPower / (10^(snrDb/10));
sig = sig + sqrt(noisePower) * randn(size(sig));

% (E) Random spikes (rare glitches)
if rand() < cfg.SpikeProb
    kmin = round(cfg.SpikeCountRange(1));
    kmax = round(cfg.SpikeCountRange(2));
    k = randi([max(1,kmin) max(1,kmax)]);
    for j = 1:k
        pos = randi([1 N]);
        amp = randUniform(cfg.SpikeAmpRange(1), cfg.SpikeAmpRange(2));
        width = randi([1 5]); % spike half-width in points
        lo = max(1, pos - width);
        hi = min(N, pos + width);
        shape = exp(-((lo:hi)-pos).^2/(2*(0.7*width+0.5)^2));
        sig(lo:hi) = sig(lo:hi) + amp * shape .* sign(randn());
    end
end

end

function y = gaussSmooth1D(x, sigmaPts)
% Simple 1D Gaussian smoothing (sigma in points).
sigmaPts = max(sigmaPts, 0.01);
halfW = ceil(4*sigmaPts);
k = (-halfW:halfW);
g = exp(-(k.^2)/(2*sigmaPts^2));
g = g / sum(g);
y = conv(x, g, 'same');
end

function sig = normalizeP2P(sig)
% Peak-to-peak normalization (robust).
mx = max(sig);
mn = min(sig);
p2p = mx - mn;
if p2p < 1e-12
    sig = sig * 0;
else
    sig = sig / p2p;
end
end

function savedAs = saveOutputs(xFile, yFile, bFile, jFile, X, y, Bvis, meta)
% Save dataset to .npy (preferred) using writeNPY.
% Falls back to Python + NumPy, otherwise saves .mat.

    % Save meta JSON 
    fid = fopen(jFile, 'w');
    if fid < 0
        error('Cannot write meta file: %s', jFile);
    end
    fwrite(fid, jsonencode(meta, "PrettyPrint", true), 'char');
    fclose(fid);

    % Preferred: npy-matlab (no Python dependency) 
    if exist('writeNPY','file') == 2
        writeNPY(single(X), [xFile '.npy']);
        writeNPY(single(y), [yFile '.npy']);
        writeNPY(single(Bvis(:)), [bFile '.npy']);
        savedAs = '.npy';
        return;
    end

    % Secondary: Python + NumPy 
    try
        np = py.importlib.import_module('numpy');
        np.save([xFile '.npy'], np.array(single(X), pyargs('dtype','float32')));
        np.save([yFile '.npy'], np.array(single(y), pyargs('dtype','float32')));
        np.save([bFile '.npy'], np.array(single(Bvis(:)), pyargs('dtype','float32')));
        savedAs = '.npy';
        return;
    catch
        % Last resort: MAT 
        warning('Python+NumPy not available and writeNPY not found. Saving as .mat instead.');
        matFile = [xFile '.mat'];  % base name only; contains all variables
        save(matFile, 'X','y','Bvis','meta','-v7.3');
        savedAs = '.mat';
    end
end
