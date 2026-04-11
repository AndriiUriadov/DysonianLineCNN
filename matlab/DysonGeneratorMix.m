% ------------------------ DysonGeneratorMix.m ----------------------------
%
% Synthetic EPR Dyson / Modified-Dyson (Joshi) Dataset Generator
%
% -------------------------------------------------------------------------
%
% Authors: A.V.Uriadov, D.V.Savchenko
% National Technical University of Ukraine
% "Igor Sikorsky Kyiv Polytechnic Institute"
%
% -------------------------------------------------------------------------
%
% PURPOSE
%   Generate synthetic EPR spectra (first-derivative signal) for training
%   CNN models on Dysonian line parameters. The generator automatically
%   selects the *narrow* (standard Dyson, parameter p) or *wide* (modified
%   Dyson by Joshi, parameter alpha) model based on the requested dB range:
%       - Narrow-only:  dB_max < dB_thr (default dB_thr = 1000 G = 100 mT)
%       - Wide-only:    dB_min > dB_thr
%       - Mixed range:  error (choose either narrow-only or wide-only)
%
% KEY CONVENTIONS
%   - Units: All field-related quantities are in Gauss (G).
%   - Threshold: dB_thr = 1000 G.
%   - Output spectra: "visible right half" only (B > 0) with Npoints samples.
%   - Output signal: first derivative dI/dB (as in EPR spectrometers).
%   - Labels y = [B0, dB, p3]:
%       * Narrow mode: p3 = physical p (skin-effect parameter)
%       * Wide mode:   p3 = alpha (>=0) used in Joshi model
%     (The 3rd label is a generic "asymmetry parameter")
%
% PIPELINE (UNIFIED FOR NARROW & WIDE)
%   physics -> polarity fix -> (optional) modulation smoothing ->
%   field scale/offset warp + resample -> baseline -> noise -> spikes ->
%   robust normalization
%
% OUTPUT FILES (NEW NAMES)
%   - X_dyson_mix_<Prefix>.npy (or .mat): [N x Npoints] float32
%   - y_dyson_mix_<Prefix>.npy (or .mat): [N x 3] float32
%   - B_axis_mix_<Prefix>.npy  (or .mat): [1 x Npoints] float32 (B>0 axis)
%   - meta_mix_<Prefix>.json: generation metadata (mode, ranges, etc.)
%
% NOTES
%   - Saving as .npy requires MATLAB Python integration + NumPy installed.
%     If not available, the generator falls back to .mat.
%   - For strict reproducibility, 'Seed' is fixed and stored in meta JSON.
%
% -------------------------------------------------------------------------

%% --------------------- Load config ---------------------
% All tunable defaults come from config/dataset.json so that MATLAB and
% Python share a single source of truth. The script is still callable
% with name/value overrides via inputParser for quick experiments.
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);  % ensure load_config is discoverable
cfgDS    = load_config('dataset');
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
% Drive project folder: fullfile(drive_root_mac, project_subdir). If OutDir
% is explicitly provided but does not exist, fall back to the Drive folder
% rather than writing next to the script (where files would clutter the git
% repo working tree).
DriveProjectDir = fullfile(cfgPaths.drive_root_mac, cfgPaths.project_subdir);
if isempty(outDir)
    outDir = DriveProjectDir;
elseif ~exist(outDir,'dir')
    warning('outDir does not exist: %s\nSaving to Drive project folder: %s', ...
            outDir, DriveProjectDir);
    outDir = DriveProjectDir;
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
        sig = dysonNarrow_dIdB(Bvis, B0, dB, p3);
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

function sig = dysonNarrow_dIdB(B, B0, dBpp, p)
% Standard Dysonian first-derivative for narrow lines.
% Normalized variable for Lorentzian with peak-to-peak linewidth dBpp:
%   x = 2(B-B0)/(sqrt(3)*dBpp)
% Mixture coefficients A(p), D(p) follow Dyson/Feher skin-effect approach.
% Output is an unnormalized derivative signal.

dBpp = max(dBpp, eps);
x = 2*(B - B0) ./ (sqrt(3)*dBpp);

% Dyson coefficients (compact form)
[Acoef, Dcoef] = dysonAD_from_p(p);

den = (1 + x.^2).^2;

% Derivative terms (shape functions)
termAbs  = (2*x) ./ den;          % ~ d/dB absorption
termDisp = (1 - x.^2) ./ den;     % ~ d/dB dispersion

% Overall sign is a convention; polarityFix() later unifies it.
sig = (-Acoef .* termAbs) + (Dcoef .* termDisp);
end

function [Acoef, Dcoef] = dysonAD_from_p(p)
% Compute Dyson mixing coefficients A(p), D(p).
%
% p = lambda/delta (or 2d/delta), lambda is sample thickness, delta skin depth.
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
