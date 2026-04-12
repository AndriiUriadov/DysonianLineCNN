function results = FitEasyspin(setName, spectrumId)
%% FitEasyspin — Single-spectrum Dysonian fit using EasySpin esfit
%
% FitEasyspin(setName, spectrumId)
%   setName     — e.g. 'set-1'
%   spectrumId  — e.g. '1' (basename without .DTA)
%
% Uses esfit (Levenberg–Marquardt) with two-step approach:
%   Step 1: fix p, fit B0/dBpp/A/off
%   Step 2: fix baseline, fit B0/dBpp/p/A
%
% Output:
%   results/set-N/easyspin/<id>_fit.json — fit parameters
%   results/set-N/easyspin/<id>_fit.png  — overlay plot
%
% Authors: A.V. Uriadov, D.V. Savchenko
% National Technical University of Ukraine
% "Igor Sikorsky Kyiv Polytechnic Institute"
% -----------------------------------------------------------------

if nargin < 2
    error('Usage: FitEasyspin(setName, spectrumId). Example: FitEasyspin(''set-1'', ''1'')');
end

setName    = char(setName);
spectrumId = char(spectrumId);

%% Resolve paths
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(thisDir));

dataDir    = fullfile(repoRoot, 'data', setName);
resultsDir = fullfile(repoRoot, 'results', setName, 'easyspin');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

dtaFile = fullfile(dataDir, [spectrumId '.DTA']);
assert(isfile(dtaFile), 'DTA file not found: %s', dtaFile);

%% Load EasySpin
if exist('esfit','file') ~= 2 || exist('eprload','file') ~= 2
    addpath('/Users/a.uriadov/Documents/easyspin-6.0.11/easyspin');
    rehash toolboxcache;
end

%% Load spectrum
[B, spc] = eprload(dtaFile);
B   = double(B(:));
spc = double(spc(:));

%% Read mwFreq from .DSC
mwFreqGHz = 9.86;
got_mwFreqGHz = false;
dscPath = fullfile(dataDir, [spectrumId '.DSC']);
if isfile(dscPath)
    txt = fileread(dscPath);
    freqPattern = regexp(txt, 'FrequencyMon\s+([\d\.Ee\+\-]+)\s*GHz', 'tokens', 'once');
    if ~isempty(freqPattern)
        mwFreqGHz = str2double(freqPattern{1});
        got_mwFreqGHz = true;
    end
end

%% Fit window and baseline
fitWin = [min(B), max(B)];
mask   = (B >= fitWin(1)) & (B <= fitWin(2));
Bb = B(mask);  yb = spc(mask);

t = (Bb - mean(Bb));
M = [ones(size(t)), t, t.^2];
c = M \ yb;
y0 = yb - M*c;
signfix = sign(max(y0)) * sign(abs(min(y0)) - abs(max(y0)));
if signfix > 0
    y0 = -y0;
    yb = -yb;
end

%% Initial estimates
[ypk, ipk] = max(y0);
[ymn, imn] = min(y0);
Bpk = Bb(ipk);  Bmn = Bb(imn);

i1 = min(ipk,imn); i2 = max(ipk,imn);
zeroIdx = i1 + find(y0(i1:i2).*circshift(y0(i1:i2),-1) < 0, 1,'first') - 1;
if ~isempty(zeroIdx)
    B0_init = interp1(y0(zeroIdx:zeroIdx+1), Bb(zeroIdx:zeroIdx+1), 0,'linear','extrap');
else
    B0_init = Bb(round((ipk+imn)/2));
end

dBpp_init = abs(Bpk - Bmn);
Ramp   = max(abs(ypk), 1e-9) / max(abs(ymn), 1e-9);
p_init = min(4.0, max(0.2, 0.9 + 0.8*(Ramp-1)));
A_init = (ypk - ymn);

Bstep = median(diff(Bb));
Bmod  = min(1.0, 0.03*dBpp_init/sqrt(3));

%% Bounds
lb = [fitWin(1)+20, 5,    0.1, 0,   -Inf, -Inf, -Inf, 0.995, -10];
ub = [fitWin(2)-20, 1000, 5.0, Inf,  Inf,  Inf,  Inf, 1.010,  10];

%% Full model function (9 params)
model_fun = @(p,Bvec) dyson_with_baseline( ...
    Bvec, p(1), p(2), p(3), p(4), p(5), p(6), p(7), ...
    p(8), p(9), Bmod, Bstep);

%% esfit options
FitOpt = struct();
FitOpt.Method    = 'levmar';
FitOpt.AutoScale = 'none';
FitOpt.Display   = 'off';
FitOpt.BaseLine  = 0;
FitOpt.MaxIter   = 200;
FitOpt.TolStep   = 1e-10;
FitOpt.TolFun    = 1e-10;

% Mask: wings only (exclude zero-crossing region)
Bz1 = B0_init - 0.6*dBpp_init;
Bz2 = B0_init + 0.6*dBpp_init;
maskFit = (Bb >= (B0_init-6*dBpp_init) & Bb <= (B0_init+6*dBpp_init)) ...
          & ~(Bb >= Bz1 & Bb <= Bz2);
maskFit = logical(maskFit(:));
FitOpt.Mask = maskFit;

%% Step 1: fix p, fit B0/dBpp/A/off
p_fixed = p_init;
offMax = 0.005 * max(abs(yb));

esfit_fcn1 = @(u) dyson_with_baseline( ...
    Bb, u(1), u(2), p_fixed, u(3), ...
    u(4), 0, 0, 1.0, 0.0, Bmod, Bstep);

u0  = [B0_init, dBpp_init, A_init, 0];
lb1 = [lb(1), lb(2), 0, -offMax];
ub1 = [ub(1), ub(2), Inf, offMax];

fit1 = esfit(yb, esfit_fcn1, u0, lb1, ub1, FitOpt);
uf   = fit1.pfit(:).';

B0_1 = uf(1); dBpp_1 = uf(2); A_1 = uf(3); off_1 = uf(4);

%% Step 2: fix baseline, fit B0/dBpp/p/A
esfit_fcn2 = @(q) dyson_with_baseline( ...
    Bb, q(1), q(2), q(3), q(4), ...
    off_1, 0, 0, 1.0, 0.0, Bmod, Bstep);

p0_step2 = min(ub(3), max(lb(3), 0.85*p_fixed));
q0 = [B0_1, dBpp_1, p0_step2, A_1];
lb2 = [lb(1), lb(2), lb(3), 0];
ub2 = [ub(1), ub(2), ub(3), Inf];

fit2 = esfit(yb, esfit_fcn2, q0, lb2, ub2, FitOpt);
qf   = fit2.pfit(:).';

pf = [qf(1), qf(2), qf(3), qf(4), off_1, 0, 0, 1.0, 0.0];
yfit = model_fun(pf, Bb);

B0 = pf(1); dB = pf(2); p = pf(3); A = pf(4);

%% g-factor
muB = 9.2740100783e-24;
h   = 6.62607015e-34;
g_est = (h*(mwFreqGHz*1e9)) / (muB*(B0*1e-4));

%% RMSD
rmsd = sqrt(mean((yfit - yb).^2));

%% Save JSON
results = struct();
results.spectrum_id   = spectrumId;
results.set_name      = setName;
results.method        = 'easyspin_levmar';
results.B0_G          = B0;
results.dB_G          = dB;
results.p             = p;
results.A             = A;
results.g_factor      = g_est;
results.mwFreq_GHz    = mwFreqGHz;
results.rmsd          = rmsd;
results.Npoints       = numel(Bb);

jsonFile = fullfile(resultsDir, [spectrumId '_fit.json']);
fid = fopen(jsonFile, 'w');
fwrite(fid, jsonencode(results, 'PrettyPrint', true), 'char');
fclose(fid);

%% Plot
fig = figure('Visible','off','Color','w','Position',[140 180 1000 540]);
ax = axes(fig);
ax.Color = 'w'; ax.XColor = 'k'; ax.YColor = 'k';
ax.GridColor = [0.8 0.8 0.8];
grid(ax,'on'); hold(ax,'on');

plot(Bb, yb, 'g-', 'LineWidth', 1.0);
plot(Bb, yfit, 'r-', 'LineWidth', 1.2);
xlabel('Magnetic field B (G)'); ylabel('dI/dB (a.u.)');

if got_mwFreqGHz
    titleStr = sprintf('%s/%s | B0=%.2f G, dB=%.2f G, p=%.3f | g=%.5f', ...
        setName, spectrumId, B0, dB, p, g_est);
else
    titleStr = sprintf('%s/%s | B0=%.2f G, dB=%.2f G, p=%.3f', ...
        setName, spectrumId, B0, dB, p);
end

title(ax, titleStr, 'FontWeight','bold', 'Interpreter','none', 'Color','k');

lgd = legend(ax, 'Experimental', 'EasySpin fit (LM)', 'Location','best');
lgd.Color = 'w'; lgd.TextColor = 'k'; lgd.EdgeColor = 'k';

pngFile = fullfile(resultsDir, [spectrumId '_fit.png']);
saveas(fig, pngFile);
close(fig);

fprintf('[FitEasyspin] %s/%s: B0=%.2f dB=%.2f p=%.3f g=%.5f RMSD=%.3g\n', ...
    setName, spectrumId, B0, dB, p, g_est, rmsd);
end

%% ========================= Local functions =========================

function y = dyson_with_baseline(B, B0, dBpp, p, A, off, slope, quad, Bscale, Bshift, Bmod, Bstep)
    Bcorr = Bshift + Bscale .* B;
    x = 2*(Bcorr - B0)/(sqrt(3)*dBpp);

    % Full formula with two terms (Holiatkina et al., J. Appl. Phys. 134, 125306 (2023))
    denom1 = 2*p.*(cosh(p) + cos(p));
    denom2 = (cosh(p) + cos(p)).^2;
    Acoef = (sinh(p) + sin(p))./denom1 + (1 + cosh(p).*cos(p))./denom2;
    Dcoef = (sinh(p) - sin(p))./denom1 + (sinh(p).*sin(p))./denom2;
    yclean = (-Acoef.*2.*x) ./ (1 + x.^2).^2 + Dcoef.*(1 - x.^2) ./ (1 + x.^2).^2;

    if Bmod > 0
        sigma = (Bmod/2.355) / max(Bstep,eps);
        k = max(1, ceil(6*sigma));
        xi = (-k:k);
        g = exp(-0.5*(xi./sigma).^2); g = g/sum(g);
        yclean = conv(yclean, g, 'same');
    end

    t = (B - mean(B));
    y = A*yclean + off + slope*t + quad*(t.^2 - mean(t.^2));
end
