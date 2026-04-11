%% ========================================================================
%  Bruker EPR Inspector + Dysonian Derivative Fit
%  (EasySpin esfit - Simplex, one-step optimization)
%  (Feher–Kip metallic regime, plate geometry)
%
%  ОПИС:
%  Спрощений скрипт для фіту експериментальних EPR-спектрів Bruker
%  з автоматичним фітом похідної Dysonian-лінії (модель Feher–Kip).
%  Використовує одно кроковий Nelder-Mead simplex алгоритм.
%
%  INPUT:
%    – .DTA (обов'язково)
%    – .DSC (опційно, для зчитування mwFreq через FrequencyMon)
%    – файли мають знаходитись у тій самій директорії, що і скрипт
%
%  OUTPUT:
%    – PNG-файл накладання: "<ім'я>_fit_Easyspin_Simplex.png"
%    – у консоль виводяться оцінки B0, dBpp, p, A та g-фактор
%
%  МОДЕЛЬ (Dysonian derivative):
%    – B0 (G)    : резонансне магнітне поле
%    – dBpp (G)  : linewidth (peak-to-peak)
%    – p (-)     : параметр асиметрії Dyson
%    – A (a.u.)  : амплітуда
%    – baseline  : offset
%
%  ОПТИМІЗАЦІЯ:
%      – esfit з Nelder-Mead simplex (один крок, robust)
%      – AutoScale='lsq' для автоматичного масштабування
%
%  ВИМОГИ:
%    – EasySpin (eprload, esfit)
%
%  ПЗ:
%    – MATLAB R2025b (macOS)
%
%  АВТОРИ:
%    A.V. Uriadov, D.V. Savchenko
%
%  National Technical University of Ukraine
%  "Igor Sikorsky Kyiv Polytechnic Institute"
%
%  ДЖЕРЕЛА:
%    – Holiatkina et al., J. Appl. Phys. 134, 125306 (2023)
%      (повна формула коефіцієнтів A і D для Feher–Kip моделі)
% ========================================================================

clear; clc;

% EasySpin path safety
if exist('esfit','file')~=2 || exist('eprload','file')~=2
    addpath('/Users/a.uriadov/Documents/easyspin-6.0.11/easyspin');
    rehash toolboxcache;
end

%% 1. Вказуємо назву файлу DTA
%filename = '042312.DTA';    % <--- Вказуємо імʼя DTA файлу тут
%filename = '022704.DTA';
filename = '1.DTA';

scriptDir = fileparts(mfilename('fullpath'));
fpath = fullfile(scriptDir, filename);
assert(isfile(fpath), 'Файл %s не знайдено у цій папці!', filename);

%% 2. Завантажуємо спектр через EasySpin
fprintf('Відкриваю файл: %s\n', fpath);
[B, spc] = eprload(fpath);
B = double(B(:));
spc = double(spc(:));

fprintf('   Завантажено: %s\n', filename);
fprintf('   Кількість точок: %d\n', numel(B));
fprintf('   Діапазон поля: %.1f – %.1f G\n', min(B), max(B));
fprintf('   Інтенсивність: %.3g – %.3g (a.u.)\n', min(spc), max(spc));

%% 3. Обрізаємо магнітне вікно під фіт
fitWin = [min(B), max(B)];
mask = (B >= fitWin(1)) & (B <= fitWin(2));
Bb = B(mask);
yb = spc(mask);

%% 4. Грубе вирівнювання базової лінії для оцінки стартових параметрів
t = (Bb - mean(Bb));
M = [ones(size(t)), t, t.^2];
c = M \ yb;
y0 = yb - M*c;   % знімаємо базу для визначення B0

% Перевірка полярності
signfix = sign(max(y0)) * sign(abs(min(y0)) - abs(max(y0)));
if signfix > 0
   y0 = -y0;
   yb = -yb;
end

%% 5. Початкові оцінки B0, dBpp, p
[ypk, ipk] = max(y0);
[ymn, imn] = min(y0);
Bpk = Bb(ipk);
Bmn = Bb(imn);

% Резонансне поле (через нульове перетинання)
i1 = min(ipk,imn);
i2 = max(ipk,imn);
zeroIdx = i1 + find(y0(i1:i2).*circshift(y0(i1:i2),-1) < 0, 1,'first') - 1;
if ~isempty(zeroIdx)
    B0_init = interp1(y0(zeroIdx:zeroIdx+1), Bb(zeroIdx:zeroIdx+1), 0,'linear','extrap');
else
    B0_init = Bb(round((ipk+imn)/2));
end

% Ширина лінії
dBpp_init = abs(Bpk - Bmn);

% Параметр асиметрії p (грубa оцінка з A/B ratio)
Ramp = max(abs(ypk), 1e-9) / max(abs(ymn), 1e-9);
p_init = min(4.0, max(0.2, 0.9 + 0.8*(Ramp-1)));

% Амплітуда
A_init = (ypk - ymn);

fprintf('\nСтартові параметри:\n');
fprintf('  B0 = %.2f G\n', B0_init);
fprintf('  dBpp = %.2f G\n', dBpp_init);
fprintf('  p = %.3f\n', p_init);
fprintf('  A = %.3g\n', A_init);

%% 6. Налаштування модуляції та частоти
Bstep = median(diff(Bb));
Bmod = min(1.0, 0.03*dBpp_init/sqrt(3));

% Зчитуємо мікрохвильову частоту з DSC
mwFreqGHz = 9.86;  % за замовчуванням
dscPath = fullfile(scriptDir, [erase(filename, '.DTA') '.DSC']);
got_mwFreqGHz = false;
if isfile(dscPath)
    txt = fileread(dscPath);
    freqPattern = regexp(txt, 'FrequencyMon\s+([\d\.Ee\+\-]+)\s*GHz', 'tokens', 'once');
    if ~isempty(freqPattern)
        mwFreqGHz = str2double(freqPattern{1});
        fprintf('Мікрохвильова частота зчитана з DSC: %.6f GHz\n', mwFreqGHz);
        got_mwFreqGHz = true;
    end
end

%% 7. Налаштування esfit з simplex алгоритмом
FitOpt = struct();
FitOpt.Method    = 'simplex fcn';  % Nelder-Mead simplex
FitOpt.AutoScale = 'none';         % Без автомасштабування (A - параметр)
FitOpt.Verbosity = 0;              % Тихий режим
FitOpt.BaseLine  = [];             % Baseline в моделі
FitOpt.TolFun    = 1e-5;           % Tolerance для функції
FitOpt.TolEdgeLength = 1e-4;       % Tolerance для розміру simplex
FitOpt.delta     = 0.15;           % Розмір початкового simplex

% Маска - використовуємо тільки область резонансу
maskFit = (Bb >= (B0_init-6*dBpp_init)) & (Bb <= (B0_init+6*dBpp_init));
maskFit = logical(maskFit(:));
FitOpt.Mask = maskFit;

fprintf('\nМаска фіту: %d/%d точок (%.1f%%)\n', ...
    nnz(maskFit), numel(maskFit), 100*nnz(maskFit)/numel(maskFit));

%% 8. Визначення функції моделі
% Параметри: v = [B0, dBpp, p, A, off]
% Зафіксовані: slope=0, quad=0, Bscale=1, Bshift=0
esfit_fcn = @(v) dyson_with_baseline( ...
    Bb, v(1), v(2), v(3), v(4), ...
    v(5), 0, 0, ...          % off, slope=0, quad=0
    1.0, 0.0, Bmod, Bstep);  % Bscale=1, Bshift=0

% Початкові значення
v0 = [B0_init, dBpp_init, p_init, A_init, 0];

% Bounds
offMax = 0.01 * max(abs(yb));
lb_fit = [fitWin(1)+20, 5,    0.2, 0,   -offMax];
ub_fit = [fitWin(2)-20, 1500, 5.0, Inf,  offMax];

%% 9. Запуск фітування
fprintf('\n=== Simplex fit (one-step) ===\n');
fprintf('Fitting...\n');

fitResult = esfit(yb, esfit_fcn, v0, lb_fit, ub_fit, FitOpt);
vf = fitResult.pfit(:).';

B0_fit   = vf(1);
dBpp_fit = vf(2);
p_fit    = vf(3);
A_fit    = vf(4);
off_fit  = vf(5);

% Симулюємо з фінальними параметрами
yfit = esfit_fcn(vf);

fprintf('Fit complete!\n');

%% 10. Обчислюємо g-фактор
muB = 9.2740100783e-24;
h = 6.62607015e-34;
g_est = (h*(mwFreqGHz*1e9)) / (muB*(B0_fit*1e-4));

fprintf('\n=== Dyson fit results (Feher–Kip, plate) ===\n');
fprintf('B0   = %.2f G\n', B0_fit);
fprintf('dB   = %.2f G\n', dBpp_fit);
fprintf('p    = %.3f\n', p_fit);
fprintf('A    = %.3g\n', A_fit);
fprintf('off  = %.3g\n', off_fit);
fprintf('g    = %.5f  (ν = %.5f GHz)\n', g_est, mwFreqGHz);
fprintf('RMSD = %.3g\n', fitResult.rmsd);

%% 11. Графік та збереження
fig = figure('Color','w','Position',[140 180 1000 540]);

ax = axes(fig);
ax.Color = 'w';
ax.XColor = 'k';
ax.YColor = 'k';
ax.GridColor = [0.8 0.8 0.8];
grid(ax,'on');
hold(ax,'on');

plot(Bb, yb, 'g-', 'LineWidth', 1.0);
plot(Bb, yfit, 'r-', 'LineWidth', 1.2);

xlabel('Magnetic field B (G)');
ylabel('dI/dB (a.u.)');

if got_mwFreqGHz
    titleStr = sprintf('%s  |  B0=%.2f G, dB=%.2f G, p=%.3f  |  g=%.5f, ν=%.5f GHz', ...
        erase(filename,'.DTA'), B0_fit, dBpp_fit, p_fit, g_est, mwFreqGHz);
else
    titleStr = sprintf('%s  |  B0=%.2f G, dB=%.2f G, p=%.3f', ...
        erase(filename,'.DTA'), B0_fit, dBpp_fit, p_fit);
end

title(ax, titleStr, 'FontWeight','bold', 'Interpreter','none', 'Color','k');

lgd = legend(ax, 'Experimental', 'Simulated Dyson (EasySpin Simplex)', 'Location','best');
lgd.Color = 'w';
lgd.TextColor = 'k';
lgd.EdgeColor = 'k';

outFit = fullfile(scriptDir, [erase(filename,'.DTA') '_fit_Easyspin_Simplex.png']);
saveas(gcf, outFit);
fprintf('\nЗбережено графік: %s\n', outFit);

%% 12. Допоміжна функція моделі Dyson
function y = dyson_with_baseline(B, B0, dBpp, p, A, off, slope, quad, Bscale, Bshift, Bmod, Bstep)
    % Каліброване поле
    Bcorr = Bshift + Bscale .* B;

    % Чиста похідна Dyson (Feher–Kip, plate) на Bcorr
    x = 2*(Bcorr - B0)/(sqrt(3)*dBpp);

    % Повна формула коефіцієнтів A і D (два доданки, Holiatkina et al., 2023)
    denom1 = 2*p.*(cosh(p) + cos(p));
    denom2 = (cosh(p) + cos(p)).^2;
    Acoef = (sinh(p) + sin(p))./denom1 + (1 + cosh(p).*cos(p))./denom2;
    Dcoef = (sinh(p) - sin(p))./denom1 + (sinh(p).*sin(p))./denom2;

    yclean = (-Acoef.*2.*x) ./ (1 + x.^2).^2 + Dcoef.*(1 - x.^2) ./ (1 + x.^2).^2;

    % Модуляційне розмиття (гаус)
    if Bmod > 0
        sigma = (Bmod/2.355) / max(Bstep,eps);
        k = max(1, ceil(6*sigma));
        xi = (-k:k);
        g = exp(-0.5*(xi./sigma).^2);
        g = g/sum(g);
        yclean = conv(yclean, g, 'same');
    end

    % Базова лінія
    t = (B - mean(B));
    y = A*yclean + off + slope*t + quad*(t.^2 - mean(t.^2));
end
