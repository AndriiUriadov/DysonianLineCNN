%% ========================================================================
%  Bruker EPR Inspector + Dysonian Derivative Fit
%  (EasySpin / Standalone modes)
%  (Feher–Kip metallic regime, plate geometry)
%
%  ОПИС:
%  Скрипт для перегляду та аналізу експериментальних EPR-спектрів Bruker
%  з автоматичним фітом похідної Dysonian-лінії (модель Feher–Kip).
%  Підтримуються режим оптимізації:
%    – EasySpin (esfit, Levenberg–Marquardt);
%
%  INPUT:
%    – .DTA (обов'язково)
%    – .DSC (опційно, для зчитування mwFreq через FrequencyMon)
%    – файли мають знаходитись у тій самій директорії, що і скрипт
%
%  OUTPUT:
%    – PNG-файл накладання:
%        "<ім'я>_fit_Easyspin.png"  (EasySpin mode)
%    – у консоль виводяться оцінки B0, dBpp, p, A та g-фактор
%
%  МОДЕЛЬ (Dysonian derivative):
%    – B0 (G)    : резонансне магнітне поле
%    – dBpp (G)  : linewidth
%    – p (-)     : параметр асиметрії 
%    – A (a.u.)  : амплітуда
%    – baseline  : off / slope / quad
%    – калібрування осі: Bscale / Bshift
%
%  ОПТИМІЗАЦІЯ:
%      – esfit (Levenberg–Marquardt)
%      – підтримка масок, двокроковий фіт (p → {B0,dBpp,A} → p)
%
%  ВИМОГИ:
%    – EasySpin (eprload, esfit) 
%
%  ПЗ:
%    – MATLAB R2025b (macOS)
%    – Optimization Toolbox 25.2
%
%  АВТОРИ:
%    A.V. Uriadov, D.V. Savchenko
%
%  National Technical University of Ukraine
%  "Igor Sikorsky Kyiv Polytechnic Institute"
% ========================================================================

clear; clc;

%% 0. Режим фіту 
%fitMode = 'easyspin';   % EasySpin esfit 

% EasySpin path safety (щоб esfit/eprload завжди знаходились)
if exist('esfit','file')~=2 || exist('eprload','file')~=2
    addpath('/Users/a.uriadov/Documents/easyspin-6.0.11/easyspin');
    rehash toolboxcache;
end

%% 1. Вказуємо назву файлу DTA (без шляху, тільки ім'я) 
% Вважаємо, що файл лежить в тій самій папці, що і скрипт
%                            % <--------------------------------
%filename = '022704.DTA';    % <--- Вказуємо імʼя DTA файлу тут
%filename = '042312.DTA';    % <--------------------------------
filename = '5.DTA';
%
% Знаходимо директорію, де лежить цей скрипт 
scriptDir = fileparts(mfilename('fullpath'));

fpath = fullfile(scriptDir, filename);
assert(isfile(fpath), 'Файл %s не знайдено у цій папці!', filename);


%% 2. Завантажуємо спектр через EasySpin 
fprintf('Відкриваю файл: %s\n', fpath);
[B, spc] = eprload(fpath);
B = double(B(:));
spc = double(spc(:));


%% 3. Виводимо базову інформацію 
fprintf('   Завантажено: %s\n', filename);
fprintf('   Кількість точок: %d\n', numel(B));
fprintf('   Діапазон поля: %.1f – %.1f G\n', min(B), max(B));
fprintf('   Інтенсивність: %.3g – %.3g (a.u.)\n', min(spc), max(spc));


%% 4. Обрізаємо магнітне вікно під фіт
fitWin = [min(B), max(B)]; % або зафіксувати магнітне вікно: [2500 6500]
% fitWin = [3300 3400]; % можна зафіксувати магнітне вікно 
mask   = (B >= fitWin(1)) & (B <= fitWin(2));
Bb = B(mask);  yb = spc(mask);


%% 5. Грубе вирівнювання базової лінії (offset + slope + quad)
% важливо: y0 - тільки для визначення стартових параметрів, фіт по yb 
t = (Bb - mean(Bb)); %
M = [ones(size(t)), t, t.^2];
c = M \ yb;      % найменші квадрати
y0 = yb - M*c;   % зняли базу (щоб не заважала B0)
signfix = sign(max(y0)) * sign(abs(min(y0)) - abs(max(y0))); % crude polarity check
if signfix > 0 
   y0 = -y0; 
   yb = -yb;    % інколи треба інвертувати  
end              


%% 6. Початкові оцінки B0, dB, p
% максимум/мінімум та їх положення
[ypk, ipk] = max(y0);
[ymn, imn] = min(y0);
Bpk = Bb(ipk);  Bmn = Bb(imn);

% нуль між піком і мінімумом (лінійна інтерполяція)
i1 = min(ipk,imn); i2 = max(ipk,imn);
zeroIdx = i1 + find(y0(i1:i2).*circshift(y0(i1:i2),-1) < 0, 1,'first') - 1;
if ~isempty(zeroIdx)
    B0_init = interp1(y0(zeroIdx:zeroIdx+1), Bb(zeroIdx:zeroIdx+1), 0,'linear','extrap');
else
    B0_init = Bb(round((ipk+imn)/2));  % запасний варіант
end

% оцінка ширини: dBpp = |Bpk - Bmn|; (для лоренца dB = dBpp/√3)
dBpp_init = abs(Bpk - Bmn);      % це dBpp
dB_init   = dBpp_init;           % щоб не плутатись

% оцінка p за асиметрією R = |max|/|min| (груба емпірика для старту)
Ramp   = max(abs(ypk), 1e-9) / max(abs(ymn), 1e-9);
p_init = min(4.0, max(0.2, 0.9 + 0.8*(Ramp-1)));  % просто стартова точка

% амплітуда після нормування бази
A_init = (ypk - ymn);
off_init = 0; slope_init = 0; quad_init = 0;

% нові параметри калібрування осі поля
Bscale_init = 1.0;   % ~1
Bshift_init = 0.0;   % ~0 Гс

%    [ B0       dBpp       p       A       off       slope       quad       Bscale       Bshift ]
p0 = [ B0_init, dBpp_init, p_init, A_init, off_init, slope_init, quad_init, Bscale_init, Bshift_init ];

% (Bscale у вузькому коридорі ±1%, Bshift ±10 G):
lb = [fitWin(1)+20,   5,    0.1,   0,   -Inf,   -Inf,   -Inf,   0.995,  -10];
ub = [fitWin(2)-20, 1000,   5.0,  Inf,   Inf,    Inf,    Inf,   1.010,   10];


%% 7. Фіт Dyson (Feher–Kip, plate geometry) + невелике розмиття модуляцією
mwFreqGHz = 9.86; % <-- встановити вручну якщо немає файла .DSC

Bstep = median(diff(Bb));

if exist('dBpp_init','var') && ~isempty(dBpp_init)
    dB_forMod = dBpp_init/sqrt(3);
else
    dB_forMod = dB_init;   % запасний варіант, щоб код не впав
end
Bmod = min(1.0, 0.03*dB_forMod);

% mwFreq з DSC
dscPath = fullfile(scriptDir, [erase(filename, '.DTA') '.DSC']);
got_mwFreqGHz = false;
if isfile(dscPath)
    txt = fileread(dscPath);
    freqPattern = regexp(txt, 'FrequencyMon\s+([\d\.Ee\+\-]+)\s*GHz', 'tokens', 'once');
    if ~isempty(freqPattern)
        mwFreqGHz = str2double(freqPattern{1});
        fprintf('Мікрохвильова частота зчитана з DSC: %.6f GHz\n', mwFreqGHz);
        got_mwFreqGHz = true;
    else
        fprintf('Не знайдено FrequencyMon у DSC, використано стандартне значення %.3f GHz\n', mwFreqGHz);
    end
else
    fprintf('Файл DSC не знайдено, використано значення за замовчуванням %.3f GHz\n', mwFreqGHz);
end

% Повна модель для standalone (9 параметрів)
model_fun = @(p,Bvec) dyson_with_baseline( ...
    Bvec, p(1), p(2), p(3), p(4), p(5), p(6), p(7), ...
    p(8), p(9), Bmod, Bstep);

res_fun = @(p) model_fun(p,Bb) - yb;

% Маска
w = 5;
B1 = B0_init - w*dBpp_init;
B2 = B0_init + w*dBpp_init;
maskFit = (Bb >= B1) & (Bb <= B2);
maskFit = logical(maskFit(:));
fprintf('Mask: %d/%d points used (%.1f%%)\n', ...
    nnz(maskFit), numel(maskFit), 100*nnz(maskFit)/numel(maskFit));

% Оцінка масштабу quad
BbRange = max(Bb) - min(Bb);
BbRange = max(BbRange, eps);
quadMax = 0.002 * max(abs(yb)) / (BbRange^2);


% FitOpt
FitOpt = struct();
FitOpt.Method    = 'levmar';
FitOpt.AutoScale = 'none';
FitOpt.Display   = 'off';     % для дебагу: 'iter'
FitOpt.BaseLine  = 0;         % baseline всередині 
FitOpt.MaxIter = 200;         % або 400
FitOpt.TolStep = 1e-10;       % було ~1e-6, зменшуємо
FitOpt.TolFun  = 1e-10;       % 

Bz1 = B0_init - 0.6*dBpp_init;
Bz2 = B0_init + 0.6*dBpp_init;
maskFit = (Bb >= (B0_init-6*dBpp_init) & Bb <= (B0_init+6*dBpp_init)) ...
          & ~(Bb >= Bz1 & Bb <= Bz2);
maskFit = logical(maskFit(:));

fprintf('Mask (wings): %d/%d points used (%.1f%%)\n', ...
    nnz(maskFit), numel(maskFit), 100*nnz(maskFit)/numel(maskFit));

FitOpt.Mask = maskFit;

% easyspin оптимізатор "залипає" на p, тож робимо за 2 кроки
% =========================================================
% КРОК 1: p фіксований, фітимо B0, dBpp, A, off
% slope=0, Bscale=1, Bshift=0
% =========================================================
p_fixed = p0(3);

% u = [B0 dBpp A off]
esfit_fcn1 = @(u) dyson_with_baseline( ...
    Bb, u(1), u(2), p_fixed, u(3), ...
    u(4), 0, 0, ...          % off, slope=0, quad=0
    1.0, 0.0, Bmod, Bstep);

u0  = [p0(1) p0(2) p0(4) 0];

offMax = 0.005 * max(abs(yb));   % було 0.02
lb1 = [lb(1) lb(2) 0  -offMax];
ub1 = [ub(1) ub(2) Inf offMax];

fit1 = esfit(yb, esfit_fcn1, u0, lb1, ub1, FitOpt);
uf   = fit1.pfit(:).';

B0_1   = uf(1);
dBpp_1 = uf(2);
A_1    = uf(3);
off_1  = uf(4);
quad_1 = 0; 

fprintf('Step1 (p fixed=%.3f): B0=%.2f dBpp=%.2f A=%.3g off=%.3g quad=%.3g\n', ...
    p_fixed, B0_1, dBpp_1, A_1, off_1, quad_1);



% =========================================================
% КРОК 2: baseline фіксований (off, quad), фітимо B0,dBpp,p,A
% slope=0, Bscale=1, Bshift=0
% =========================================================
% q = [B0 dBpp p A]
esfit_fcn2 = @(q) dyson_with_baseline( ...
    Bb, q(1), q(2), q(3), q(4), ...
    off_1, 0, quad_1, ...
    1.0, 0.0, Bmod, Bstep);

p0_step2 = min(ub(3), max(lb(3), 0.85*p_fixed));   % або 1.15*p_fixed
q0 = [B0_1 dBpp_1 p0_step2 A_1];

lb2 = [lb(1) lb(2) lb(3) 0];
ub2 = [ub(1) ub(2) ub(3) Inf];

fit2 = esfit(yb, esfit_fcn2, q0, lb2, ub2, FitOpt);
qf   = fit2.pfit(:).';

% pf = [B0 dBpp p A off slope quad Bscale Bshift]
pf = [qf(1) qf(2) qf(3) qf(4)  off_1 0 quad_1  1.0 0.0];

yfit = model_fun(pf, Bb);
exitflag = 1;


B0 = pf(1); dB = pf(2); p = pf(3); A = pf(4);
off = pf(5); sl = pf(6); qd = pf(7);
Bscale = pf(8); Bshift = pf(9);

%% 8. Обчислюємо g (= h*ν / (muB * B0))
muB = 9.2740100783e-24; 
h = 6.62607015e-34;
g_est = (h*(mwFreqGHz*1e9)) / (muB*(B0*1e-4));   % 1 Гс = 1e-4 Тл

fprintf('\n=== Dyson fit (Feher–Kip, plate) ===\n');
fprintf('B0 = %.2f G,  dB = %.2f G,  p = %.3f,  A = %.3g\n', B0, dB, p, A);
fprintf('baseline: off=%.3g, slope=%.3g, quad=%.3g\n', off, sl, qd);
fprintf('g = %.5f  (ν = %.5f GHz)\n', g_est, mwFreqGHz);


%% 9. Накладання на графік і збереження
fig = figure('Color','w','Position',[140 180 1000 540]);

% окремо вказуємо параметри фона, заголовків тощо, щоб темна схема не заважала
ax = axes(fig);
ax.Color = 'w';          % фон області графіка
ax.XColor = 'k';         % колір підписів/осьових ліній
ax.YColor = 'k';
ax.GridColor = [0.8 0.8 0.8];  % м'яка сіра сітка
grid(ax,'on'); hold(ax,'on');

plot(Bb, yb, 'g-', 'LineWidth', 1.0); hold on;  % експериментальний
plot(Bb, yfit, 'r-', 'LineWidth', 1.2);         % симульований
grid on; xlabel('Magnetic field B (G)'); ylabel('dI/dB (a.u.)');

if got_mwFreqGHz % якщо прочитали mwFreqGHz з .DSC
    titleStr = sprintf('%s  |  B0=%.2f G, dB=%.2f G, p=%.3f  |  g=%.5f, ν=%.5f GHz', ...
        erase(filename,'.DTA'), B0, dB, p, g_est, mwFreqGHz);
else
    titleStr = sprintf('%s  |  B0=%.2f G, dB=%.2f G, p=%.3f', ...
        erase(filename,'.DTA'), B0, dB, p);
end

t = title(ax, titleStr, ...
    'FontWeight','bold', ...
    'Interpreter','none');

t.Color = 'k';   


lgd = legend(ax, ...
    'Experimental', ...
    'Simulated Dyson (EasySpin fit)', ...
    'Location','best');
outFit = fullfile(scriptDir, [erase(filename,'.DTA') '_fit_Easyspin.png']);

lgd.Color = 'w';       
lgd.TextColor = 'k';   
lgd.EdgeColor = 'k';   

saveas(gcf, outFit);
fprintf('Збережено накладання: %s\n', outFit);


%% 10. Блок допоміжних функцій 
function y = dyson_with_baseline(B, B0, dBpp, p, A, off, slope, quad, Bscale, Bshift, Bmod, Bstep)
    % Каліброване поле
    Bcorr = Bshift + Bscale .* B;

    % Чиста похідна Dyson (Feher–Kip, plate) на Bcorr
    x = 2*(Bcorr - B0)/(sqrt(3)*dBpp);
    Acoef = (sinh(p) + sin(p)) ./ (2*p.*(cosh(p) + cos(p)));
    Dcoef = (sinh(p) - sin(p)) ./ (2*p.*(cosh(p) + cos(p)));
    yclean = (-Acoef.*2.*x) ./ (1 + x.^2).^2 + Dcoef.*(1 - x.^2) ./ (1 + x.^2).^2;

    % Слабке "модульне" розмиття (ефект лок-іну), гаус
    if Bmod > 0
        sigma = (Bmod/2.355) / max(Bstep,eps);
        k = max(1, ceil(6*sigma));
        xi = (-k:k);
        g = exp(-0.5*(xi./sigma).^2); g = g/sum(g);
        yclean = conv(yclean, g, 'same');
    end

    % Базова лінія — обчислюємо на тій же осі, що й спектр (B)
    t = (B - mean(B));
    y = A*yclean + off + slope*t + quad*(t.^2 - mean(t.^2));
end

function p = bound_params(p, lb, ub)
    p = max(p, lb); p = min(p, ub);
end