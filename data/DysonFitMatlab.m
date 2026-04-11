%% ========================================================================
%  Bruker EPR Inspector + Standalone Dysonian Derivative Fit
%  (Feher–Kip metallic regime, plate geometry)
%
%  ОПИС:
%  Скрипт для швидкого перегляду та базового аналізу експериментальних
%  EPR спектрів Bruker. Завантажує .DTA через EasySpin, виконує
%  автоматичний фіт Dysonian-похідної (Feher–Kip) та зберігає графік.
%
%  INPUT:
%    – .DTA (обов'язково), .DSC (опційно для читання mwFreq через FrequencyMon)
%    – файли мають лежати в тій самій папці, що і цей скрипт
%
%  OUTPUT:
%    – PNG-файл накладання: "<ім'я>_fit_Matlab.png" у тій самій директорії
%    – у консоль виводяться оцінки B0, dB, p, A та g (за mwFreq)
%
%  ПАРАМЕТРИ моделі (основні):
%    – B0 (G): резонансне магнітне поле
%    – dB (G): параметр ширини лінії
%    – p (-): параметр асиметрії Dysonian лінії
%    – A (a.u.): амплітуда; baseline: off/slope/quad; калібрування осі: Bscale/Bshift
%
%  ОПТИМІЗАЦІЯ:
%    – lsqnonlin - нелінійний метод найменших квадратів,
%      least-squares по residuals, з bounds - мінімізує суму квадратів 
%      точкових помилок між моделлю і експериментом, не дозволяючи 
%      параметрам вийти за фізично допустимі межі (з Optimization Toolbox)
%    – fallback: fminsearch по MSE + ручне обмеження параметрів у межах lb/ub
%
%  ВИМОГИ:
%    – EasySpin (eprload)
%    – Optimization Toolbox (необов'язково) або базовий fminsearch
%
%  ПЗ:
%    – MATLAB R2025b (macOS)
%    - Optimization Toolbox 25.2
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

%% 1. Вказуємо назву файлу DTA (без шляху, тільки ім'я) 
% Вважаємо, що файл лежить в тій самій папці, що і скрипт
%                            % <--------------------------------
%filename = '022704.DTA';    % <--- Вказуємо імʼя DTA файлу тут
filename = '042312.DTA';    % <--------------------------------
%filename = '6.DTA';
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
%fitWin = [0 7000]; % можна зафіксувати магнітне вікно 
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
dBpp   = abs(Bpk - Bmn);
dB_init = max(5, dBpp / sqrt(3));   % в Гаусах

% оцінка p за асиметрією R = |max|/|min| (груба емпірика для старту)
Ramp   = max(abs(ypk), 1e-9) / max(abs(ymn), 1e-9);
p_init = min(4.0, max(0.2, 0.9 + 0.8*(Ramp-1)));  % просто стартова точка

% амплітуда після нормування бази
A_init = (ypk - ymn);
off_init = 0; slope_init = 0; quad_init = 0;

% нові параметри калібрування осі поля
Bscale_init = 1.0;   % ~1
Bshift_init = 0.0;   % ~0 Гс

%    [ B0       dBpp     p       A       off       slope       quad       Bscale       Bshift ]
p0 = [ B0_init, dB_init, p_init, A_init, off_init, slope_init, quad_init, Bscale_init, Bshift_init ];

% (Bscale у вузькому коридорі ±1%, Bshift ±10 G):
lb = [fitWin(1)+20,   5,    0.2,   0,   -Inf,   -Inf,   -Inf,   0.995,  -10];
ub = [fitWin(2)-20, 1500,   5.0,  Inf,   Inf,    Inf,    Inf,   1.010,   10];


%% 7. Фіт Dyson (Feher–Kip, plate geometry) + невелике розмиття модуляцією
mwFreqGHz = 9.859842;             % частота за замовчуванням (ГГц), далі намагаємося прочитати з DSC файлу
Bstep = median(diff(Bb));         % крок осі
Bmod  = min(1.0, 0.03*dB_init);   % ефективна мод. амплітуда (Г), обережно мала

% Зчитуємо мікрохвильову частоту з .DSC файлу, якщо він існує
dscPath = fullfile(scriptDir, [erase(filename, '.DTA') '.DSC']);
got_mwFreqGHz = false; % FrequencyMon зчитано з DSC -> флаг = True
if isfile(dscPath)
    txt = fileread(dscPath);
    % Шукаємо рядок виду: "FrequencyMon       9.848925 GHz"
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

model_fun = @(p,Bvec) dyson_with_baseline( ...
    Bvec, p(1), p(2), p(3), p(4), p(5), p(6), p(7), ...  % B0,dBpp,p,A,off,slope,quad
    p(8), p(9), ...                                      % Bscale, Bshift
    Bmod, Bstep);                                        % 

% втрати (L2)
res_fun = @(p) model_fun(p,Bb) - yb;

% Використовуємо Optimization Toolbox (lsqnonlin) для фізично коректного
% нелінійного least-squares фіту з обмеженнями параметрів.
% За відсутності toolbox —> fallback на fminsearch з мінімізацією MSE.
useLSQ = license('test','Optimization_Toolbox') && exist('lsqnonlin','file')==2;
if useLSQ
    opts = optimoptions('lsqnonlin','Display','off','MaxFunctionEvaluations',5e3,'MaxIterations',400);
    [pf,~,~,exitflag] = lsqnonlin(res_fun, p0, lb, ub, opts);
    fprintf('Optimization Toolbox знайдено, використовуємо lsqnonlin');
else
    fprintf('Optimization Toolbox не знайдено, використовуємо базовий fminsearch');
    % fallback -> fminsearch по MSE, піджимаємо у межі вручну
    obj = @(p) mean((res_fun(bound_params(p,lb,ub))).^2);
    opts = optimset('Display','off','MaxFunEvals',5e3,'MaxIter',1e3);
    pf = fminsearch(obj, p0, opts);
    pf = bound_params(pf, lb, ub);
    exitflag = 1;
end

B0 = pf(1); dB = pf(2); p = pf(3); A = pf(4);
off = pf(5); sl = pf(6); qd = pf(7);
Bscale = pf(8); Bshift = pf(9);
yfit = model_fun(pf, Bb);
% fprintf('calib:  Bscale=%.6f,  Bshift=%.2f G\n', Bscale, Bshift);


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
    'Simulated Dyson (MATLAB stand alone fit)', ...
    'Location','best');

lgd.Color = 'w';       
lgd.TextColor = 'k';   
lgd.EdgeColor = 'k';   

outFit = fullfile(scriptDir, [erase(filename,'.DTA') '_fit_Matlab.png']);
saveas(gcf, outFit);
fprintf('Збережено накладання: %s\n', outFit);


%% 10. Блок допоміжних функцій 
function y = dyson_with_baseline(B, B0, dBpp, p, A, off, slope, quad, Bscale, Bshift, Bmod, Bstep)
    % Каліброване поле
    Bcorr = Bshift + Bscale .* B;

    % Чиста похідна Dyson (Feher–Kip, plate) на Bcorr
    x = 2*(Bcorr - B0)/(sqrt(3)*dBpp);
    % Повна формула коефіцієнтів A і D (два доданки, як у Holiatkina et al., 2023)
    denom1 = 2*p.*(cosh(p) + cos(p));
    denom2 = (cosh(p) + cos(p)).^2;
    Acoef = (sinh(p) + sin(p))./denom1 + (1 + cosh(p).*cos(p))./denom2;
    Dcoef = (sinh(p) - sin(p))./denom1 + (sinh(p).*sin(p))./denom2;
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