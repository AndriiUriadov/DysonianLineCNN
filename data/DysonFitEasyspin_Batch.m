%% ========================================================================
%  Batch Dyson Fit - EasySpin Simplex
%  Обробка всіх DTA файлів у директорії з esfit
%
%  Обробляє всі .DTA файли та зберігає результати у таблицю
% ========================================================================

clear; clc;

% EasySpin path safety
if exist('esfit','file')~=2 || exist('eprload','file')~=2
    addpath('/Users/a.uriadov/Documents/easyspin-6.0.11/easyspin');
    rehash toolboxcache;
end

% Знаходимо директорію скрипту
scriptDir = fileparts(mfilename('fullpath'));
cd(scriptDir);

% Знаходимо всі DTA файли
dtaFiles = dir(fullfile(scriptDir, '*.DTA'));
nFiles = length(dtaFiles);

fprintf('=== Batch Dyson Fit (EasySpin Simplex) ===\n');
fprintf('Знайдено %d DTA файлів\n\n', nFiles);

% Масив для результатів
results = cell(nFiles, 12);

% Обробляємо кожен файл
for i = 1:nFiles
    filename = dtaFiles(i).name;
    fprintf('[%d/%d] Обробка: %s\n', i, nFiles, filename);

    try
        % Завантажуємо спектр
        fpath = fullfile(scriptDir, filename);
        [B, spc] = eprload(fpath);
        B = double(B(:));
        spc = double(spc(:));

        % Магнітне вікно
        fitWin = [min(B), max(B)];
        mask = (B >= fitWin(1)) & (B <= fitWin(2));
        Bb = B(mask);
        yb = spc(mask);

        % Базова лінія для стартових параметрів
        t = (Bb - mean(Bb));
        M = [ones(size(t)), t, t.^2];
        c = M \ yb;
        y0 = yb - M*c;
        signfix = sign(max(y0)) * sign(abs(min(y0)) - abs(max(y0)));
        if signfix > 0
           y0 = -y0;
           yb = -yb;
        end

        % Початкові параметри
        [ypk, ipk] = max(y0);
        [ymn, imn] = min(y0);
        Bpk = Bb(ipk);
        Bmn = Bb(imn);

        i1 = min(ipk,imn);
        i2 = max(ipk,imn);
        zeroIdx = i1 + find(y0(i1:i2).*circshift(y0(i1:i2),-1) < 0, 1,'first') - 1;
        if ~isempty(zeroIdx)
            B0_init = interp1(y0(zeroIdx:zeroIdx+1), Bb(zeroIdx:zeroIdx+1), 0,'linear','extrap');
        else
            B0_init = Bb(round((ipk+imn)/2));
        end

        dBpp_init = abs(Bpk - Bmn);
        Ramp = max(abs(ypk), 1e-9) / max(abs(ymn), 1e-9);
        p_init = min(4.0, max(0.2, 0.9 + 0.8*(Ramp-1)));
        A_init = (ypk - ymn);

        % Частота
        mwFreqGHz = 9.859842;
        dscPath = fullfile(scriptDir, [erase(filename, '.DTA') '.DSC']);
        if isfile(dscPath)
            txt = fileread(dscPath);
            freqPattern = regexp(txt, 'FrequencyMon\s+([\d\.Ee\+\-]+)\s*GHz', 'tokens', 'once');
            if ~isempty(freqPattern)
                mwFreqGHz = str2double(freqPattern{1});
            end
        end

        Bstep = median(diff(Bb));
        Bmod = min(1.0, 0.03*dBpp_init/sqrt(3));

        % FitOpt для simplex
        FitOpt = struct();
        FitOpt.Method    = 'simplex fcn';
        FitOpt.AutoScale = 'none';
        FitOpt.Verbosity = 0;
        FitOpt.BaseLine  = [];
        FitOpt.TolFun    = 1e-5;
        FitOpt.TolEdgeLength = 1e-4;
        FitOpt.delta     = 0.15;

        % Маска
        maskFit = (Bb >= (B0_init-6*dBpp_init)) & (Bb <= (B0_init+6*dBpp_init));
        maskFit = logical(maskFit(:));
        FitOpt.Mask = maskFit;

        % Функція моделі
        esfit_fcn = @(v) dyson_with_baseline( ...
            Bb, v(1), v(2), v(3), v(4), ...
            v(5), 0, 0, ...
            1.0, 0.0, Bmod, Bstep);

        v0 = [B0_init, dBpp_init, p_init, A_init, 0];

        offMax = 0.01 * max(abs(yb));
        lb_fit = [fitWin(1)+20, 5,    0.2, 0,   -offMax];
        ub_fit = [fitWin(2)-20, 1500, 5.0, Inf,  offMax];

        % Фітування
        fitResult = esfit(yb, esfit_fcn, v0, lb_fit, ub_fit, FitOpt);
        vf = fitResult.pfit(:).';

        B0_fit   = vf(1);
        dBpp_fit = vf(2);
        p_fit    = vf(3);
        A_fit    = vf(4);
        off_fit  = vf(5);

        yfit = esfit_fcn(vf);

        % g-фактор
        muB = 9.2740100783e-24;
        h = 6.62607015e-34;
        g_est = (h*(mwFreqGHz*1e9)) / (muB*(B0_fit*1e-4));

        % Uncertainty для p
        p_std = NaN;
        p_ci95 = [NaN NaN];
        if isfield(fitResult, 'pstd') && ~isempty(fitResult.pstd)
            p_std = fitResult.pstd(3);  % стандартне відхилення для p
        end
        if isfield(fitResult, 'ci95') && ~isempty(fitResult.ci95)
            p_ci95 = fitResult.ci95(3,:);  % 95% довірчий інтервал для p
        end

        % Зберігаємо результати
        results{i,1} = filename;
        results{i,2} = B0_fit;
        results{i,3} = dBpp_fit;
        results{i,4} = p_fit;
        results{i,5} = A_fit;
        results{i,6} = g_est;
        results{i,7} = mwFreqGHz;
        results{i,8} = numel(B);
        results{i,9} = fitResult.rmsd;
        results{i,10} = p_std;
        results{i,11} = p_ci95(1);
        results{i,12} = p_ci95(2);

        fprintf('  B0=%.2f G, dB=%.2f G, p=%.3f±%.3f, g=%.5f, RMSD=%.3g\n', ...
            B0_fit, dBpp_fit, p_fit, p_std, g_est, fitResult.rmsd);

        % Графік
        fig = figure('Visible','off','Color','w','Position',[140 180 1000 540]);
        ax = axes(fig);
        ax.Color = 'w';
        ax.XColor = 'k';
        ax.YColor = 'k';
        ax.GridColor = [0.8 0.8 0.8];
        grid(ax,'on');
        hold(ax,'on');

        plot(Bb, yb, 'g-', 'LineWidth', 1.0);
        plot(Bb, yfit, 'r-', 'LineWidth', 1.2);
        grid on;
        xlabel('Magnetic field B (G)');
        ylabel('dI/dB (a.u.)');

        titleStr = sprintf('%s | B0=%.2f G, dB=%.2f G, p=%.3f±%.3f | g=%.5f, ν=%.5f GHz', ...
            erase(filename,'.DTA'), B0_fit, dBpp_fit, p_fit, p_std, g_est, mwFreqGHz);

        title(ax, titleStr, 'FontWeight','bold', 'Interpreter','none', 'Color','k');

        lgd = legend(ax, 'Experimental', 'Simulated Dyson (EasySpin Simplex)', 'Location','best');
        lgd.Color = 'w';
        lgd.TextColor = 'k';
        lgd.EdgeColor = 'k';

        outFit = fullfile(scriptDir, [erase(filename,'.DTA') '_fit_Easyspin.png']);
        saveas(gcf, outFit);
        close(fig);

    catch ME
        fprintf('  ПОМИЛКА: %s\n', ME.message);
        results{i,1} = filename;
        results{i,2:12} = deal(NaN);
    end
    fprintf('\n');
end

% Виводимо зведену таблицю
fprintf('\n=== ЗВЕДЕНІ РЕЗУЛЬТАТИ (EasySpin Simplex) ===\n');
fprintf('%-15s %10s %10s %10s %12s %10s %10s\n', ...
    'Файл', 'B0 (G)', 'dB (G)', 'p±std', 'A (a.u.)', 'g-factor', 'ν (GHz)');
fprintf('%s\n', repmat('-', 1, 95));
for i = 1:nFiles
    if ~isnan(results{i,2})
        fprintf('%-15s %10.2f %10.2f %10s %12.3g %10.5f %10.6f\n', ...
            erase(results{i,1},'.DTA'), results{i,2}, results{i,3}, ...
            sprintf('%.3f±%.3f', results{i,4}, results{i,10}), ...
            results{i,5}, results{i,6}, results{i,7});
    else
        fprintf('%-15s %s\n', results{i,1}, 'FAILED');
    end
end

fprintf('\nГотово! Всі графіки збережено як *_fit_Easyspin.png\n');

%% Допоміжна функція
function y = dyson_with_baseline(B, B0, dBpp, p, A, off, slope, quad, Bscale, Bshift, Bmod, Bstep)
    Bcorr = Bshift + Bscale .* B;
    x = 2*(Bcorr - B0)/(sqrt(3)*dBpp);

    % Повна формула коефіцієнтів A і D (Holiatkina et al., 2023)
    denom1 = 2*p.*(cosh(p) + cos(p));
    denom2 = (cosh(p) + cos(p)).^2;
    Acoef = (sinh(p) + sin(p))./denom1 + (1 + cosh(p).*cos(p))./denom2;
    Dcoef = (sinh(p) - sin(p))./denom1 + (sinh(p).*sin(p))./denom2;

    yclean = (-Acoef.*2.*x) ./ (1 + x.^2).^2 + Dcoef.*(1 - x.^2) ./ (1 + x.^2).^2;

    if Bmod > 0
        sigma = (Bmod/2.355) / max(Bstep,eps);
        k = max(1, ceil(6*sigma));
        xi = (-k:k);
        g = exp(-0.5*(xi./sigma).^2);
        g = g/sum(g);
        yclean = conv(yclean, g, 'same');
    end

    t = (B - mean(B));
    y = A*yclean + off + slope*t + quad*(t.^2 - mean(t.^2));
end
