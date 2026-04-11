%% ========================================================================
%  Batch Dyson Fit - обробка всіх DTA файлів у директорії
%
%  Обробляє всі .DTA файли та зберігає результати у таблицю
% ========================================================================

clear; clc;

% Знаходимо директорію скрипту
scriptDir = fileparts(mfilename('fullpath'));
cd(scriptDir);

% Знаходимо всі DTA файли
dtaFiles = dir(fullfile(scriptDir, '*.DTA'));
nFiles = length(dtaFiles);

fprintf('=== Batch Dyson Fit ===\n');
fprintf('Знайдено %d DTA файлів\n\n', nFiles);

% Масив для результатів
results = cell(nFiles, 9);

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
        Bb = B(mask);  yb = spc(mask);

        % Базова лінія
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
        Bpk = Bb(ipk);  Bmn = Bb(imn);

        i1 = min(ipk,imn); i2 = max(ipk,imn);
        zeroIdx = i1 + find(y0(i1:i2).*circshift(y0(i1:i2),-1) < 0, 1,'first') - 1;
        if ~isempty(zeroIdx)
            B0_init = interp1(y0(zeroIdx:zeroIdx+1), Bb(zeroIdx:zeroIdx+1), 0,'linear','extrap');
        else
            B0_init = Bb(round((ipk+imn)/2));
        end

        dBpp = abs(Bpk - Bmn);
        dB_init = max(5, dBpp / sqrt(3));
        Ramp = max(abs(ypk), 1e-9) / max(abs(ymn), 1e-9);
        p_init = min(4.0, max(0.2, 0.9 + 0.8*(Ramp-1)));
        A_init = (ypk - ymn);

        p0 = [B0_init, dB_init, p_init, A_init, 0, 0, 0, 1.0, 0.0];
        lb = [fitWin(1)+20, 5, 0.2, 0, -Inf, -Inf, -Inf, 0.995, -10];
        ub = [fitWin(2)-20, 1500, 5.0, Inf, Inf, Inf, Inf, 1.010, 10];

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
        Bmod = min(1.0, 0.03*dB_init);

        model_fun = @(p,Bvec) dyson_with_baseline( ...
            Bvec, p(1), p(2), p(3), p(4), p(5), p(6), p(7), ...
            p(8), p(9), Bmod, Bstep);

        res_fun = @(p) model_fun(p,Bb) - yb;

        % Фітування
        useLSQ = license('test','Optimization_Toolbox') && exist('lsqnonlin','file')==2;
        if useLSQ
            opts = optimoptions('lsqnonlin','Display','off','MaxFunctionEvaluations',5e3,'MaxIterations',400);
            [pf,~,~,exitflag] = lsqnonlin(res_fun, p0, lb, ub, opts);
        else
            obj = @(p) mean((res_fun(bound_params(p,lb,ub))).^2);
            opts = optimset('Display','off','MaxFunEvals',5e3,'MaxIter',1e3);
            pf = fminsearch(obj, p0, opts);
            pf = bound_params(pf, lb, ub);
            exitflag = 1;
        end

        B0 = pf(1); dB = pf(2); p = pf(3); A = pf(4);
        yfit = model_fun(pf, Bb);

        % g-фактор
        muB = 9.2740100783e-24;
        h = 6.62607015e-34;
        g_est = (h*(mwFreqGHz*1e9)) / (muB*(B0*1e-4));

        % Зберігаємо результати
        results{i,1} = filename;
        results{i,2} = B0;
        results{i,3} = dB;
        results{i,4} = p;
        results{i,5} = A;
        results{i,6} = g_est;
        results{i,7} = mwFreqGHz;
        results{i,8} = numel(B);
        results{i,9} = 'OK';

        fprintf('  B0=%.2f G, dB=%.2f G, p=%.3f, g=%.5f\n', B0, dB, p, g_est);

        % Графік
        fig = figure('Visible','off','Color','w','Position',[140 180 1000 540]);
        ax = axes(fig);
        ax.Color = 'w'; ax.XColor = 'k'; ax.YColor = 'k';
        ax.GridColor = [0.8 0.8 0.8];
        grid(ax,'on'); hold(ax,'on');

        plot(Bb, yb, 'g-', 'LineWidth', 1.0); hold on;
        plot(Bb, yfit, 'r-', 'LineWidth', 1.2);
        grid on; xlabel('Magnetic field B (G)'); ylabel('dI/dB (a.u.)');

        titleStr = sprintf('%s | B0=%.2f G, dB=%.2f G, p=%.3f | g=%.5f, ν=%.5f GHz', ...
            erase(filename,'.DTA'), B0, dB, p, g_est, mwFreqGHz);

        t = title(ax, titleStr, 'FontWeight','bold', 'Interpreter','none');
        t.Color = 'k';

        lgd = legend(ax, 'Experimental', 'Simulated Dyson (MATLAB)', 'Location','best');
        lgd.Color = 'w'; lgd.TextColor = 'k'; lgd.EdgeColor = 'k';

        outFit = fullfile(scriptDir, [erase(filename,'.DTA') '_fit_Matlab.png']);
        saveas(gcf, outFit);
        close(fig);

    catch ME
        fprintf('  ПОМИЛКА: %s\n', ME.message);
        results{i,1} = filename;
        results{i,2:8} = deal(NaN);
        results{i,9} = ME.message;
    end
    fprintf('\n');
end

% Виводимо зведену таблицю
fprintf('\n=== ЗВЕДЕНІ РЕЗУЛЬТАТИ ===\n');
fprintf('%-15s %10s %10s %8s %12s %10s %10s\n', 'Файл', 'B0 (G)', 'dB (G)', 'p', 'A (a.u.)', 'g-factor', 'ν (GHz)');
fprintf('%s\n', repmat('-', 1, 90));
for i = 1:nFiles
    if strcmp(results{i,9}, 'OK')
        fprintf('%-15s %10.2f %10.2f %8.3f %12.3g %10.5f %10.6f\n', ...
            erase(results{i,1},'.DTA'), results{i,2}, results{i,3}, results{i,4}, ...
            results{i,5}, results{i,6}, results{i,7});
    else
        fprintf('%-15s %s\n', results{i,1}, 'FAILED');
    end
end

fprintf('\nГотово! Всі графіки збережено як *_fit_Matlab.png\n');

%% Допоміжні функції
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
        g = exp(-0.5*(xi./sigma).^2); g = g/sum(g);
        yclean = conv(yclean, g, 'same');
    end

    t = (B - mean(B));
    y = A*yclean + off + slope*t + quad*(t.^2 - mean(t.^2));
end

function p = bound_params(p, lb, ub)
    p = max(p, lb); p = min(p, ub);
end
