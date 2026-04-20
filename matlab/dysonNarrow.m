function sig = dysonNarrow(B, B0, dB, p, geometry)
%% ========================================================================
%  dysonNarrow — Narrow Dysonian EPR first-derivative lineshape
%                (geometry-aware: plate or sphere)
%
%  Authors:  A.V. Uriadov, D.V. Savchenko
%  National Technical University of Ukraine
%  "Igor Sikorsky Kyiv Polytechnic Institute"
%
% =========================================================================
%
%  PURPOSE
%    Return the unnormalized Dysonian first-derivative signal dI/dB on the
%    magnetic-field grid B. Used by the dataset generator (DysonGeneratorMix),
%    the fit validator (Validator), and both classical fitters
%    (FitMatlab, FitEasyspin).
%
%  MODEL
%    Both geometries share the same functional form
%
%        sig = -A * 2x / (1 + x^2)^2 + D * (1 - x^2) / (1 + x^2)^2
%
%    but differ in two places:
%
%      - Plate geometry (Holiatkina 2023):
%            x = 2(B - B0) / (sqrt(3) * dB)
%            [A, D] = dysonAD_plate(p)
%
%      - Sphere geometry (Savchenko 2022):
%            x = (B - B0) / dB
%            [A, D] = dysonAD_sphere(p)
%
%    The overall sign convention (minus on the A term) matches the
%    existing codebase. A later polarityFix() step inverts the signal if
%    the first extremum is a minimum.
%
%  USAGE
%    sig = dysonNarrow(B, B0, dB, p)               % default 'plate'
%    sig = dysonNarrow(B, B0, dB, p, 'plate')
%    sig = dysonNarrow(B, B0, dB, p, 'sphere')
%
%  INPUT
%    B        — magnetic field axis (vector, Gauss)
%    B0       — resonance field (scalar, Gauss)
%    dB       — linewidth parameter (scalar, Gauss)
%    p        — skin-effect parameter (scalar, p > 0)
%    geometry — 'plate' (default) or 'sphere'
%
%  OUTPUT
%    sig      — same shape as B, unnormalized first-derivative signal
%
% =========================================================================

    if nargin < 5 || isempty(geometry)
        geometry = 'plate';
    end

    dB = max(dB, eps);

    switch lower(char(geometry))
        case 'plate'
            % Holiatkina 2023: x normalized so x=1 at B = B0 +- sqrt(3)/2 * dB
            x = 2 .* (B - B0) ./ (sqrt(3) .* dB);
        case 'sphere'
            % Savchenko 2022: x = (B - B_res) / dB (no sqrt(3) factor)
            x = (B - B0) ./ dB;
        otherwise
            error('dysonNarrow:BadGeometry', ...
                  'Unknown geometry ''%s'': expected ''plate'' or ''sphere''.', ...
                  char(geometry));
    end

    [Acoef, Dcoef] = dysonAD(p, geometry);

    den = (1 + x.^2).^2;
    sig = (-Acoef .* 2 .* x) ./ den + Dcoef .* (1 - x.^2) ./ den;
end
