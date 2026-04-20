function [Acoef, Dcoef] = dysonAD_sphere(p)
%% ========================================================================
%  dysonAD_sphere — Dysonian A and D coefficients for SPHERICAL particles
%                   (powder grain geometry, Savchenko et al. 2022, Eq. 3)
%
%  Authors:  A.V. Uriadov, D.V. Savchenko
%  National Technical University of Ukraine
%  "Igor Sikorsky Kyiv Polytechnic Institute"
%
% =========================================================================
%
%  PURPOSE
%    Return the A (dispersion) and D (absorption) mixing coefficients of
%    the Dysonian first-derivative EPR lineshape for a spherical powder
%    particle of diameter d, with p = d / delta (delta = skin depth).
%
%    Used in carbon-dot / nanocomposite samples where the EPR line is
%    formed in submicron spherical grains and the microwave field enters
%    from all sides, yielding a different boundary-value problem than the
%    flat-plate (Feher-Kip) geometry.
%
%  MODEL
%    Savchenko et al., J. Phys. Chem. Solids 162 (2022) 110536, Eq. (3):
%
%        dchi/dB = A * (-2x) / (1 + x^2)^2 + D * (1 - x^2) / (1 + x^2)^2
%
%    where x = (B - B_res) / dB and
%
%        4A/9 = 8/p^4
%             - 8(sinh p + sin p) / [p^3 (cosh p - cos p)]
%             + 8 sinh p * sin p / [p^2 (cosh p - cos p)^2]
%             + (sinh p - sin p) / [p (cosh p - cos p)]
%             - (sinh^2 p - sin^2 p) / (cosh p - cos p)^2
%             + 1
%
%        4D/9 = 8(sinh p - sin p) / [p^3 (cosh p - cos p)]
%             - 4(sinh^2 p - sin^2 p) / [p^2 (cosh p - cos p)^2]
%             + (sinh p + sin p) / [p (cosh p - cos p)]
%             - 2 sinh p * sin p / (cosh p - cos p)^2
%
%  USAGE
%    [A, D] = dysonAD_sphere(p)
%
%  INPUT
%    p   — scalar or array, skin-effect parameter p = d/delta (p > 0)
%
%  OUTPUT
%    Acoef, Dcoef — same shape as p
%
%  NOTE
%    This is DIFFERENT from the plate/Feher-Kip geometry formula used in
%    Holiatkina et al. (2023). The sphere formula uses (cosh p - cos p)
%    in denominators, the plate formula uses (cosh p + cos p). The sphere
%    formula has extra 1/p^3 and 1/p^4 terms that dominate at small p.
%
% =========================================================================

    p  = max(p, 1e-9);          % avoid divide-by-zero at p=0
    sh = sinh(p); s  = sin(p);
    ch = cosh(p); c  = cos(p);
    den = ch - c;               % NOTE: MINUS, unlike plate geometry
    den2 = den.^2;

    % 4A/9 — six additive terms
    A49 =   8 ./ (p.^4) ...
          - 8 .* (sh + s) ./ ((p.^3) .* den) ...
          + 8 .* sh .* s ./ ((p.^2) .* den2) ...
          + (sh - s) ./ (p .* den) ...
          - (sh.^2 - s.^2) ./ den2 ...
          + 1;

    % 4D/9 — four additive terms
    D49 =   8 .* (sh - s) ./ ((p.^3) .* den) ...
          - 4 .* (sh.^2 - s.^2) ./ ((p.^2) .* den2) ...
          + (sh + s) ./ (p .* den) ...
          - 2 .* sh .* s ./ den2;

    Acoef = (9/4) .* A49;
    Dcoef = (9/4) .* D49;
end
