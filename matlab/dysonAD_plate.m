function [Acoef, Dcoef] = dysonAD_plate(p)
%% ========================================================================
%  dysonAD_plate — Dysonian A and D coefficients for FLAT PLATE geometry
%                  (Feher-Kip model, Holiatkina et al. 2023, Eq. 3)
%
%  Authors:  A.V. Uriadov, D.V. Savchenko
%  National Technical University of Ukraine
%  "Igor Sikorsky Kyiv Polytechnic Institute"
%
% =========================================================================
%
%  PURPOSE
%    Return the A (dispersion) and D (absorption) mixing coefficients of
%    the Dysonian first-derivative EPR lineshape for a flat metallic plate
%    of thickness 2d, with p = 2d / delta (delta = skin depth).
%
%  MODEL
%    Holiatkina et al., J. Appl. Phys. 134, 145702 (2023), Eq. (3):
%
%        A = (sinh p + sin p) / (2p (cosh p + cos p))
%          + (1 + cosh p * cos p) / (cosh p + cos p)^2
%
%        D = (sinh p - sin p) / (2p (cosh p + cos p))
%          + sinh p * sin p / (cosh p + cos p)^2
%
%  USAGE
%    [A, D] = dysonAD_plate(p)
%
%  INPUT
%    p   — scalar or array, skin-effect parameter p = 2d/delta (p > 0)
%
%  OUTPUT
%    Acoef, Dcoef — same shape as p
%
%  NOTE
%    This is different from the sphere geometry (Savchenko et al. 2022).
%    See matlab/dysonAD_sphere.m. For geometry-agnostic dispatch use
%    matlab/dysonAD.m.
%
% =========================================================================

    p  = max(p, 1e-9);                 % avoid divide-by-zero
    sh = sinh(p); s  = sin(p);
    ch = cosh(p); c  = cos(p);
    den = ch + c;                      % PLUS, unlike sphere geometry
    Acoef = (sh + s) ./ (2 * p .* den) + (1 + ch .* c) ./ den.^2;
    Dcoef = (sh - s) ./ (2 * p .* den) + (sh .* s)     ./ den.^2;
end
