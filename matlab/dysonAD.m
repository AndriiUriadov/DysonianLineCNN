function [Acoef, Dcoef] = dysonAD(p, geometry)
%% ========================================================================
%  dysonAD — Geometry-aware Dysonian A/D coefficient dispatcher
%
%  Authors:  A.V. Uriadov, D.V. Savchenko
%  National Technical University of Ukraine
%  "Igor Sikorsky Kyiv Polytechnic Institute"
%
% =========================================================================
%
%  PURPOSE
%    Single entry point to Dysonian A/D mixing coefficients for either
%    flat-plate (Feher-Kip, Holiatkina 2023) or spherical-particle
%    (Savchenko 2022) geometry.
%
%  USAGE
%    [A, D] = dysonAD(p)                  % default 'plate'
%    [A, D] = dysonAD(p, 'plate')         % Holiatkina 2023
%    [A, D] = dysonAD(p, 'sphere')        % Savchenko 2022
%
%  INPUT
%    p        — scalar or array, skin-effect parameter (p > 0)
%    geometry — 'plate' (default) or 'sphere'
%
%  OUTPUT
%    Acoef, Dcoef — same shape as p
%
% =========================================================================

    if nargin < 2 || isempty(geometry)
        geometry = 'plate';
    end

    switch lower(char(geometry))
        case 'plate'
            [Acoef, Dcoef] = dysonAD_plate(p);
        case 'sphere'
            [Acoef, Dcoef] = dysonAD_sphere(p);
        otherwise
            error('dysonAD:BadGeometry', ...
                  'Unknown geometry ''%s'': expected ''plate'' or ''sphere''.', ...
                  char(geometry));
    end
end
