%% ========================================================================
%  test_dyson_formulas.m
%  Unit tests for plate (Holiatkina 2023) and sphere (Savchenko 2022)
%  Dysonian A/D mixing coefficient formulas.
%
%  Authors:  A.V. Uriadov, D.V. Savchenko
%  National Technical University of Ukraine
%  "Igor Sikorsky Kyiv Polytechnic Institute"
%
% =========================================================================
%
%  Run from the repository root:
%      /Applications/MATLAB_R2026a.app/bin/matlab -batch \
%          "addpath('matlab'); run('matlab/test_dyson_formulas.m')"
%
%  The script asserts physical and numerical properties expected from both
%  formulas and exits non-zero on any failure. All tests are silent on
%  pass; a summary line is printed at the end.
% =========================================================================

thisFile = mfilename('fullpath');
thisDir  = fileparts(thisFile);
addpath(thisDir);  % ensures dysonAD_sphere is resolvable

nTests = 0; nPass = 0;

%% ---------- Asymptotic limits ----------

% PLATE geometry: at p -> 0 both A and D are bounded (A -> 1, D -> 0).
p = 1e-6;
[A_pl, D_pl] = dysonAD_plate_local(p);
check('plate: D -> 0 at p -> 0', abs(D_pl) < 1e-3);
check('plate: A -> 1 at p -> 0', abs(A_pl - 1.0) < 1e-3);

% SPHERE geometry: at p -> 0 the denominator (cosh p - cos p) -> p^2 -> 0,
% so A ~ 1/p^4 and D ~ 1/p^2 both diverge. Only the RATIO D/A -> 0, which
% corresponds to a symmetric Lorentzian derivative (the physical limit of
% a grain much smaller than the skin depth). Check D/A, not D alone.
[A_sp, D_sp] = dysonAD_sphere(p);
check('sphere: D/A -> 0 at p -> 0 (symmetric Lorentzian limit)', ...
      abs(D_sp / A_sp) < 1e-3);

% A must be finite and positive for both geometries at moderate p
for p = [1e-3, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    [A_pl, ~] = dysonAD_plate_local(p);
    [A_sp, ~] = dysonAD_sphere(p);
    check(sprintf('plate: A>0 finite at p=%.3g', p),  isfinite(A_pl) && A_pl > 0);
    check(sprintf('sphere: A>0 finite at p=%.3g', p), isfinite(A_sp) && A_sp > 0);
end

%% ---------- Plate and sphere differ (they must) ----------

% If they matched, refactoring would be a no-op. Confirm they don't.
% Note: at p=1 both A_plate and A_sphere are ~1 so they happen to agree on A.
% Use p=3 where they differ substantially.
p = 3.0;
[A_pl, D_pl] = dysonAD_plate_local(p);
[A_sp, D_sp] = dysonAD_sphere(p);
rel_A = abs(A_pl - A_sp) / max(abs(A_pl), 1e-12);
rel_D = abs(D_pl - D_sp) / max(abs(D_pl), 1e-12);
check('plate/sphere A disagree at p=3', rel_A > 0.1);
check('plate/sphere D disagree at p=3', rel_D > 0.1);

%% ---------- Plate formula: sanity hand-computation at p=1 ----------

% Plate (Holiatkina 2023, Eq. 3):
%   A = (sh+s)/(2p(ch+c)) + (1 + ch*c)/(ch+c)^2
%   D = (sh-s)/(2p(ch+c)) + (sh*s)/(ch+c)^2
p = 1.0;
sh = sinh(p); s = sin(p); ch = cosh(p); c = cos(p);
den = ch + c;
A_expected = (sh+s)/(2*p*den) + (1 + ch*c)/den^2;
D_expected = (sh-s)/(2*p*den) + (sh*s)/den^2;
[A_pl, D_pl] = dysonAD_plate_local(p);
check('plate: hand-coded A match at p=1', abs(A_pl - A_expected) < 1e-12);
check('plate: hand-coded D match at p=1', abs(D_pl - D_expected) < 1e-12);

%% ---------- Sphere formula: sanity hand-computation at p=1 ----------

% Sphere (Savchenko 2022, Eq. 3):
p = 1.0;
sh = sinh(p); s = sin(p); ch = cosh(p); c = cos(p);
den = ch - c;
A49_expected = ...
       8/p^4 ...
    - 8*(sh + s)/(p^3*den) ...
    + 8*sh*s/(p^2*den^2) ...
    + (sh - s)/(p*den) ...
    - (sh^2 - s^2)/den^2 ...
    + 1;
D49_expected = ...
       8*(sh - s)/(p^3*den) ...
    - 4*(sh^2 - s^2)/(p^2*den^2) ...
    + (sh + s)/(p*den) ...
    - 2*sh*s/den^2;
A_expected = (9/4) * A49_expected;
D_expected = (9/4) * D49_expected;
[A_sp, D_sp] = dysonAD_sphere(p);
check('sphere: hand-coded A match at p=1', abs(A_sp - A_expected) < 1e-10);
check('sphere: hand-coded D match at p=1', abs(D_sp - D_expected) < 1e-10);

%% ---------- Vectorization: accept array input ----------

p_vec = [0.1, 1.0, 2.0, 5.0];
[A_vec, D_vec] = dysonAD_sphere(p_vec);
check('sphere: vectorized A has correct size',  isequal(size(A_vec), size(p_vec)));
check('sphere: vectorized D has correct size',  isequal(size(D_vec), size(p_vec)));
% Each element must match the scalar call
for i = 1:numel(p_vec)
    [Ai, Di] = dysonAD_sphere(p_vec(i));
    check(sprintf('sphere: vec[%d] == scalar', i), ...
          abs(A_vec(i)-Ai) < 1e-12 && abs(D_vec(i)-Di) < 1e-12);
end

%% ---------- Derivative sign convention (A/B ratio) ----------

% Savchenko paper states: A/B = 1 + 1.45 * D / A (in their notation),
% where A, B are peak heights of the first-derivative EPR line. At p -> 0
% the ratio -> 1 (pure Lorentzian, symmetric). Verify this qualitative
% behavior: D/A -> 0 as p -> 0.
p_small = 0.01;
[As, Ds] = dysonAD_sphere(p_small);
ratio = abs(Ds / As);
check('sphere: D/A -> 0 at p -> 0 (ratio < 0.01)', ratio < 0.01);

% At large p the ratio D/A should be finite and positive (metallic regime).
p_large = 5.0;
[Al, Dl] = dysonAD_sphere(p_large);
check('sphere: D>0 at p=5 (metallic regime)', Dl > 0 && isfinite(Dl));
check('sphere: A>0 at p=5 (metallic regime)', Al > 0 && isfinite(Al));

%% ---------- Summary ----------

fprintf('\n=== %d/%d tests passed ===\n', nPass, nTests);
if nPass < nTests
    error('test_dyson_formulas:FAILED', '%d test(s) failed', nTests - nPass);
end

%% ======================================================================
%  Helper: plate (Holiatkina 2023) reference implementation, used for
%  cross-verification against the copies embedded in DysonGeneratorMix.m,
%  Validator.m, FitMatlab.m, FitEasyspin.m.
% =======================================================================

function [Acoef, Dcoef] = dysonAD_plate_local(p)
    p = max(p, 1e-9);
    ch = cosh(p); sh = sinh(p); c = cos(p); s = sin(p);
    den = ch + c;
    Acoef = (sh + s)./(2*p.*den) + (1 + ch.*c)./(den.^2);
    Dcoef = (sh - s)./(2*p.*den) + (sh.*s)./(den.^2);
end

function check(name, cond)
    % Tiny assertion helper. nTests/nPass are captured from the caller's
    % workspace via assignin (script-level counters).
    nT = evalin('caller', 'nTests') + 1;
    assignin('caller', 'nTests', nT);
    if cond
        assignin('caller', 'nPass', evalin('caller', 'nPass') + 1);
    else
        fprintf(2, 'FAIL: %s\n', name);
    end
end
