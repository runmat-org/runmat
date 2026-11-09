function r = camRadius(theta, base, lift, rise, dwell)
% Pitch radius of a dwell-rise-dwell-return disc cam at cam angle theta (SI: m, rad).
% Cycloidal rise over `rise`, a dwell of `dwell` at full lift, a cycloidal return over
% `rise`, then a dwell at the base circle.
s = 0;
if theta < rise
    s = cycloidal(theta / rise) * lift;
elseif theta < rise + dwell
    s = lift;
elseif theta < 2 * rise + dwell
    s = (1 - cycloidal((theta - rise - dwell) / rise)) * lift;
end
r = base + s;
end
