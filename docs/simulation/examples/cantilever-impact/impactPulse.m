function force = impactPulse(time, peak, duration)
arguments
    time (:,1) double
    peak (1,1) double
    duration (1,1) double
end

insidePulse = time <= duration;
phase = min(time ./ duration, 1);
force = peak .* sin(pi .* phase) .* insidePulse;
end
