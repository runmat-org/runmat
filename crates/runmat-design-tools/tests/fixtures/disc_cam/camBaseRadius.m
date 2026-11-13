function rb = camBaseRadius(lift, rise, max_pressure)
% Smallest base-circle radius that keeps a cycloidal rise under the pressure-angle limit.
% A cycloidal rise of `lift` over `rise` radians peaks at ds/dtheta = 2*lift/rise, at
% mid-rise (s = lift/2). Inputs and result are SI (m, rad).
rb = (2 * lift / rise) / tan(max_pressure) - lift / 2;
end
