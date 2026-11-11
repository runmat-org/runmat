function y = cycloidal(u)
% Normalized cycloidal motion law: 0 at u = 0, 1 at u = 1, zero velocity and acceleration at both ends.
y = u - sin(2 * pi * u) / (2 * pi);
end
