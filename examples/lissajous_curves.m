% Lissajous Curves - Beautiful Parametric Patterns
% Creates various Lissajous figures with different frequency ratios

% Time parameter
t = [];
for i = 1:100
    t = [t, (i - 1) * 6.28 / 99]; % 0 to 2*pi
end

% Lissajous 2:3 ratio
x_2_3 = [];
y_2_3 = [];
for i = 1:100
    x_2_3 = [x_2_3, sin(2 * t(i))];
    y_2_3 = [y_2_3, sin(3 * t(i))];
end

% Lissajous 3:4 ratio
x_3_4 = [];
y_3_4 = [];
for i = 1:100
    x_3_4 = [x_3_4, sin(3 * t(i))];
    y_3_4 = [y_3_4, sin(4 * t(i))];
end

% Lissajous 1:2 ratio (figure-8)
x_1_2 = [];
y_1_2 = [];
for i = 1:100
    x_1_2 = [x_1_2, sin(t(i))];
    y_1_2 = [y_1_2, sin(2 * t(i))];
end

% Rose curve (simplified version using cartesian approximation)
rose_x = [];
rose_y = [];
for i = 1:100
    angle = t(i);
    % r = cos(3*θ) approximated in cartesian
    r = cos(3 * angle);
    rose_x = [rose_x, r * cos(angle)];
    rose_y = [rose_y, r * sin(angle)];
end

% Cycloid curve (simplified)
cycloid_x = [];
cycloid_y = [];
for i = 1:100
    theta = t(i) * 2; % 0 to 4*pi for full cycloid
    cycloid_x = [cycloid_x, theta - sin(theta)];
    cycloid_y = [cycloid_y, 1 - cos(theta)];
end

% Calculations
frequency_ratio_a = 2;
frequency_ratio_b = 3;
phase_shift = 0;