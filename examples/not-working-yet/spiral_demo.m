% Spiral Pattern - Beautiful Mathematical Curve
% Creates an Archimedean spiral

% Parameters
turns = 3;
points_per_turn = 20;
total_points = turns * points_per_turn;

% Create spiral coordinates
t = [];
x = [];
y = [];

for i = 1:total_points
    angle = (i - 1) * 6.28 * turns / total_points; % 2*pi * turns / total_points
    radius = angle / 6.28; % Grows linearly with angle
    
    t = [t, angle];
    x = [x, radius * cos(angle)];
    y = [y, radius * sin(angle)];
end

% Additional parametric curves
% Heart curve (simplified)
heart_x = [];
heart_y = [];
for i = 1:20
    theta = (i - 1) * 6.28 / 19; % 0 to 2*pi
    heart_x = [heart_x, 16 * sin(theta)^3 / 16];
    heart_y = [heart_y, (13 * cos(theta) - 5 * cos(2*theta) - 2 * cos(3*theta) - cos(4*theta)) / 16];
end

% Simple calculations
spiral_radius = turns;
max_x = spiral_radius;
min_x = -spiral_radius;