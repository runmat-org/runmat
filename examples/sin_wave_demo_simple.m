% Simplified Mario Kart style sin wave point cloud demo
chaos = 0.1;
x_range = linspace(-4*pi, 4*pi, 20);
wave_amplitude = 2.0;
points = [];

% Simple loop to test
for i = 1:length(x_range)
    x = x_range(i);
    y = wave_amplitude * sin(x);
    z = 0;
    points = [points; [x, y, z]];
end

% Extract coordinates
x_coords = points(:, 1);
y_coords = points(:, 2);
z_coords = points(:, 3);

% Plot
scatter3(x_coords, y_coords, z_coords)