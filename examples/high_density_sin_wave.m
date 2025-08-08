% High-Density Mario Kart Style Sin Wave - RunMat Performance Demo
% This showcases RunMat's high-performance 3D point cloud capabilities

% Generate dense sin wave coordinates using linspace
x_points = linspace(-2*pi, 2*pi, 50);  % 50 points along X

% Create the point cloud arrays
x_coords = [];
y_coords = [];
z_coords = [];

% Generate sin wave with multiple Z layers for cubic density
wave_amplitude = 1.5;
z_layers = [-0.3, -0.1, 0.1, 0.3];  % 4 Z layers

% Build the 3D cubic sin wave
for z_layer = z_layers
    for x = x_points
        y = wave_amplitude * sin(x);
        x_coords = [x_coords, x];
        y_coords = [y_coords, y];
        z_coords = [z_coords, z_layer];
    end
end

% This creates 200 points total (50 x-points * 4 z-layers)
% Render the high-performance cubic sin wave
scatter3(x_coords, y_coords, z_coords)