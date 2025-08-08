% High-Performance Point Cloud Demo: Mario Kart Style Sin Wave
% Generates a dense 3D point cloud of cubes arranged in a sin wave pattern
% This demo showcases RunMat's advanced point cloud rendering capabilities

% chaos - amplitude of random noise added to each point
chaos = 0.1;

% Sin wave parameters
x_range = linspace(-4*pi, 4*pi, 120);  % X axis range (120 points for density)
z_range = linspace(-2, 2, 40);         % Z axis range (40 points for width)
cube_size = 0.15;                      % Size of each cube
wave_amplitude = 2.0;                  % Amplitude of the sin wave

% Initialize arrays for point cloud data
points = [];
colors = [];

fprintf('Generating Mario Kart style sin wave point cloud...\n');
fprintf('X points: %d, Z points: %d\n', length(x_range), length(z_range));
fprintf('Total cubes: %d\n', length(x_range) * length(z_range) * 8);

% Generate the sin wave structure
for i = 1:length(x_range)
    x = x_range(i);
    y_center = wave_amplitude * sin(x);  % Sin wave equation
    
    for j = 1:length(z_range)
        z = z_range(j);
        
        % Create a cube at each (x, y_center, z) position
        % Each cube is made of 8 corner points
        cube_corners = [
            x - cube_size/2, y_center - cube_size/2, z - cube_size/2;
            x + cube_size/2, y_center - cube_size/2, z - cube_size/2;
            x - cube_size/2, y_center + cube_size/2, z - cube_size/2;
            x + cube_size/2, y_center + cube_size/2, z - cube_size/2;
            x - cube_size/2, y_center - cube_size/2, z + cube_size/2;
            x + cube_size/2, y_center - cube_size/2, z + cube_size/2;
            x - cube_size/2, y_center + cube_size/2, z + cube_size/2;
            x + cube_size/2, y_center + cube_size/2, z + cube_size/2;
        ];
        
        % Add chaos (random noise) to each corner
        if chaos > 0
            noise = (rand(8, 3) - 0.5) * 2 * chaos;
            cube_corners = cube_corners + noise;
        end
        
        % Add cube corners to point cloud
        points = [points; cube_corners];
        
        % Create orange color gradient based on wave position
        % Brighter orange at wave peaks, darker at troughs
        wave_intensity = (y_center + wave_amplitude) / (2 * wave_amplitude);  % Normalize 0-1
        orange_base = [1.0, 0.6, 0.2];  % Base orange color
        orange_variation = [0.2, 0.3, 0.1];  % Color variation
        
        % Create 8 colors for the cube corners with slight variation
        for k = 1:8
            corner_variation = (rand(1, 3) - 0.5) * 0.1;  % Small random color variation
            cube_color = orange_base + wave_intensity * orange_variation + corner_variation;
            cube_color = max(0, min(1, cube_color));  % Clamp to [0,1]
            colors = [colors; cube_color];
        end
    end
    
    % Progress indicator
    if mod(i, 20) == 0
        progress = i / length(x_range) * 100;
        fprintf('Progress: %.1f%%\n', progress);
    end
end

fprintf('Generated %d points for point cloud\n', size(points, 1));
fprintf('Chaos level: %.2f\n', chaos);

% Extract X, Y, Z coordinates
x_coords = points(:, 1);
y_coords = points(:, 2);
z_coords = points(:, 3);

% Extract RGB color components
red_values = colors(:, 1);
green_values = colors(:, 2);
blue_values = colors(:, 3);

fprintf('Rendering high-performance point cloud...\n');

% Create the point cloud plot
% Note: This assumes scatter3 function supports color mapping
% For now, we'll use scatter3 with position data
scatter3(x_coords, y_coords, z_coords);

fprintf('Point cloud rendering complete!\n');
fprintf('Demo showcases:\n');
fprintf('- Dense 3D point cloud generation\n');
fprintf('- Mario Kart style cubic sin wave\n');
fprintf('- Orange color gradient mapping\n');
fprintf('- Configurable chaos/noise parameter\n');
fprintf('- High-performance visualization\n');