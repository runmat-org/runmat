% Mario Kart Style Sin Wave Point Cloud Demo
% Generates a high-performance 3D point cloud visualization

% Chaos parameter - amplitude of random noise added to each point
chaos = 0.05;

% Generate sin wave data points manually (avoiding advanced indexing)
x1 = -4; y1 = 2.0 * sin(x1); z1 = 0;
x2 = -3; y2 = 2.0 * sin(x2); z2 = 0;
x3 = -2; y3 = 2.0 * sin(x3); z3 = 0;
x4 = -1; y4 = 2.0 * sin(x4); z4 = 0;
x5 = 0;  y5 = 2.0 * sin(x5); z5 = 0;
x6 = 1;  y6 = 2.0 * sin(x6); z6 = 0;
x7 = 2;  y7 = 2.0 * sin(x7); z7 = 0;
x8 = 3;  y8 = 2.0 * sin(x8); z8 = 0;
x9 = 4;  y9 = 2.0 * sin(x9); z9 = 0;

% Add some noise (chaos)
x1 = x1 + chaos * (rand() - 0.5);
x2 = x2 + chaos * (rand() - 0.5);
x3 = x3 + chaos * (rand() - 0.5);
x4 = x4 + chaos * (rand() - 0.5);
x5 = x5 + chaos * (rand() - 0.5);
x6 = x6 + chaos * (rand() - 0.5);
x7 = x7 + chaos * (rand() - 0.5);
x8 = x8 + chaos * (rand() - 0.5);
x9 = x9 + chaos * (rand() - 0.5);

y1 = y1 + chaos * (rand() - 0.5);
y2 = y2 + chaos * (rand() - 0.5);
y3 = y3 + chaos * (rand() - 0.5);
y4 = y4 + chaos * (rand() - 0.5);
y5 = y5 + chaos * (rand() - 0.5);
y6 = y6 + chaos * (rand() - 0.5);
y7 = y7 + chaos * (rand() - 0.5);
y8 = y8 + chaos * (rand() - 0.5);
y9 = y9 + chaos * (rand() - 0.5);

% Create arrays for the sin wave
x_coords = [x1, x2, x3, x4, x5, x6, x7, x8, x9];
y_coords = [y1, y2, y3, y4, y5, y6, y7, y8, y9];
z_coords = [z1, z2, z3, z4, z5, z6, z7, z8, z9];

% Render the high-performance Mario Kart style point cloud
scatter3(x_coords, y_coords, z_coords)