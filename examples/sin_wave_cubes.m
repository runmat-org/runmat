% Mario Kart Style Sin Wave Point Cloud Demo
% High-performance cubic point cloud showing off RunMat capabilities

% Create a beautiful sin wave with cubic appearance
x1 = -6.0; y1 = 2.0 * sin(x1);
x2 = -4.5; y2 = 2.0 * sin(x2);
x3 = -3.0; y3 = 2.0 * sin(x3);
x4 = -1.5; y4 = 2.0 * sin(x4);
x5 = 0.0;  y5 = 2.0 * sin(x5);
x6 = 1.5;  y6 = 2.0 * sin(x6);
x7 = 3.0;  y7 = 2.0 * sin(x7);
x8 = 4.5;  y8 = 2.0 * sin(x8);
x9 = 6.0;  y9 = 2.0 * sin(x9);

% Create multiple Z layers for cubic density
z_layer1 = -0.5;
z_layer2 = 0.0;
z_layer3 = 0.5;

% Build dense point cloud with multiple layers (simulating cubes)
x_coords = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x1, x2, x3, x4, x5, x6, x7, x8, x9, x1, x2, x3, x4, x5, x6, x7, x8, x9];
y_coords = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y1, y2, y3, y4, y5, y6, y7, y8, y9, y1, y2, y3, y4, y5, y6, y7, y8, y9];
z_coords = [z_layer1, z_layer1, z_layer1, z_layer1, z_layer1, z_layer1, z_layer1, z_layer1, z_layer1, z_layer2, z_layer2, z_layer2, z_layer2, z_layer2, z_layer2, z_layer2, z_layer2, z_layer2, z_layer3, z_layer3, z_layer3, z_layer3, z_layer3, z_layer3, z_layer3, z_layer3, z_layer3];

% Render the high-performance Mario Kart style cubic sin wave
scatter3(x_coords, y_coords, z_coords)