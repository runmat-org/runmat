% RunMat High-Performance Point Cloud Showcase
% Mario Kart Style Orange Sin Wave with Cubic Appearance
% 
% This demo showcases RunMat's advanced 3D visualization capabilities:
% - High-performance scatter3 point cloud rendering
% - Mario Kart inspired cubic orange aesthetic  
% - GPU-accelerated PointCloudPlot with debug output
% - Multi-layered 3D sin wave structure

% Generate a dense, layered sin wave that looks like Mario Kart blocks

% Layer 1: Bottom layer of the sin wave (Z = -0.4)
x1 = -6.0; y1 = 1.8 * sin(x1); z1 = -0.4;
x2 = -5.0; y2 = 1.8 * sin(x2); z2 = -0.4;
x3 = -4.0; y3 = 1.8 * sin(x3); z3 = -0.4;
x4 = -3.0; y4 = 1.8 * sin(x4); z4 = -0.4;
x5 = -2.0; y5 = 1.8 * sin(x5); z5 = -0.4;
x6 = -1.0; y6 = 1.8 * sin(x6); z6 = -0.4;
x7 = 0.0;  y7 = 1.8 * sin(x7); z7 = -0.4;
x8 = 1.0;  y8 = 1.8 * sin(x8); z8 = -0.4;
x9 = 2.0;  y9 = 1.8 * sin(x9); z9 = -0.4;
x10 = 3.0; y10 = 1.8 * sin(x10); z10 = -0.4;
x11 = 4.0; y11 = 1.8 * sin(x11); z11 = -0.4;
x12 = 5.0; y12 = 1.8 * sin(x12); z12 = -0.4;
x13 = 6.0; y13 = 1.8 * sin(x13); z13 = -0.4;

% Layer 2: Middle layer (Z = 0.0) 
x14 = -6.0; y14 = 1.8 * sin(x14); z14 = 0.0;
x15 = -5.0; y15 = 1.8 * sin(x15); z15 = 0.0;
x16 = -4.0; y16 = 1.8 * sin(x16); z16 = 0.0;
x17 = -3.0; y17 = 1.8 * sin(x17); z17 = 0.0;
x18 = -2.0; y18 = 1.8 * sin(x18); z18 = 0.0;
x19 = -1.0; y19 = 1.8 * sin(x19); z19 = 0.0;
x20 = 0.0;  y20 = 1.8 * sin(x20); z20 = 0.0;
x21 = 1.0;  y21 = 1.8 * sin(x21); z21 = 0.0;
x22 = 2.0;  y22 = 1.8 * sin(x22); z22 = 0.0;
x23 = 3.0;  y23 = 1.8 * sin(x23); z23 = 0.0;
x24 = 4.0;  y24 = 1.8 * sin(x24); z24 = 0.0;
x25 = 5.0;  y25 = 1.8 * sin(x25); z25 = 0.0;
x26 = 6.0;  y26 = 1.8 * sin(x26); z26 = 0.0;

% Layer 3: Top layer (Z = 0.4)
x27 = -6.0; y27 = 1.8 * sin(x27); z27 = 0.4;
x28 = -5.0; y28 = 1.8 * sin(x28); z28 = 0.4;
x29 = -4.0; y29 = 1.8 * sin(x29); z29 = 0.4;
x30 = -3.0; y30 = 1.8 * sin(x30); z30 = 0.4;
x31 = -2.0; y31 = 1.8 * sin(x31); z31 = 0.4;
x32 = -1.0; y32 = 1.8 * sin(x32); z32 = 0.4;
x33 = 0.0;  y33 = 1.8 * sin(x33); z33 = 0.4;
x34 = 1.0;  y34 = 1.8 * sin(x34); z34 = 0.4;
x35 = 2.0;  y35 = 1.8 * sin(x35); z35 = 0.4;
x36 = 3.0;  y36 = 1.8 * sin(x36); z36 = 0.4;
x37 = 4.0;  y37 = 1.8 * sin(x37); z37 = 0.4;
x38 = 5.0;  y38 = 1.8 * sin(x38); z38 = 0.4;
x39 = 6.0;  y39 = 1.8 * sin(x39); z39 = 0.4;

% Combine all layers into dense point cloud arrays
x_all = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39];
y_all = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39];
z_all = [z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15, z16, z17, z18, z19, z20, z21, z22, z23, z24, z25, z26, z27, z28, z29, z30, z31, z32, z33, z34, z35, z36, z37, z38, z39];

% Render the stunning Mario Kart style sin wave point cloud!
% This will showcase RunMat's high-performance 3D capabilities
scatter3(x_all, y_all, z_all)