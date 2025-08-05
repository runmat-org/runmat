% Comprehensive plotting test showcasing different plot types
fprintf('Testing comprehensive plotting capabilities...\n');

% Line plot
x = [1, 2, 3, 4, 5];
y = [2, 4, 1, 5, 3];
plot(x, y);
fprintf('✓ Line plot created\n');

% Scatter plot
x_scatter = [1, 2, 3, 4, 5, 6];
y_scatter = [1.2, 2.1, 2.9, 4.2, 4.8, 6.1];
scatter(x_scatter, y_scatter);
fprintf('✓ Scatter plot created\n');

% Bar chart
categories = [1, 2, 3, 4];
values = [10, 25, 17, 30];
bar(categories, values);
fprintf('✓ Bar chart created\n');

% Histogram
data = [1, 2, 2, 3, 3, 3, 4, 4, 5];
hist(data, 5);
fprintf('✓ Histogram created\n');

% 3D Surface plot
[X, Y] = meshgrid(-2:0.2:2, -2:0.2:2);
Z = X .* exp(-X.^2 - Y.^2);
surf(X, Y, Z);
fprintf('✓ 3D surface plot created\n');

% 3D Point cloud
x3d = [1, 2, 3, 4, 5];
y3d = [1, 4, 2, 5, 3];
z3d = [2, 1, 3, 1, 4];
values = [10, 20, 15, 25, 18];
scatter3(x3d, y3d, z3d, values);
fprintf('✓ 3D point cloud created\n');

fprintf('All plotting capabilities working! 🎉\n');