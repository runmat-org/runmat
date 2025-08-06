% Test Currently Supported RustMat Features
% This tests features we know are implemented

fprintf('=== RustMat Current Feature Test ===\n');

% Basic matrix operations
A = [1, 2, 3; 4, 5, 6];
B = A .* 2;
element = A(2, 1);

% Mathematical functions
x = linspace(0, pi, 5);
y = sin(x);
cosine_vals = cos(x);

% Constants
const_test = pi + e + sqrt2;

% Array generation
zeros_mat = zeros(3, 3);
ones_mat = ones(2, 4);
range_vals = range(1, 10, 2);

% Statistical functions
data = [1, 2, 3, 4, 5];
mean_val = mean(data);
max_val = max(data);
min_val = min(data);
sum_val = sum(data);

% Power operations
power_test = pow(2, 3);
elem_power = x .^ 2;

% Matrix concatenation
horizontal = [x, y];
vertical = [x; y];

% Complex operations
result = A .* B + ones(2, 3);

fprintf('All basic operations completed successfully!\n');