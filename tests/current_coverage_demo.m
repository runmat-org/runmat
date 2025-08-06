% RustMat MATLAB Language Coverage Demonstration
% This file showcases all currently implemented MATLAB features

% === BASIC DATA TYPES ===
% Scalars
scalar_int = 42;
scalar_float = 3.14159;
scalar_negative = -2.5;

% === MATRICES AND ARRAYS ===
% Matrix literals
matrix_2x3 = [1, 2, 3; 4, 5, 6];
row_vector = [1, 2, 3, 4, 5];
col_vector = [1; 2; 3; 4];

% Array generation functions
zeros_matrix = zeros(3, 4);
ones_matrix = ones(2, 5);
linspace_array = linspace(0, 10, 11);
range_simple = 1:10;
range_step = 1:2:20;
range_reverse = 10:-1:1;

% === MATHEMATICAL CONSTANTS ===
const_pi = pi;
const_e = e;
const_sqrt2 = sqrt2;
const_combination = pi + e + sqrt2;

% === ARRAY INDEXING ===
% Single element access
element = matrix_2x3(2, 1);
linear_index = row_vector(3);

% Range indexing
range_subset = row_vector(2:4);
step_subset = linspace_array(1:2:11);

% === MATHEMATICAL FUNCTIONS ===
% Trigonometric functions
sin_values = sin(linspace_array);
cos_values = cos(range_simple);
tan_values = tan([0, pi/4, pi/2]);

% Power functions
power_scalar = pow(2, 8);
power_elementwise = linspace_array .^ 2;

% Statistical functions
data_vector = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
mean_value = mean(data_vector);
max_value = max(data_vector);
min_value = min(data_vector);
sum_value = sum(data_vector);

% === ELEMENT-WISE OPERATIONS ===
A = [1, 2; 3, 4];
B = [2, 1; 1, 2];

% Element-wise arithmetic
elem_mult = A .* B;
elem_div = A ./ B;
elem_pow = A .^ 2;

% Mixed operations
mixed_ops = A .* 2 + ones(2, 2);

% === MATRIX CONCATENATION ===
% Horizontal concatenation
horizontal = [row_vector, row_vector];

% Vertical concatenation
vertical = [row_vector; row_vector];

% Mixed concatenation
mixed_concat = [1, row_vector, 2];

% === COMPLEX EXPRESSIONS ===
% Nested function calls
nested_result = sin(cos(pi/4));

% Complex mathematical expressions
complex_expr = sin(linspace_array) .^ 2 + cos(linspace_array) .^ 2;

% Matrix operations with functions
matrix_sin = sin(matrix_2x3);
matrix_stats = [min(matrix_2x3), max(matrix_2x3)];

% === PERFORMANCE FEATURES ===
% Large array operations (testing JIT performance)
large_array = linspace(1, 1000, 1000);
large_sin = sin(large_array);
large_stats = [mean(large_sin), sum(large_sin)];

final_result = [scalar_int, mean_value, max(large_stats)];