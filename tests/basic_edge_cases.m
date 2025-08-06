% Basic Edge Cases for Currently Supported Features

% 1. Basic Matrix Operations
A = [1, 2, 3; 4, 5, 6];
B = A .* 2;
C = A + B;
element = A(2, 1);

% 2. Array Generation
x = linspace(0, pi, 10);
y = zeros(3, 3);
z = ones(2, 4);

% 3. Mathematical Functions
trig_result = sin(x);
cos_result = cos(x);
tan_result = tan(x);

% 4. Constants
const_pi = pi;
const_e = e;
const_sum = pi + e + sqrt2;

% 5. Matrix Concatenation
horizontal = [x, trig_result];
vertical = [x; trig_result];

% 6. Element-wise Operations
elem_mult = A .* B;
elem_div = A ./ B;
elem_pow = A .^ 2;

% 7. Statistical Functions
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
mean_val = mean(data);
max_val = max(data);
min_val = min(data);
sum_val = sum(data);

% 8. Indexing Edge Cases
first_elem = data(1);
last_elem = data(10);
middle_range = data(3:7);

% 9. Power Functions
power_result = pow(2, 8);
matrix_power = A .^ 3;

% 10. Complex Matrix Operations
large_A = rand(5, 5);
large_B = rand(5, 5);
matrix_mult = large_A * large_B;
elem_mult_large = large_A .* large_B;