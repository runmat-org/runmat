% MATLAB Data Type Edge Cases and Conversions
% This file tests all MATLAB data types and their edge cases

%% 1. Numeric Types and Precision
% Double precision (default)
double_max = realmax;                 % Largest double
double_min = realmin;                 % Smallest positive double
double_eps = eps;                     % Machine epsilon
double_inf = Inf;                     % Positive infinity
double_ninf = -Inf;                   % Negative infinity
double_nan = NaN;                     % Not a Number

% Integer types
int8_max = int8(127);
int8_min = int8(-128);
uint8_max = uint8(255);
int16_max = int16(32767);
uint16_max = uint16(65535);
int32_max = int32(2147483647);
uint32_max = uint32(4294967295);
int64_max = int64(9223372036854775807);
uint64_max = uint64(18446744073709551615);

% Single precision
single_val = single(3.14159);
single_eps = eps(single_val);

%% 2. Type Conversion Edge Cases
% Overflow and underflow
overflow_int8 = int8(200);            % Should saturate to 127
underflow_int8 = int8(-200);          % Should saturate to -128
overflow_uint8 = uint8(-50);          % Should saturate to 0

% Precision loss
precise_double = 1234567890123456789;
converted_single = single(precise_double);
back_to_double = double(converted_single);

% NaN and Inf conversions
nan_to_int = int32(NaN);              % Should be 0
inf_to_int = int32(Inf);              % Should be max int32

%% 3. Logical Arrays
% Basic logical operations
log_true = true;
log_false = false;
log_array = [true, false, true];

% Logical indexing
numbers = [1, 2, 3, 4, 5];
mask = numbers > 3;                   % [false, false, false, true, true]
filtered = numbers(mask);             % [4, 5]

% Logical arithmetic
log_add = true + false;               % Should be 1
log_mult = true * 2;                  % Should be 2
log_and = true & false;               % Should be false
log_or = true | false;                % Should be true
log_not = ~true;                      % Should be false

%% 4. Character Arrays and Strings
% Character arrays (older MATLAB)
char_array = 'Hello World';
char_length = length(char_array);
char_indexing = char_array(1:5);      % 'Hello'
char_concat = [char_array, ' Again'];

% Multi-row character arrays
char_matrix = char('Hello', 'World');  % Pads to same length
char_matrix_size = size(char_matrix);

% String arrays (newer MATLAB, R2016b+)
string_scalar = "Hello World";
string_array = ["Hello", "World", "Test"];
string_length = strlength(string_scalar);
string_concat = string_scalar + " Again";

% Mixed string operations
char_to_string = string(char_array);
string_to_char = char(string_scalar);

%% 5. Cell Arrays
% Basic cell arrays
empty_cell = {};
mixed_cell = {1, 'hello', [1,2,3], true};
cell_size = size(mixed_cell);

% Nested cell arrays
nested_cell = {{1, 2}, {'a', 'b'}, {[1,2], [3,4]}};
deep_access = nested_cell{1}{2};      % Should be 2

% Cell array expansion
cell_numbers = {1, 2, 3, 4};
expanded = [cell_numbers{:}];         % [1, 2, 3, 4]

% Cell array indexing
cell_content = mixed_cell{1};         % Content (1)
cell_reference = mixed_cell(1);       % Reference ({1})

%% 6. Structure Arrays
% Basic structures
simple_struct.name = 'John';
simple_struct.age = 30;
simple_struct.height = 5.9;

% Structure arrays
people(1).name = 'Alice';
people(1).age = 25;
people(2).name = 'Bob';
people(2).age = 30;

% Dynamic field names
field_name = 'age';
dynamic_access = people(1).(field_name);

% Structure manipulation
field_names = fieldnames(people);
has_field = isfield(people, 'age');
struct_size = size(people);

% Nested structures
nested.person.name = 'Charlie';
nested.person.details.age = 35;
deep_field = nested.person.details.age;

%% 7. Function Handles
% Anonymous functions
add_func = @(x, y) x + y;
square_func = @(x) x.^2;
result_anon = add_func(3, 4);         % Should be 7

% Function handles to built-in functions
sin_handle = @sin;
cos_handle = @cos;
result_builtin = sin_handle(pi/2);    % Should be 1

% Function handles with multiple outputs
% [min_val, min_idx] = @(x) [min(x), find(x == min(x))];

%% 8. Complex Numbers
% Basic complex numbers
complex1 = 3 + 4i;
complex2 = 2 + 3j;                    % Alternative imaginary unit
complex3 = complex(3, 4);             % Explicit construction

% Complex operations
complex_add = complex1 + complex2;
complex_mult = complex1 * complex2;
complex_conj = conj(complex1);
complex_abs = abs(complex1);
complex_angle = angle(complex1);
complex_real = real(complex1);
complex_imag = imag(complex1);

% Complex arrays
complex_array = [1+2i, 3+4i, 5+6i];
complex_matrix = [1+i, 2+2i; 3+3i, 4+4i];

%% 9. Sparse Arrays
% Sparse matrix creation
sparse_matrix = sparse([1,2,3], [1,2,3], [10,20,30], 5, 5);
sparse_size = size(sparse_matrix);
sparse_nnz = nnz(sparse_matrix);      % Number of non-zeros

% Sparse matrix operations
sparse_full = full(sparse_matrix);    % Convert to full matrix
sparse_find = find(sparse_matrix);    % Find non-zero indices

% Sparse logical
sparse_logical = sparse(true(3,3));

%% 10. Tables (if supported)
% Table creation (R2013b+)
% age = [25; 30; 35];
% height = [5.6; 5.9; 6.1];
% weight = [120; 150; 180];
% names = {'Alice'; 'Bob'; 'Charlie'};
% 
% people_table = table(age, height, weight, 'RowNames', names);
% alice_age = people_table{'Alice', 'age'};

%% 11. Categorical Arrays (if supported)
% Categorical data (R2013b+)
% colors = categorical({'red', 'blue', 'green', 'red', 'blue'});
% color_categories = categories(colors);
% color_counts = countcats(colors);

%% 12. DateTime and Duration (if supported)
% DateTime arrays (R2014b+)
% current_time = datetime('now');
% date_array = datetime([2023, 2024], [1, 6], [15, 20]);
% 
% % Duration
% time_duration = duration(1, 30, 45);  % 1 hour, 30 min, 45 sec

%% 13. Type Checking and Conversion Functions
% Type identification
var_class = class(mixed_cell);
is_numeric = isnumeric(123);
is_char = ischar('hello');
is_logical = islogical(true);
is_cell = iscell(mixed_cell);
is_struct = isstruct(simple_struct);
is_function = isa(sin_handle, 'function_handle');

% Size and dimension queries
var_size = size(char_matrix);
var_length = length(string_array);
var_ndims = ndims(complex_matrix);
var_numel = numel(people);

%% 14. Type Coercion in Operations
% Numeric hierarchy: double > single > int64 > ... > logical
mixed_double_single = double(5) + single(3);  % Result is double
mixed_int_logical = int32(5) + true;          % Result depends on MATLAB version

% String and char interactions
string_char_concat = string_scalar + char_array;  % If supported

%% 15. Memory and Performance Considerations
% Pre-allocation vs dynamic growth
large_array = zeros(1000, 1000);     % Pre-allocated (efficient)
% growing_array = [];                 % Dynamic growth (inefficient)
% for i = 1:1000
%     growing_array(i) = i;           % Avoid this pattern
% end

% Memory usage
memory_info = whos();                 % Variable memory usage

%% 16. Edge Cases and Special Values
% Empty arrays of different types
empty_double = double.empty(0, 5);    % 0x5 double array
empty_char = char.empty(3, 0);        % 3x0 char array
empty_cell = cell.empty(2, 0, 4);     % 2x0x4 cell array

% Boundary value testing
tiny_positive = realmin * eps;
huge_negative = -realmax;

% Special matrix types
identity = eye(3);                    % Identity matrix
ones_matrix = ones(2, 3);             % All ones
zeros_matrix = zeros(4, 2);           % All zeros
random_matrix = rand(3, 3);           % Random values [0,1]
normal_random = randn(3, 3);          % Normal distribution

%% 17. Type Conversion Edge Cases
% Rounding behavior in conversions
float_val = 3.7;
floor_convert = int32(floor(float_val));  % 3
round_convert = int32(round(float_val));  % 4
ceil_convert = int32(ceil(float_val));    % 4

% Complex to real conversion
complex_to_real = real(3 + 4i);       % 3 (imaginary part lost)

% Logical to numeric
logical_to_num = double([true, false]); % [1, 0]

fprintf('Data type edge case tests completed.\n');