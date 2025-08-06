% MATLAB Advanced Language Features and Edge Cases
% This file tests advanced syntax and lesser-known MATLAB features

%% 1. Command Line Syntax vs Function Syntax
% Command syntax (space-separated arguments, treated as strings)
% clear x y z              % Equivalent to clear('x', 'y', 'z')
% help sin                 % Equivalent to help('sin')
% load data.mat            % Equivalent to load('data.mat')

% Function syntax (explicit parentheses and arguments)
clear('x', 'y', 'z');
% help('sin');

%% 2. Multiple Assignment and Comma-Separated Lists
% Multiple assignment from functions
[max_val, max_idx] = max([3, 1, 4, 1, 5]);

% Comma-separated list expansion
cell_data = {1, 2, 3, 4, 5};
array_from_cell = [cell_data{:}];           % Expands to [1, 2, 3, 4, 5]

% Structure array expansion
data.field = [10, 20, 30];
values = [data.field];                      % Direct field access

%% 3. Advanced Indexing Patterns
% Linear indexing in multi-dimensional arrays
A = reshape(1:24, [2, 3, 4]);              % 2x3x4 array
linear_access = A(13);                      % 13th element in linear order

% Sub2ind and ind2sub conversions
[row, col] = ind2sub(size(A(:,:,1)), 5);    % Convert linear to subscript
linear_idx = sub2ind(size(A(:,:,1)), 2, 3); % Convert subscript to linear

% Boolean indexing combinations
B = magic(5);
complex_mask = (B > 10) & (B < 20);        % Complex logical condition
masked_values = B(complex_mask);

%% 4. Colon Operator Advanced Usage
% Colon in different contexts
all_rows = B(:, 3);                        % All rows, column 3
all_cols = B(2, :);                        % Row 2, all columns
all_elements = B(:);                       % Vectorize entire matrix

% Step patterns
every_other = 1:2:10;                      % [1, 3, 5, 7, 9]
reverse_step = 10:-2:1;                    % [10, 8, 6, 4, 2]
fractional_step = 0:0.1:1;                 % [0, 0.1, 0.2, ..., 1.0]

%% 5. End Keyword Advanced Usage
% End in different dimensions
C = rand(5, 6, 3);
last_page = C(:, :, end);                  % Last page of 3D array
last_element = C(end, end, end);           % Very last element
second_to_last = C(end-1, end-2, 1);      % Arithmetic with end

% End in expressions
middle_to_end = C(3:end, 2:end-1, :);      % Complex range with end

%% 6. Implicit Expansion (Broadcasting)
% Matrix-scalar operations
matrix = [1, 2; 3, 4];
scalar_add = matrix + 10;                  % Adds 10 to each element

% Vector-matrix operations (if supported)
% row_vector = [1, 2];
% col_vector = [10; 20];
% broadcast_result = matrix + row_vector;    % Row-wise addition

%% 7. Cell Array Advanced Operations
% Nested cell arrays
nested = {{1, 2}, {3, {4, 5}}, {6}};
deep_access = nested{2}{2}{1};             % Should be 4

% Cell array assignment patterns
cell_array = cell(2, 3);
cell_array{1, 1} = [1, 2, 3];
cell_array{1, 2} = 'hello';
cell_array{2, 1} = magic(3);

% Comma-separated list in function calls
% func_result = max(cell_data{:});         % Equivalent to max(1,2,3,4,5)

%% 8. Structure Advanced Operations
% Dynamic field creation
s = struct();
field_name = 'dynamic_field';
s.(field_name) = 42;                       % Dynamic field access

% Nested structure creation
nested_struct.level1.level2.value = 'deep';
deep_value = nested_struct.level1.level2.value;

% Structure array operations
struct_array(1).name = 'first';
struct_array(2).name = 'second';
names_cell = {struct_array.name};          % Extract all names

%% 9. Function Handle Advanced Features
% Function handles with closures
function_factory = @(n) @(x) x^n;         % Returns a function
square_func = function_factory(2);         % Creates x^2 function
cube_func = function_factory(3);          % Creates x^3 function

% Anonymous function with multiple statements (limited)
% complex_anon = @(x) deal(x^2, x^3);     % Multiple outputs

%% 10. Variable Argument Lists (varargin/varargout)
% Simulating variable arguments
% function result = var_arg_demo(varargin)
%     switch nargin
%         case 1
%             result = varargin{1}^2;
%         case 2
%             result = varargin{1} + varargin{2};
%         otherwise
%             result = sum([varargin{:}]);
%     end
% end

%% 11. Global and Persistent Variables
% Global variable usage
global GLOBAL_CONFIG;
GLOBAL_CONFIG = struct('version', 1.0, 'debug', true);

% Persistent variable simulation
% function count = counter()
%     persistent counter_value;
%     if isempty(counter_value)
%         counter_value = 0;
%     end
%     counter_value = counter_value + 1;
%     count = counter_value;
% end

%% 12. Error Handling Advanced Patterns
% Custom error creation
% error('MyToolbox:InvalidInput', 'The input %d is not valid', 42);

% Error structure
error_struct = struct('message', 'Custom error', ...
                     'identifier', 'Test:CustomError', ...
                     'stack', []);

% Try-catch with error re-throwing
try
    % Some operation that might fail
    risky_value = 1 / 0;
catch ME
    % Log error and re-throw
    fprintf('Logged error: %s\n', ME.message);
    % rethrow(ME);
end

%% 13. Vectorization Patterns
% Vectorized operations vs loops
x = 1:1000;

% Element-wise operations
vectorized1 = sin(x).^2 + cos(x).^2;      % Should all be 1

% Conditional vectorization
y = x;
y(x > 500) = x(x > 500) * 2;              % Conditional modification

% Logical indexing for replacement
z = x;
z(mod(z, 2) == 0) = -z(mod(z, 2) == 0);   % Negate even numbers

%% 14. Memory Pre-allocation Patterns
% Efficient pre-allocation
n = 1000;
preallocated = zeros(n, 1);               % Pre-allocate memory
for i = 1:n
    preallocated(i) = i^2;
end

% Cell array pre-allocation
cell_prealloc = cell(n, 1);
for i = 1:10  % Just first 10 for testing
    cell_prealloc{i} = rand(3, 3);
end

%% 15. Advanced Matrix Operations
% Matrix reshaping and permutation
original_matrix = rand(2, 3, 4);
reshaped = reshape(original_matrix, [6, 4]);
permuted = permute(original_matrix, [3, 1, 2]); % Swap dimensions

% Matrix replication
replicated = repmat([1, 2], 3, 2);        % Replicate pattern

%% 16. Special Matrix Types and Properties
% Special matrices
hilbert_like = 1 ./ (1:5 + (1:5)' - 1);   % Hilbert-like matrix
toeplitz_like = ones(5) + diag(1:4, 1) + diag(1:4, -1);

% Matrix properties
is_symmetric = isequal(hilbert_like, hilbert_like');
matrix_rank = rank(hilbert_like);

%% 17. Complex Number Advanced Operations
% Complex arithmetic edge cases
complex_array = [1+2i, 3-4i, 5+0i];
complex_ops = [real(complex_array); imag(complex_array)];

% Phase and magnitude
magnitudes = abs(complex_array);
phases = angle(complex_array);

%% 18. Sparse Matrix Advanced Operations
% Sparse matrix creation and manipulation
[i, j, v] = find(magic(5) > 15);          % Find large elements
sparse_from_triplet = sparse(i, j, v, 5, 5);

% Sparse operations
sparse_ops = sparse_from_triplet * 2;
sparse_logical = sparse_from_triplet > 0;

%% 19. Type System Edge Cases
% Type conversion chains
original = int8(100);
converted = double(single(original));      % int8 -> single -> double

% Type preservation in operations
int_result = int8(50) + int8(30);         % Result type
mixed_result = int8(50) + 30;             % Mixed type operation

%% 20. Performance and Memory Profiling Concepts
% Memory usage awareness
large_matrix = rand(1000, 1000);
% memory_info = whos('large_matrix');     % Check memory usage

% Timing operations
tic;
computation_result = large_matrix * large_matrix';
elapsed_time = toc;

%% 21. Import/Export and File I/O Edge Cases
% Data export concepts (would need actual files)
% save('test_data.mat', 'large_matrix', '-v7.3');  % HDF5 format
% csvwrite('test_data.csv', large_matrix(1:10, 1:10));

%% 22. Graphics and Visualization Edge Cases
% Basic plotting (commented out to avoid display issues)
% figure;
% plot(1:10, rand(1, 10), 'o-');
% title('Sample Plot');
% xlabel('X Values');
% ylabel('Y Values');

%% 23. Parallel Computing Concepts
% Parallel processing concepts (if Parallel Computing Toolbox available)
% parfor i = 1:10
%     parallel_result(i) = expensive_computation(i);
% end

%% 24. Object-Oriented Programming Advanced Features
% Class concepts (would be in separate files)
% Property validation attributes
% Method attributes (Static, Abstract, etc.)
% Event handling
% Package system

%% 25. GPU Computing Concepts (if available)
% GPU array concepts
% gpu_array = gpuArray(rand(1000, 1000));
% gpu_result = gpu_array * gpu_array';
% cpu_result = gather(gpu_result);

fprintf('Advanced MATLAB features test completed.\n');
fprintf('Maximum value: %.2f at index %d\n', max_val, max_idx);
fprintf('Array from cell expansion: length = %d\n', length(array_from_cell));
fprintf('Complex deep access result: %d\n', deep_access);
fprintf('Vectorized computation time: %.6f seconds\n', elapsed_time);