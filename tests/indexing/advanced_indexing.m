% Advanced MATLAB Indexing Edge Cases
% This file tests all complex indexing patterns in MATLAB

%% 1. Basic Linear Indexing
A = [1, 2, 3; 4, 5, 6; 7, 8, 9];
linear1 = A(1);      % First element (row-major)
linear2 = A(5);      % Fifth element 
linear3 = A(end);    % Last element using 'end'

%% 2. Advanced 2D Indexing
row_access = A(2, :);       % Entire row
col_access = A(:, 2);       % Entire column
submatrix = A(1:2, 2:3);    % Submatrix
diag_access = A([1,5,9]);   % Diagonal elements via linear indexing

%% 3. Logical Indexing
B = [1, -2, 3, -4, 5];
positive_vals = B(B > 0);           % Elements > 0
negative_idx = B < 0;               % Logical array
negative_vals = B(negative_idx);    % Using logical array
B(B < 0) = 0;                      % Assignment with logical indexing

%% 4. Advanced Array Indexing
C = magic(4);
idx_array = [1, 3, 4];
selected_rows = C(idx_array, :);    % Multiple rows
selected_elements = C([1,2], [2,3]); % Specific elements

%% 5. Colon Operator Edge Cases
D = 1:10;
range1 = D(2:5);         % Standard range
range2 = D(1:2:end);     % Step size 2
range3 = D(end:-1:1);    % Reverse order
empty_range = D(5:3);    % Empty range (5 > 3)

%% 6. Cell Array Indexing
cell_arr = {[1,2], 'hello', magic(2)};
cell_content = cell_arr{1};         % Content indexing with {}
cell_ref = cell_arr(1);            % Reference indexing with ()
cell_multi = cell_arr{[1,3]};      % ERROR: Should fail

%% 7. Structure Array Indexing
students.name = {'Alice', 'Bob', 'Charlie'};
students.age = [20, 21, 19];
students.grade = [85, 92, 78];

alice_name = students.name{1};     % Field access + cell indexing
all_ages = students.age;           % Field access to numeric array
alice_age = students.age(1);       % Field + array indexing

%% 8. Multi-dimensional Arrays
E = rand(3, 4, 2);                 % 3D array
slice_2d = E(:, :, 1);            % 2D slice
single_val = E(2, 3, 1);          % Single element from 3D
linear_3d = E(10);                % Linear indexing in 3D

%% 9. Sparse Matrix Indexing
sparse_mat = sparse([1,2,3], [1,2,3], [10,20,30], 5, 5);
sparse_val = sparse_mat(1, 1);    % Access sparse element
sparse_row = sparse_mat(1, :);    % Sparse row access

%% 10. String Indexing (Character Arrays)
str = 'Hello World';
char1 = str(1);                   % First character
substr = str(1:5);                % Substring
char_end = str(end);              % Last character

%% 11. Edge Cases and Error Conditions
try
    out_of_bounds = A(10, 10);    % Should error: index out of bounds
catch
    disp('Caught out of bounds error');
end

try
    wrong_dims = A(1, 2, 3);      % Should error: too many dimensions
catch
    disp('Caught dimension error');
end

%% 12. Complex Assignment Patterns
F = zeros(5, 5);
F(1:3, 1:3) = magic(3);           % Block assignment
F([1,5], [1,5]) = [99, 88; 77, 66]; % Scattered assignment
F(F < 5) = -1;                    % Logical assignment

%% 13. Dynamic Field Names
data.field1 = [1, 2, 3];
field_name = 'field1';
dynamic_access = data.(field_name); % Dynamic field access

%% 14. Nested Indexing
nested_cell = {{[1,2,3], [4,5,6]}, {[7,8,9]}};
deep_access = nested_cell{1}{2}(2); % Should give 5

%% 15. Broadcasting in Assignment
G = ones(3, 3);
G(:, 2) = [10; 20; 30];           % Column assignment
G(2, :) = 100;                    % Row assignment with scalar broadcast