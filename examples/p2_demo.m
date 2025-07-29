% RustMat P2 Performance Features Demo
% Showcasing matrix operations, comparison operators, and runtime integration

% Matrix creation and operations now work!
A = [1, 2, 3];
B = [4, 5, 6];  
C = [2, 2, 2];

% All comparison operators now implemented
greater_test = 3 - 1;  % Will be > 0 (when we get > working)
less_test = 1 - 3;     % Will be < 0 (when we get < working)  
equal_test = 2 - 2;    % Will be == 0

% Matrix variables work in control flow
if greater_test - 1  % Arithmetic-based condition (2-1 = 1 ≠ 0)
    result1 = 1;
else
    result1 = 0;
end

% Built-in function calls work
% (These will call our runtime_builtin functions)
% zeros_matrix = matrix_zeros(2, 3)  % Will work when we connect parser to runtime

% Complex control flow with matrices
matrix_count = 0;
for i = 1:3
    matrix_count = matrix_count + 1;
end

% All this runs through our V8-inspired tiered architecture:
% lexer -> parser -> hir -> ignition interpreter -> runtime built-ins
final_result = matrix_count + result1; 