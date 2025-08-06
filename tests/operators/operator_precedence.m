% MATLAB Operator Precedence and Edge Cases
% This file tests complex operator combinations and precedence rules

%% 1. Arithmetic Operator Precedence
% MATLAB precedence: () > ^ > .^ > unary +/- > */ .* ./ .\ / \ > +- > :

% Basic precedence tests
result1 = 2 + 3 * 4;        % Should be 14 (not 20)
result2 = 2 * 3 + 4;        % Should be 10
result3 = 2^3^2;            % Should be 512 (right associative: 2^(3^2))
result4 = 2^3*4;            % Should be 32 (2^3 = 8, then 8*4)

% Parentheses override
result5 = (2 + 3) * 4;      % Should be 20
result6 = 2 + (3 * 4);      % Should be 14 (same as without parens)
result7 = 2^(3^2);          % Should be 512
result8 = (2^3)^2;          % Should be 64

%% 2. Element-wise vs Matrix Operations
A = [1, 2; 3, 4];
B = [2, 1; 1, 2];

% Matrix operations
matrix_mult = A * B;        % Matrix multiplication
matrix_power = A^2;         % Matrix power

% Element-wise operations
elem_mult = A .* B;         % Element-wise multiplication
elem_power = A .^ 2;        % Element-wise power
elem_div = A ./ B;          % Element-wise division
elem_left_div = A .\ B;     % Element-wise left division

% Mixed operations (precedence matters)
mixed1 = A + B .* 2;        % B .* 2 first, then A + result
mixed2 = (A + B) .* 2;      % Addition first, then element-wise mult
mixed3 = A .* B + 1;        % Element-wise mult first, then add 1

%% 3. Unary Operator Precedence
x = 5;
neg_power = -x^2;           % Should be -25 (-(x^2), not (-x)^2)
neg_power_paren = (-x)^2;   % Should be 25
pos_neg = +-x;              % Should be -5
double_neg = --x;           % Should be 5

% With matrices
neg_matrix = -A^2;          % Negation of matrix power
neg_elem = -A.^2;           % Element-wise: -(A.^2)

%% 4. Logical Operator Precedence
% Precedence: ~ > & | > && ||
a = true; b = false; c = true;

logical1 = ~a & b;          % Should be false (true & false)
logical2 = a & b | c;       % Should be true ((a & b) | c)
logical3 = a | b & c;       % Should be true (a | (b & c))
logical4 = ~a | b & c;      % Should be false ((~a) | (b & c))

% Short-circuit vs element-wise
logical5 = a && b || c;     % Short-circuit: true
logical6 = a & b | c;       % Element-wise (same result for scalars)

%% 5. Comparison Operator Precedence
% Precedence: < <= > >= == ~=
x = 5; y = 10; z = 5;

comp1 = x < y == z < y;     % Should be true (true == true)
comp2 = x <= z >= y;        % Should be false (true >= false)
comp3 = x ~= y == z ~= y;   % Should be true (true == true)

% With logical operators
comp_logical = x < y & z == x;  % Should be true (true & true)
comp_logical2 = x > y | z <= x; % Should be true (false | true)

%% 6. Colon Operator Precedence
% Colon has lower precedence than most operators
range1 = 1:2+3;             % Should be 1:5 (1 2 3 4 5)
range2 = 1+2:5;             % Should be 3:5 (3 4 5)
range3 = 1:2*3;             % Should be 1:6 (1 2 3 4 5 6)
range4 = 2*1:3;             % Should be 2:3 (2 3)

% Step size
step_range1 = 1:2:2*5;      % Should be 1:2:10 (1 3 5 7 9)
step_range2 = 1:2+1:10;     % Should be 1:3:10 (1 4 7 10)

%% 7. Complex Operator Combinations
% Multiple levels of precedence
complex1 = 2 + 3 * 4^2 - 1;           % Should be 49 (2 + 3*16 - 1)
complex2 = -2^2 + 3 * -4;             % Should be -16 (-(2^2) + 3*(-4))
complex3 = 2^3^2 / 4 + 1;             % Should be 129 (512/4 + 1)

% With element-wise operations
A = [1, 2]; B = [3, 4];
complex4 = A + B .* 2^2;              % Should be [13, 18] (A + B*4)
complex5 = (A + B) .* 2.^A;           % Should be [8, 24] ([4,6] .* [2,4])

%% 8. Assignment Operator Precedence
% Assignment has very low precedence
x = y = z = 5;              % Right associative assignment (if supported)

% Compound assignment (if supported)
% x += 2 * 3;               % Should be equivalent to x = x + (2 * 3)
% x *= 2 + 3;               % Should be equivalent to x = x * (2 + 3)

%% 9. Function Call Precedence
% Function calls have high precedence
func_result = sin(pi/2) + cos(0);     % Both function calls first
func_result2 = sin(pi/2 + 0);         % Addition inside function call
func_result3 = sin(pi) / 2 + 1;       % sin(pi) first, then /2, then +1

% With matrix operations
matrix_func = sin(A) + cos(B);        % Element-wise functions
matrix_func2 = sin(A * B);            % Matrix mult first, then sin

%% 10. Indexing Precedence
% Indexing has very high precedence
C = [1, 2, 3; 4, 5, 6];
index1 = C(1,:) + 1;                  % Indexing first: [2, 3, 4]
index2 = C(1,:) * 2;                  % Indexing first: [2, 4, 6]
index3 = C(1,1:2) .* [10, 20];        % Complex indexing with element-wise

% Dynamic indexing
idx = 2;
dyn_index = C(idx, :) + C(1, :);      % Both indexing ops first

%% 11. Transpose Operator Precedence
% Transpose has high precedence
D = [1, 2, 3];
trans1 = D' + 1;                      % Transpose first, then add
trans2 = (D + 1)';                    % Addition first, then transpose
trans3 = D' * D;                      % Both transposes first, then multiply

% Complex transpose
trans4 = D.^2';                       % Power first, then transpose
trans5 = (D.^2)';                     % Same as above
trans6 = D.^(2');                     % Transpose of 2, then power

%% 12. Cell Array and Structure Precedence
% High precedence for {} and . operators
cell_data = {[1,2], [3,4]};
struct_data.field = [5, 6];

cell_op = cell_data{1} + cell_data{2};     % Cell access first
struct_op = struct_data.field * 2;         % Field access first
mixed_op = cell_data{1} + struct_data.field; % Both access ops first

%% 13. Error-prone Precedence Cases
% Cases that commonly cause errors
error_prone1 = 2 * 3^2;               % 18, not 36
error_prone2 = -3^2;                  % -9, not 9
error_prone3 = 2 + 3 == 5;            % true, not 2
error_prone4 = 1:5 + 1;               % [2,3,4,5,6], not 1:6

% Logical precedence confusion
bool1 = true; bool2 = false; bool3 = true;
error_prone5 = bool1 | bool2 & bool3;   % true (| has lower precedence)
error_prone6 = (bool1 | bool2) & bool3; % true (explicit grouping)

%% 14. Type Conversion in Operations
% Implicit type conversions affect results
int_val = int32(5);
double_val = 3.7;
mixed_type = int_val + double_val;    % Result type and value

logical_val = true;
logical_arith = logical_val + 1;      % Should be 2 (true = 1)

%% 15. NaN and Inf Propagation
nan_val = NaN;
inf_val = Inf;

nan_ops = nan_val + 1;                % NaN
nan_ops2 = nan_val * 0;               % NaN (not 0)
inf_ops = inf_val - inf_val;          % NaN
inf_ops2 = inf_val / inf_val;         % NaN

% Comparisons with NaN
nan_compare = nan_val == nan_val;     % false!
nan_compare2 = isnan(nan_val);        % true

fprintf('Operator precedence tests completed.\n');