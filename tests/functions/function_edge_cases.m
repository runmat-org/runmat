% MATLAB Function Definition and Call Edge Cases
% This file tests all aspects of MATLAB function syntax and semantics

%% 1. Function Definition Patterns (in separate files typically)

% Basic function (would be in separate .m file)
% function result = simple_add(a, b)
%     result = a + b;
% end

% Multiple output function
% function [sum_val, diff_val, prod_val] = math_ops(a, b)
%     sum_val = a + b;
%     diff_val = a - b;
%     prod_val = a * b;
% end

% Variable number of inputs (varargin)
% function result = variable_inputs(varargin)
%     result = sum([varargin{:}]);
% end

% Variable number of outputs (varargout)
% function varargout = variable_outputs(x)
%     varargout{1} = x^2;
%     if nargout > 1
%         varargout{2} = x^3;
%     end
%     if nargout > 2
%         varargout{3} = x^4;
%     end
% end

%% 2. Anonymous Functions
% Basic anonymous functions
add_anon = @(x, y) x + y;
square_anon = @(x) x.^2;
complex_anon = @(x, y, z) sqrt(x^2 + y^2 + z^2);

% Test anonymous functions
result1 = add_anon(3, 4);             % Should be 7
result2 = square_anon([1, 2, 3]);     % Should be [1, 4, 9]
result3 = complex_anon(3, 4, 5);      % Should be ~7.07

% Anonymous functions with captured variables
multiplier = 5;
scale_func = @(x) x * multiplier;
scaled_result = scale_func(10);       % Should be 50

% Nested anonymous functions
compose_func = @(f, g, x) f(g(x));
double_square = compose_func(square_anon, @(x) x*2, 3); % (3*2)^2 = 36

%% 3. Function Handles
% Handle to built-in functions
sin_handle = @sin;
cos_handle = @cos;
max_handle = @max;

% Function handle operations
trig_result = sin_handle(pi/2);       % Should be 1
array_max = max_handle([1, 5, 3]);    % Should be 5

% Function handle comparison and properties
func_info = functions(sin_handle);
is_same = isequal(sin_handle, @sin);

%% 4. Nested Function Scope (would be in .m file)
% function result = outer_function(x)
%     factor = 10;
%     
%     function inner_result = inner_function(y)
%         inner_result = y * factor;  % Access to outer scope
%     end
%     
%     result = inner_function(x) + 5;
% end

%% 5. Function Input/Output Edge Cases

% Test with different argument counts
% single_out = math_ops(5, 3);         % Only first output
% [s, d] = math_ops(5, 3);             % Two outputs
% [s, d, p] = math_ops(5, 3);          % All three outputs

% Variable arguments
% var_result1 = variable_inputs(1);
% var_result2 = variable_inputs(1, 2, 3, 4);

% Variable outputs
% single_var = variable_outputs(3);    % x^2 = 9
% [sq, cb] = variable_outputs(3);      % x^2=9, x^3=27
% [sq, cb, qt] = variable_outputs(3);  % x^2=9, x^3=27, x^4=81

%% 6. Input Validation and Error Handling

% Function with input validation (nargin/nargout)
% function result = validated_function(a, b, c)
%     if nargin < 2
%         error('At least 2 arguments required');
%     end
%     if nargin < 3
%         c = 0;  % Default value
%     end
%     
%     if nargout > 1
%         error('Too many output arguments');
%     end
%     
%     result = a + b + c;
% end

% Modern input validation (arguments block)
% function result = modern_validation(x, options)
%     arguments
%         x (1,1) double {mustBePositive}
%         options.method char = 'linear'
%         options.extrapolate logical = false
%     end
%     
%     result = x * 2;  % Simple operation
% end

%% 7. Function Visibility and Scope

% Local functions (in same file, after main function)
% function main_result = main_function()
%     main_result = local_helper(5);
% end
% 
% function helper_result = local_helper(x)
%     helper_result = x^2;
% end

% Private functions (in private/ subdirectory)
% These are only accessible to functions in the parent directory

%% 8. Function Overloading and Polymorphism

% Method overloading based on number of arguments
% function result = overloaded_func(a)
%     result = a^2;
% end
% 
% function result = overloaded_func(a, b)
%     result = a + b;
% end

% Class method overloading (methods in class definition)
% Would override built-in operators for custom classes

%% 9. Recursive Functions
% Simple recursion example
% function result = factorial_func(n)
%     if n <= 1
%         result = 1;
%     else
%         result = n * factorial_func(n - 1);
%     end
% end

% Mutual recursion
% function result = even_func(n)
%     if n == 0
%         result = true;
%     else
%         result = odd_func(n - 1);
%     end
% end
% 
% function result = odd_func(n)
%     if n == 0
%         result = false;
%     else
%         result = even_func(n - 1);
%     end
% end

%% 10. Function as Data (Higher-order functions)

% Array of function handles
func_array = {@sin, @cos, @tan};
angle = pi/4;
trig_results = cellfun(@(f) f(angle), func_array);

% Function that returns a function
% function func_handle = make_multiplier(factor)
%     func_handle = @(x) x * factor;
% end
% 
% times_three = make_multiplier(3);
% result = times_three(7);  % Should be 21

%% 11. Built-in Function Edge Cases

% Functions with optional arguments
rand_default = rand();                % Single random number
rand_size = rand(3, 4);              % 3x4 matrix
rand_like = rand(size([1,2,3]));      % Size from another array

% Functions with name-value pairs
% plot_result = plot([1,2,3], [4,5,6], 'Color', 'red', 'LineWidth', 2);

%% 12. Function Performance and Optimization

% Vectorized vs loop comparison
x = 1:1000;

% Inefficient: element-by-element processing
% for i = 1:length(x)
%     slow_result(i) = sin(x(i))^2 + cos(x(i))^2;
% end

% Efficient: vectorized processing
fast_result = sin(x).^2 + cos(x).^2;

% Function caching/memoization concept
% persistent cache;
% if isempty(cache)
%     cache = containers.Map();
% end

%% 13. Function Handle Edge Cases

% Function handles with different signatures
simple_handle = @(x) x + 1;
complex_handle = @(x, y, varargin) x + y + sum([varargin{:}]);

% Calling with different argument counts
simple_call = simple_handle(5);
% complex_call = complex_handle(1, 2, 3, 4, 5);

% Function handle to methods
% obj_method = @(obj, x) obj.process(x);

%% 14. Error Handling in Functions

% Function that can throw errors
% function result = error_prone_func(x)
%     if x < 0
%         error('Input must be non-negative');
%     elseif x == 0
%         warning('Input is zero, result may be undefined');
%         result = NaN;
%     else
%         result = sqrt(x);
%     end
% end

% try
%     safe_result = error_prone_func(-5);
% catch ME
%     fprintf('Caught error: %s\n', ME.message);
% end

%% 15. Function Documentation and Help

% Functions with help text
% function result = documented_function(x, y)
% % DOCUMENTED_FUNCTION - Adds two numbers with documentation
% %
% % Syntax: result = documented_function(x, y)
% %
% % Inputs:
% %   x - First number (double)
% %   y - Second number (double)
% %
% % Output:
% %   result - Sum of x and y
% %
% % Example:
% %   result = documented_function(3, 4);  % Returns 7
% 
%     result = x + y;
% end

%% 16. Function Workspace and Variable Scope

% Global variables
global GLOBAL_CONSTANT;
GLOBAL_CONSTANT = 42;

% function result = global_user()
%     global GLOBAL_CONSTANT;
%     result = GLOBAL_CONSTANT * 2;
% end

% Persistent variables
% function result = counter()
%     persistent count;
%     if isempty(count)
%         count = 0;
%     end
%     count = count + 1;
%     result = count;
% end

%% 17. Function Call Syntax Variations

% Standard function call
std_result = max([1, 5, 3]);

% Command syntax (for functions accepting char arrays)
% help sin     % Equivalent to help('sin')
% clear x y z  % Equivalent to clear('x', 'y', 'z')

% Subscript function calls (rare)
% subsref_result = max.([1, 5, 3]);  % Alternative syntax

%% 18. Special Function Types

% Script functions (no function keyword, just statements)
% script_var = 42;

% Function files vs script files
% .m files can contain either functions or scripts

% Live functions (in .mlx files, if supported)

%% Test some working function handle operations
fprintf('Testing function handles:\n');
fprintf('sin(π/2) = %.6f\n', sin_handle(pi/2));
fprintf('Anonymous add(3,4) = %d\n', add_anon(3, 4));
fprintf('Square of [1,2,3] = [%d, %d, %d]\n', square_anon([1,2,3]));
fprintf('Function edge case tests completed.\n');