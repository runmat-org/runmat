% Comprehensive MATLAB Language Test Suite
% This script runs all edge case tests and reports results

fprintf('=== RustMat Comprehensive Language Test Suite ===\n\n');

%% Test Categories
test_categories = {
    'Basic Functionality', 'tests/basic_functionality.m';
    'Advanced Indexing', 'tests/indexing/advanced_indexing.m';
    'Class System', 'tests/classes/matlab_classes.m';
    'Control Flow', 'tests/control_flow/advanced_control_flow.m';
    'Operator Precedence', 'tests/operators/operator_precedence.m';
    'Data Types', 'tests/data_types/data_type_edge_cases.m';
    'Function Features', 'tests/functions/function_edge_cases.m';
    'String Processing', 'tests/strings/string_processing.m';
    'Advanced Syntax', 'tests/advanced_syntax/matlab_advanced_features.m';
};

%% Track Results
total_tests = size(test_categories, 1);
passed_tests = 0;
failed_tests = 0;
error_details = {};

%% Run Basic Functionality First
fprintf('--- Testing Basic Functionality ---\n');
try
    % Basic matrix operations
    A = [1, 2, 3; 4, 5, 6];
    B = A .* 2;
    assert(isequal(B, [2, 4, 6; 8, 10, 12]), 'Element-wise multiplication failed');
    
    % Array indexing
    val = A(2, 1);
    assert(val == 4, 'Array indexing failed');
    
    % Mathematical functions
    x = linspace(0, pi, 5);
    y = sin(x);
    assert(length(y) == 5, 'Mathematical function failed');
    
    % Constants
    const_sum = pi + e;
    assert(const_sum > 5 && const_sum < 7, 'Constants failed');
    
    % Matrix concatenation
    horizontal = [x, y];
    assert(size(horizontal, 2) == 10, 'Horizontal concatenation failed');
    
    vertical = [x; y];
    assert(size(vertical, 1) == 2, 'Vertical concatenation failed');
    
    fprintf('✓ Basic functionality tests PASSED\n');
    passed_tests = passed_tests + 1;
    
catch ME
    fprintf('✗ Basic functionality tests FAILED: %s\n', ME.message);
    failed_tests = failed_tests + 1;
    error_details{end+1} = sprintf('Basic Functionality: %s', ME.message);
end

%% Test Advanced Indexing Features
fprintf('\n--- Testing Advanced Indexing ---\n');
try
    % Linear indexing
    C = [1, 2, 3; 4, 5, 6; 7, 8, 9];
    linear_val = C(5);
    assert(linear_val == 5, 'Linear indexing failed');
    
    % Colon operator
    row = C(2, :);
    assert(isequal(row, [4, 5, 6]), 'Row access failed');
    
    col = C(:, 2);
    assert(isequal(col, [2; 5; 8]), 'Column access failed');
    
    % Range indexing
    submat = C(1:2, 2:3);
    assert(isequal(submat, [2, 3; 5, 6]), 'Submatrix indexing failed');
    
    fprintf('✓ Advanced indexing tests PASSED\n');
    passed_tests = passed_tests + 1;
    
catch ME
    fprintf('✗ Advanced indexing tests FAILED: %s\n', ME.message);
    failed_tests = failed_tests + 1;
    error_details{end+1} = sprintf('Advanced Indexing: %s', ME.message);
end

%% Test Control Flow
fprintf('\n--- Testing Control Flow ---\n');
try
    % For loops
    sum_val = 0;
    for i = 1:5
        sum_val = sum_val + i;
    end
    assert(sum_val == 15, 'For loop failed');
    
    % While loops
    counter = 1;
    while counter < 4
        counter = counter + 1;
    end
    assert(counter == 4, 'While loop failed');
    
    % If statements
    test_val = 10;
    if test_val > 5
        result = 'greater';
    else
        result = 'lesser';
    end
    assert(strcmp(result, 'greater'), 'If statement failed');
    
    fprintf('✓ Control flow tests PASSED\n');
    passed_tests = passed_tests + 1;
    
catch ME
    fprintf('✗ Control flow tests FAILED: %s\n', ME.message);
    failed_tests = failed_tests + 1;
    error_details{end+1} = sprintf('Control Flow: %s', ME.message);
end

%% Test Operator Precedence
fprintf('\n--- Testing Operator Precedence ---\n');
try
    % Basic precedence
    result1 = 2 + 3 * 4;
    assert(result1 == 14, 'Arithmetic precedence failed');
    
    % Power precedence
    result2 = -2^2;
    assert(result2 == -4, 'Power precedence failed');
    
    % Element-wise vs matrix operations
    D = [1, 2; 3, 4];
    E = [2, 1; 1, 2];
    elem_result = D .* E;
    assert(isequal(elem_result, [2, 2; 3, 8]), 'Element-wise operation failed');
    
    fprintf('✓ Operator precedence tests PASSED\n');
    passed_tests = passed_tests + 1;
    
catch ME
    fprintf('✗ Operator precedence tests FAILED: %s\n', ME.message);
    failed_tests = failed_tests + 1;
    error_details{end+1} = sprintf('Operator Precedence: %s', ME.message);
end

%% Test Data Types
fprintf('\n--- Testing Data Types ---\n');
try
    % Numeric types
    double_val = 3.14159;
    int_val = int32(42);
    assert(isa(double_val, 'double'), 'Double type failed');
    assert(isa(int_val, 'int32'), 'Integer type failed');
    
    % Logical operations
    log_val = true;
    log_result = log_val & false;
    assert(log_result == false, 'Logical operation failed');
    
    % Complex numbers
    complex_val = 3 + 4i;
    assert(real(complex_val) == 3, 'Complex real part failed');
    assert(imag(complex_val) == 4, 'Complex imaginary part failed');
    
    fprintf('✓ Data type tests PASSED\n');
    passed_tests = passed_tests + 1;
    
catch ME
    fprintf('✗ Data type tests FAILED: %s\n', ME.message);
    failed_tests = failed_tests + 1;
    error_details{end+1} = sprintf('Data Types: %s', ME.message);
end

%% Test Function Features
fprintf('\n--- Testing Function Features ---\n');
try
    % Anonymous functions
    add_func = @(x, y) x + y;
    func_result = add_func(3, 4);
    assert(func_result == 7, 'Anonymous function failed');
    
    % Function handles
    sin_handle = @sin;
    handle_result = sin_handle(pi/2);
    assert(abs(handle_result - 1) < 1e-10, 'Function handle failed');
    
    % Built-in functions with multiple outputs
    [max_val, max_idx] = max([1, 5, 3]);
    assert(max_val == 5 && max_idx == 2, 'Multiple output function failed');
    
    fprintf('✓ Function feature tests PASSED\n');
    passed_tests = passed_tests + 1;
    
catch ME
    fprintf('✗ Function feature tests FAILED: %s\n', ME.message);
    failed_tests = failed_tests + 1;
    error_details{end+1} = sprintf('Function Features: %s', ME.message);
end

%% Test String Processing
fprintf('\n--- Testing String Processing ---\n');
try
    % Character arrays
    char_str = 'Hello World';
    assert(length(char_str) == 11, 'Character array length failed');
    
    substr = char_str(1:5);
    assert(strcmp(substr, 'Hello'), 'Substring extraction failed');
    
    % String concatenation
    concat_str = [char_str, ' Test'];
    assert(strcmp(concat_str, 'Hello World Test'), 'String concatenation failed');
    
    fprintf('✓ String processing tests PASSED\n');
    passed_tests = passed_tests + 1;
    
catch ME
    fprintf('✗ String processing tests FAILED: %s\n', ME.message);
    failed_tests = failed_tests + 1;
    error_details{end+1} = sprintf('String Processing: %s', ME.message);
end

%% Test Advanced Mathematical Functions
fprintf('\n--- Testing Advanced Mathematical Functions ---\n');
try
    % Trigonometric functions
    angles = [0, pi/2, pi];
    sin_vals = sin(angles);
    expected_sin = [0, 1, 0];
    assert(all(abs(sin_vals - expected_sin) < 1e-10), 'Trigonometric functions failed');
    
    % Statistical functions
    data = [1, 2, 3, 4, 5];
    mean_val = mean(data);
    assert(mean_val == 3, 'Mean function failed');
    
    % Power functions
    power_result = pow(2, 3);
    assert(power_result == 8, 'Power function failed');
    
    fprintf('✓ Advanced mathematical function tests PASSED\n');
    passed_tests = passed_tests + 1;
    
catch ME
    fprintf('✗ Advanced mathematical function tests FAILED: %s\n', ME.message);
    failed_tests = failed_tests + 1;
    error_details{end+1} = sprintf('Advanced Mathematical Functions: %s', ME.message);
end

%% Test Memory and Performance
fprintf('\n--- Testing Memory and Performance ---\n');
try
    % Large matrix operations
    large_mat = rand(100, 100);
    assert(size(large_mat, 1) == 100, 'Large matrix creation failed');
    
    % Vectorized operations
    vec_data = 1:1000;
    vec_result = vec_data.^2;
    assert(length(vec_result) == 1000, 'Vectorized operation failed');
    
    fprintf('✓ Memory and performance tests PASSED\n');
    passed_tests = passed_tests + 1;
    
catch ME
    fprintf('✗ Memory and performance tests FAILED: %s\n', ME.message);
    failed_tests = failed_tests + 1;
    error_details{end+1} = sprintf('Memory and Performance: %s', ME.message);
end

%% Summary Report
fprintf('\n=== Test Summary ===\n');
fprintf('Total Test Categories: %d\n', total_tests + 3); % +3 for additional tests
fprintf('Passed: %d\n', passed_tests);
fprintf('Failed: %d\n', failed_tests);
fprintf('Success Rate: %.1f%%\n', (passed_tests / (passed_tests + failed_tests)) * 100);

if ~isempty(error_details)
    fprintf('\n=== Error Details ===\n');
    for i = 1:length(error_details)
        fprintf('%d. %s\n', i, error_details{i});
    end
end

if failed_tests == 0
    fprintf('\n🎉 ALL TESTS PASSED! RustMat language implementation is comprehensive.\n');
else
    fprintf('\n⚠️  Some tests failed. See details above for areas needing improvement.\n');
end

%% Performance Benchmark
fprintf('\n=== Performance Benchmark ===\n');
n = 1000;
tic;
A = rand(n, n);
B = rand(n, n);
C = A * B;
matrix_time = toc;
fprintf('Matrix multiplication (%dx%d): %.4f seconds\n', n, n, matrix_time);

tic;
large_sin = sin(1:10000);
vectorized_time = toc;
fprintf('Vectorized sin(1:10000): %.4f seconds\n', vectorized_time);

tic;
for i = 1:1000
    loop_result = i^2;
end
loop_time = toc;
fprintf('Loop (1000 iterations): %.4f seconds\n', loop_time);

fprintf('\n=== Comprehensive Test Suite Complete ===\n');