% Advanced MATLAB Control Flow Edge Cases
% This file tests all control flow constructs with edge cases

%% 1. For Loop Variations
% Basic for loop
for i = 1:5
    fprintf('Basic loop: %d\n', i);
end

% Step size variations
for i = 1:2:10
    fprintf('Step 2: %d\n', i);
end

% Reverse iteration
for i = 10:-1:1
    fprintf('Reverse: %d\n', i);
end

% Fractional steps
for i = 0:0.5:3
    fprintf('Fractional: %.1f\n', i);
end

% Vector iteration
for val = [1, 4, 7, 10]
    fprintf('Vector element: %d\n', val);
end

% Matrix column iteration
A = [1, 2, 3; 4, 5, 6];
for col = A
    fprintf('Column sum: %d\n', sum(col));
end

% Cell array iteration
cell_data = {1, 'hello', [1,2,3]};
for element = cell_data
    fprintf('Cell type: %s\n', class(element{1}));
end

%% 2. While Loop Edge Cases
% Basic while
counter = 1;
while counter <= 3
    fprintf('While: %d\n', counter);
    counter = counter + 1;
end

% While with complex condition
x = 10;
while x > 1 && mod(x, 2) == 0
    x = x / 2;
    fprintf('While complex: %d\n', x);
end

% Infinite loop protection (commented out)
% while true
%     break;  % Immediate break
% end

%% 3. If-ElseIf-Else Variations
value = 5;

% Basic if
if value > 0
    fprintf('Positive\n');
end

% If-else
if value < 0
    fprintf('Negative\n');
else
    fprintf('Non-negative\n');
end

% If-elseif-else chain
grade = 85;
if grade >= 90
    letter = 'A';
elseif grade >= 80
    letter = 'B';
elseif grade >= 70
    letter = 'C';
elseif grade >= 60
    letter = 'D';
else
    letter = 'F';
end
fprintf('Grade: %c\n', letter);

% Complex logical conditions
a = 5; b = 10; c = 15;
if (a < b) && (b < c) || (a == c)
    fprintf('Complex condition true\n');
end

% Short-circuit evaluation
x = 0;
if x ~= 0 && (10 / x) > 1
    fprintf('This should not print due to short-circuit\n');
end

%% 4. Switch-Case Statements
day = 3;
switch day
    case 1
        day_name = 'Monday';
    case 2
        day_name = 'Tuesday';
    case 3
        day_name = 'Wednesday';
    case {4, 5}  % Multiple case values
        day_name = 'Thursday or Friday';
    case 6
        day_name = 'Saturday';
    case 7
        day_name = 'Sunday';
    otherwise
        day_name = 'Invalid day';
end
fprintf('Day: %s\n', day_name);

% String switch (if supported)
operation = 'add';
switch operation
    case 'add'
        result = a + b;
    case 'subtract'
        result = a - b;
    case 'multiply'
        result = a * b;
    case 'divide'
        if b ~= 0
            result = a / b;
        else
            result = NaN;
        end
    otherwise
        result = 0;
end

%% 5. Try-Catch Error Handling
try
    risky_operation = 1 / 0;  % Division by zero
    fprintf('No error occurred\n');
catch exception
    fprintf('Caught error: %s\n', exception.message);
end

% Try-catch with specific error handling
try
    A = [1, 2; 3, 4];
    value = A(10, 10);  % Index out of bounds
catch ME
    if strcmp(ME.identifier, 'MATLAB:badsubscript')
        fprintf('Index out of bounds error\n');
    else
        fprintf('Other error: %s\n', ME.message);
    end
end

% Nested try-catch
try
    try
        error('Inner error');
    catch inner_exception
        fprintf('Inner catch: %s\n', inner_exception.message);
        error('Outer error');  % Re-throw different error
    end
catch outer_exception
    fprintf('Outer catch: %s\n', outer_exception.message);
end

%% 6. Break and Continue
% Break in loops
for i = 1:10
    if i == 5
        break;
    end
    fprintf('Break loop: %d\n', i);
end

% Continue in loops
for i = 1:10
    if mod(i, 2) == 0
        continue;  % Skip even numbers
    end
    fprintf('Odd number: %d\n', i);
end

% Break in nested loops
for i = 1:3
    for j = 1:3
        if i == 2 && j == 2
            break;  % Only breaks inner loop
        end
        fprintf('Nested: (%d,%d)\n', i, j);
    end
end

%% 7. Nested Control Structures
for i = 1:5
    if mod(i, 2) == 0
        for j = 1:i
            if j > 3
                continue;
            end
            switch j
                case 1
                    fprintf('One ');
                case 2
                    fprintf('Two ');
                case 3
                    fprintf('Three ');
            end
        end
        fprintf('\n');
    end
end

%% 8. Logical Short-Circuiting Edge Cases
% Test short-circuit AND
false_condition = false;
if false_condition && error('This should not execute')
    fprintf('Should not reach here\n');
end

% Test short-circuit OR
true_condition = true;
if true_condition || error('This should not execute')
    fprintf('Short-circuit OR worked\n');
end

%% 9. Empty and Edge Case Loops
% Empty range
for i = 5:3  % Empty range, loop should not execute
    fprintf('This should not print\n');
end

% Single iteration
for i = 5:5
    fprintf('Single iteration: %d\n', i);
end

% Very large range (should handle efficiently)
count = 0;
for i = 1:1000000
    if i > 5  % Early termination for test
        break;
    end
    count = count + 1;
end
fprintf('Large range test, count: %d\n', count);

%% 10. Complex Condition Combinations
x = 5; y = 10; z = 15;
complex_result = (x < y && y < z) || (x > y && y > z) || (x == y);
if complex_result
    fprintf('Complex condition evaluation worked\n');
end

% Nested logical operations
if ~(~(x > 0) && ~(y > 0))  % Double negation
    fprintf('Double negation worked\n');
end

%% 11. Variable Scope in Control Structures
% For loop variable scope
clear i;  % Clear any existing i
for i = 1:3
    inner_var = i * 2;
end
% i should still exist after loop in MATLAB
fprintf('Loop variable i after loop: %d\n', i);

% If statement scope
if true
    if_scope_var = 42;
end
% if_scope_var should still exist
fprintf('Variable from if scope: %d\n', if_scope_var);

%% 12. Performance Edge Cases
% Vectorized vs loop performance comparison
n = 1000;
A = rand(n, 1);
B = rand(n, 1);

% Loop version (slower)
tic;
C_loop = zeros(n, 1);
for i = 1:n
    C_loop(i) = A(i) + B(i);
end
loop_time = toc;

% Vectorized version (faster)
tic;
C_vectorized = A + B;
vectorized_time = toc;

fprintf('Loop time: %.6f, Vectorized time: %.6f\n', loop_time, vectorized_time);