% RunMat Demo Script - showcasing current functionality
% This demonstrates what our V8-inspired MATLAB runtime can do

% Basic arithmetic and variables
x = 5;
y = 3.14;
result = x * y + 2;

% Control flow with arithmetic conditions
if result - 15
    status = 1;  % Greater than 15
else
    status = 0;  % Less than or equal to 15
end

% For loops with ranges
sum = 0;
for i = 1:5
    sum = sum + i;
end

% While loops
counter = 0;
while counter - 3  % While counter != 3
    counter = counter + 1;
end

% Nested control structures
factorial = 1;
for n = 1:5
    factorial = factorial * n;
    if factorial - 100  % If factorial > 100
        break;
    end
end 