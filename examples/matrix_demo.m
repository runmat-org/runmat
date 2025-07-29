% RustMat Matrix Demo - showcasing our new matrix support
% This demonstrates our V8-inspired MATLAB runtime with matrices

% Basic matrix creation
A = [1, 2];
B = [3, 4];

% Variable assignment with matrices
matrix1 = [1, 2, 3];
matrix2 = [4, 5, 6];

% Basic arithmetic still works
x = 1 + 2;
y = x * 3;

% Control flow with matrices
if y - 9  % Should be false (y = 9)
    result = 0;
else
    result = 1;
end

% Loops work too
sum = 0;
for i = 1:3
    sum = sum + i;
end 