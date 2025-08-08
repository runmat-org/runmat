% RunMat Real Matrix Values Demo
% Verifying that matrix compilation uses actual values

% Test 1D matrices
row_vector = [1, 2, 3];
another_vector = [10, 20];

% Test 2D matrices  
small_matrix = [1, 2; 3, 4];
larger_matrix = [1, 2, 3; 4, 5, 6];

% Test single element matrix
single = [42];

% Test empty matrix (edge case)
empty_row = [];

% All of these should now contain the actual values,
% not just zeros like before!
result = 1; 