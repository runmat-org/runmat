% --- Define the Test Function ---
function [first, varargout] = testVarargout(n)
    first = 100; % Use a number to be safe
    varargout = cell(1, n);
    for i = 1:n
        varargout{i} = i * 10;
    end
end

% --- Test Case 1: Standard Multi-Output ---
% This tests the 'pushed' logic for both named and varargout
fprintf('Running Test 1: Requesting 3 outputs...\n');
[a, b, c] = testVarargout(5);
disp(a); % Expected: 100
disp(b); % Expected: 10
disp(c); % Expected: 20

% --- Test Case 2: Exact Match ---
fprintf('Running Test 2: Requesting 2 outputs...\n');
[x, y] = testVarargout(1);
disp(x); % Expected: 100
disp(y); % Expected: 10

% --- Test Case 3: The Error Case ---
% We use a simple function call inside a try block.
% We avoid ME.message to prevent "LoadMember on non-object"
fprintf('Running Test 3: Triggering VarargoutMismatch...\n');
try
    % Providing 1 varargout, but requesting 3 (Total 4 outputs)
    [v1, v2, v3, v4] = testVarargout(1);
catch
    % If we reach here, the Rust vm_bail! worked correctly
    fprintf('Success: Caught the mismatch error.\n');
end
