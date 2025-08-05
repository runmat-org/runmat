% Polynomial Curves and Functions
% Demonstrates various polynomial shapes

% Define range
x = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3];

% Linear function
y_linear = [];
for i = 1:13
    y_linear = [y_linear, 2 * x(i) + 1];
end

% Quadratic function  
y_quad = [];
for i = 1:13
    y_quad = [y_quad, x(i)^2 - 2];
end

% Cubic function
y_cubic = [];
for i = 1:13
    y_cubic = [y_cubic, x(i)^3 - 3 * x(i)];
end

% Quartic function with multiple turning points
y_quartic = [];
for i = 1:13
    y_quartic = [y_quartic, 0.2 * x(i)^4 - x(i)^2 + 1];
end

% Exponential-like growth (using power series approximation)
y_exp = [];
for i = 1:13
    val = x(i);
    % e^x ≈ 1 + x + x²/2 + x³/6 (first few terms)
    y_exp = [y_exp, 1 + val + val^2/2 + val^3/6];
end

% Some polynomial calculations
discriminant = 4 - 4 * 1 * (-2); % b²-4ac for y = x² - 2
root1 = sqrt(2);
root2 = -sqrt(2);