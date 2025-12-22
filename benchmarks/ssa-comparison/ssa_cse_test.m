% CSE (Common Subexpression Elimination) Test
% The expression (a*b) appears twice per iteration - should be computed once with CSE

a = 3.141592653;
b = 2.718281828;
x = 0.0;

for i = 1:500000
    t = a * b + 1.0;
    u = a * b - 1.0;
    x = x + t + u;
end

result = x;
