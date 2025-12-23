% 10-second SSA benchmark

% Test 1: Simple counter - 2B iterations
x = 0.0;
for i = 1:2000000000
    x = x + 1.0;
end

% Test 2: CSE opportunity - 1B iterations
a = 3.141592653;
b = 2.718281828;
y = 0.0;
for i = 1:1000000000
    t = a * b + 1.0;
    u = a * b - 1.0;
    y = y + t + u;
end

% Test 3: LICM opportunity - 1B iterations
k = 7.0;
z = 0.0;
for i = 1:1000000000
    z = z + k * k;
end

result = x + y + z;
