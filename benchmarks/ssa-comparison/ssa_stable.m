% Stable benchmark - moderate iteration count
x = 0.0;
for i = 1:100000000
    x = x + 1.0;
end

a = 3.14159;
b = 2.71828;
y = 0.0;
for i = 1:50000000
    t = a * b + 1.0;
    u = a * b - 1.0;
    y = y + t + u;
end

k = 7.0;
z = 0.0;
for i = 1:50000000
    z = z + k * k;
end

result = x + y + z;
