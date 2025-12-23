% Simple counter benchmark for consistent timing
x = 0.0;
for i = 1:1000000000
    x = x + 1.0;
end
result = x;
