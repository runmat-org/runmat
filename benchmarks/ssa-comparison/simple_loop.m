% Simple loop without ranges - to test pure JIT path
x = 0;
n = 1000;
i = 1;
while i <= n
    x = x + 1;
    i = i + 1;
end
result = x;
