% Test symbolic math in RunMat
x = sym('x')
y = sym('y')
expr = x^2 + 3*x + 1
diff(expr, x)
