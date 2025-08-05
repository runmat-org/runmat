% Simple sine wave test
x = [0, 1, 2, 3, 4, 5, 6];
y = [0, 0.841, 0.909, 0.141, -0.757, -0.959, -0.279];  % sin(x) values

% Create the plot
plot(x, y);

% Some calculations
amplitude = 1.0;
period = 6.28;
result = amplitude * period;