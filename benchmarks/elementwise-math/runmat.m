points_default = 5000001;
if ~exist('points', 'var')
  points = points_default;
end
points = floor(points);
if points < 2
  points = 2;
end

x = single(linspace(0, 4 * pi, points));
y0 = sin(x) .* exp(-x / single(10));
y1 = y0 .* cos(x / 4) + single(0.25) .* (y0 .^ 2);
y2 = tanh(y1) + single(0.1) .* y1;

fprintf('RESULT_ok\n');

