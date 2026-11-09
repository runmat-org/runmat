function r = rcResistor(fc, c)
% The resistor that puts an RC low-pass corner at fc (SI: Hz, F, ohm).
r = 1 / (2 * pi * fc * c);
end
