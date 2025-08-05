% Beautiful Sine Wave Demonstration
% Multiple overlapping sine waves with different frequencies

% Time vector
t = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0];

% Basic sine wave
y1 = [];
for i = 1:21
    y1 = [y1, sin(6.28 * t(i))]; % 2*pi approximation
end

% Higher frequency sine wave
y2 = [];
for i = 1:21
    y2 = [y2, sin(12.56 * t(i)) * 0.5]; % 4*pi approximation
end

% Combined wave
y3 = [];
for i = 1:21
    y3 = [y3, y1(i) + y2(i)];
end

% Some test calculations
amplitude1 = 1.0;
amplitude2 = 0.5;
frequency1 = 1.0;
frequency2 = 2.0;

result = amplitude1 + amplitude2;