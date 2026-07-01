% Neutral signal-processing compatibility harness for CLI and WASM runtimes.
tol = 1e-9;

filewrite('signal.csv', sprintf('0\n1\n0\n-1\n'));
csvSignal = readmatrix('signal.csv');
assert(length(csvSignal) == 4, 'CSV import length mismatch');
assert(abs(csvSignal(2) - 1) < tol, 'CSV import value mismatch');

spectrum = fft(csvSignal);
mag = abs(spectrum);
assert(abs(mag(2) - 2) < tol, 'FFT magnitude bin 2 mismatch');
assert(abs(mag(4) - 2) < tol, 'FFT magnitude bin 4 mismatch');

filtered = filter([1; -1], 1, csvSignal);
assert(abs(filtered(2) - 1) < tol, 'filter output mismatch');
assert(abs(filtered(3) + 1) < tol, 'filter output mismatch');

convOut = conv(csvSignal, [1; 1]);
assert(length(convOut) == 5, 'conv length mismatch');
assert(abs(convOut(3) - 1) < tol, 'conv middle sample mismatch');
assert(abs(convOut(5) + 1) < tol, 'conv tail sample mismatch');

hw = hamming(4);
hn = hann(4);
bw = blackman(4);
assert(abs(hw(1) - 0.08) < 1e-9, 'hamming endpoint mismatch');
assert(abs(hn(1)) < tol, 'hann endpoint mismatch');
assert(abs(bw(1)) < 1e-9, 'blackman endpoint mismatch');

save('signal_state.mat', 'csvSignal', 'convOut', 'filtered', 'hw', 'hn', 'bw');
loaded = load('signal_state.mat');
assert(abs(loaded.csvSignal(4) + 1) < tol, 'MAT load csvSignal mismatch');
assert(abs(loaded.filtered(2) - 1) < tol, 'MAT load filtered mismatch');
assert(abs(loaded.convOut(5) + 1) < tol, 'MAT load convOut mismatch');
assert(abs(loaded.hw(1) - hw(1)) < tol, 'MAT load window mismatch');

fprintf('RESULT_signal_compat csv=%d fft=%.1f conv=%.1f mat=%.1f\n', length(csvSignal), mag(2), convOut(5), loaded.filtered(2));
