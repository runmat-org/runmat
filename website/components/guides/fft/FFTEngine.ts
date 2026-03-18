export type WindowType = "rectangular" | "hann" | "hamming" | "blackman-harris";

export interface SignalComponent {
  frequency: number;
  amplitude: number;
  phase: number;
}

/**
 * Radix-2 Cooley-Tukey in-place FFT.
 * Input arrays must have power-of-2 length; they are modified in place.
 */
export function fft(
  re: number[],
  im: number[]
): { re: number[]; im: number[] } {
  const n = re.length;
  if (n <= 1) return { re, im };

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  // Butterfly stages
  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2;
    const angle = (-2 * Math.PI) / size;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);

    for (let start = 0; start < n; start += size) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < half; k++) {
        const evenIdx = start + k;
        const oddIdx = start + k + half;
        const tRe = curRe * re[oddIdx] - curIm * im[oddIdx];
        const tIm = curRe * im[oddIdx] + curIm * re[oddIdx];

        re[oddIdx] = re[evenIdx] - tRe;
        im[oddIdx] = im[evenIdx] - tIm;
        re[evenIdx] += tRe;
        im[evenIdx] += tIm;

        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }

  return { re, im };
}

export function magnitude(re: number[], im: number[]): number[] {
  const n = re.length;
  const mag = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    mag[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
  }
  return mag;
}

function windowHann(n: number): number[] {
  const w = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  }
  return w;
}

function windowHamming(n: number): number[] {
  const w = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
  }
  return w;
}

function windowBlackmanHarris(n: number): number[] {
  const a0 = 0.35875;
  const a1 = 0.48829;
  const a2 = 0.14128;
  const a3 = 0.01168;
  const w = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const x = (2 * Math.PI * i) / (n - 1);
    w[i] = a0 - a1 * Math.cos(x) + a2 * Math.cos(2 * x) - a3 * Math.cos(3 * x);
  }
  return w;
}

export function getWindowCoefficients(n: number, windowType: WindowType): number[] {
  switch (windowType) {
    case "hann":
      return windowHann(n);
    case "hamming":
      return windowHamming(n);
    case "blackman-harris":
      return windowBlackmanHarris(n);
    case "rectangular":
    default: {
      const w = new Array<number>(n);
      for (let i = 0; i < n; i++) w[i] = 1;
      return w;
    }
  }
}

export function applyWindow(signal: number[], windowType: WindowType): number[] {
  const w = getWindowCoefficients(signal.length, windowType);
  return signal.map((s, i) => s * w[i]);
}

export function generateSignal(
  components: SignalComponent[],
  sampleRate: number,
  numSamples: number
): number[] {
  const signal = new Array<number>(numSamples).fill(0);
  const dt = 1 / sampleRate;
  for (const { frequency, amplitude, phase } of components) {
    for (let i = 0; i < numSamples; i++) {
      signal[i] += amplitude * Math.sin(2 * Math.PI * frequency * i * dt + phase);
    }
  }
  return signal;
}

/** Pad or truncate to the next power of 2 >= n. */
export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}
