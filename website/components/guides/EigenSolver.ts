/**
 * Pure 2×2 eigensolver using the quadratic formula (characteristic polynomial).
 * No backend; runs in JS for instant updates in the eigenvalue explorer.
 */

export interface ComplexEigenvalue {
  re: number;
  im: number;
}

export type StabilityStatus = "stable" | "unstable" | "marginal";

export interface EigenResult {
  lambda1: ComplexEigenvalue;
  lambda2: ComplexEigenvalue;
  isStable: boolean;
  isMarginal: boolean;
  stability: StabilityStatus;
  trace: number;
  determinant: number;
  discriminant: number;
}

const MARGINAL_EPS = 0.05;

export function solveEig(
  a: number,
  b: number,
  c: number,
  d: number
): EigenResult {
  const trace = a + d;
  const det = a * d - b * c;
  const disc = trace * trace - 4 * det;

  let lambda1: ComplexEigenvalue;
  let lambda2: ComplexEigenvalue;

  if (disc >= 0) {
    const sqrtDisc = Math.sqrt(disc);
    lambda1 = { re: (trace + sqrtDisc) / 2, im: 0 };
    lambda2 = { re: (trace - sqrtDisc) / 2, im: 0 };
  } else {
    const re = trace / 2;
    const im = Math.sqrt(-disc) / 2;
    lambda1 = { re, im };
    lambda2 = { re, im: -im };
  }

  const re1 = lambda1.re;
  const re2 = lambda2.re;
  const bothStable = re1 < 0 && re2 < 0;
  const anyUnstable = re1 > 0 || re2 > 0;
  const nearZero = (r: number) => Math.abs(r) <= MARGINAL_EPS;
  const isMarginal = nearZero(re1) || nearZero(re2);

  let stability: StabilityStatus;
  if (anyUnstable) stability = "unstable";
  else if (isMarginal) stability = "marginal";
  else stability = "stable";

  return {
    lambda1,
    lambda2,
    isStable: bothStable && !isMarginal,
    isMarginal,
    stability,
    trace,
    determinant: det,
    discriminant: disc,
  };
}

const FMT_PRECISION = 2;
const FMT_EPS = 1e-10;

function roundForDisplay(x: number): number {
  if (Math.abs(x) < FMT_EPS) return 0;
  const m = Math.pow(10, FMT_PRECISION);
  return Math.round(x * m) / m;
}

/**
 * Format a complex eigenvalue for display: "3.2 + 1.1i", "3.2", or "-0.5i".
 */
export function formatComplex(re: number, im: number): string {
  const r = roundForDisplay(re);
  const i = roundForDisplay(im);
  if (Math.abs(i) < FMT_EPS) return String(r);
  const imPart = Math.abs(i) === 1 ? (i < 0 ? "-i" : "i") : `${i}i`;
  if (Math.abs(r) < FMT_EPS) return imPart;
  const sign = i >= 0 ? " + " : " - ";
  return `${r}${sign}${Math.abs(i) === 1 ? "i" : `${Math.abs(i)}i`}`;
}
