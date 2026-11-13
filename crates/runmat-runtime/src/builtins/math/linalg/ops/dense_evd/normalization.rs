//! Homogeneous left/right pairing without absolute-smallness rejection.
//! Zero, numerically unusable or unrepresentable pairings retain kernel vectors.
use num_complex::Complex64;

pub(super) enum Metric<'a> {
    Identity,
    Pencil {
        mass: &'a [Complex64],
        beta: &'a [Complex64],
    },
}

fn scale(value: Complex64, exponent: i32) -> Complex64 {
    Complex64::new(
        libm::scalbn(value.re, exponent),
        libm::scalbn(value.im, exponent),
    )
}
fn decompose(value: Complex64) -> Option<(Complex64, i32)> {
    let largest = value.re.abs().max(value.im.abs());
    if largest == 0. || !largest.is_finite() {
        return None;
    }
    let (_, exponent) = libm::frexp(largest);
    Some((scale(value, -exponent), exponent))
}

/// A compensated, exponent-aligned sum of signed scalar products. Real and
/// imaginary components never share a factor scale: a tiny imaginary input
/// can remain the whole pairing after large real terms cancel. This is bounded
/// floating-point compensation, not a universal exact-arithmetic accumulator.
#[derive(Default)]
struct RealSum {
    value: f64,
    correction: f64,
    exponent: Option<i32>,
}
impl RealSum {
    fn product(&mut self, factors: &[f64], sign: f64) {
        let mut term = sign;
        let mut exponent = 0;
        for &factor in factors {
            if factor == 0. {
                return;
            }
            let (part, power) = libm::frexp(factor);
            term *= part;
            exponent += power;
        }
        let current = match self.exponent {
            Some(current) if self.value != 0. || self.correction != 0. => current,
            _ => {
                self.exponent = Some(exponent);
                exponent
            }
        };
        if exponent > current {
            self.value = libm::scalbn(self.value, current - exponent);
            self.correction = libm::scalbn(self.correction, current - exponent);
            self.exponent = Some(exponent);
        } else {
            term = libm::scalbn(term, exponent - current);
        }
        let adjusted = term - self.correction;
        let next = self.value + adjusted;
        self.correction = (next - self.value) - adjusted;
        self.value = next;
    }
    fn finish(self) -> Option<(f64, i32)> {
        let value = self.value - self.correction;
        if value == 0. || !value.is_finite() {
            return None;
        }
        let (part, power) = libm::frexp(value);
        Some((part, self.exponent? + power))
    }
}
#[derive(Default)]
struct Dot {
    real: RealSum,
    imag: RealSum,
    invalid: bool,
}
impl Dot {
    fn product2(&mut self, a: Complex64, b: Complex64) {
        if !finite(a) || !finite(b) {
            self.invalid = true;
            return;
        }
        self.real.product(&[a.re, b.re], 1.);
        self.real.product(&[a.im, b.im], -1.);
        self.imag.product(&[a.re, b.im], 1.);
        self.imag.product(&[a.im, b.re], 1.);
    }
    fn product3(&mut self, a: Complex64, b: Complex64, c: Complex64) {
        if !finite(a) || !finite(b) || !finite(c) {
            self.invalid = true;
            return;
        }
        self.real.product(&[a.re, b.re, c.re], 1.);
        self.real.product(&[a.im, b.im, c.re], -1.);
        self.real.product(&[a.re, b.im, c.im], -1.);
        self.real.product(&[a.im, b.re, c.im], -1.);
        self.imag.product(&[a.re, b.re, c.im], 1.);
        self.imag.product(&[a.im, b.im, c.im], -1.);
        self.imag.product(&[a.re, b.im, c.re], 1.);
        self.imag.product(&[a.im, b.re, c.re], 1.);
    }
    fn finish(self) -> Option<Pairing> {
        if self.invalid {
            return None;
        }
        let real = self.real.finish();
        let imag = self.imag.finish();
        if real.is_none() && imag.is_none() {
            return None;
        }
        Some(Pairing { real, imag })
    }
}
fn finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}
struct Pairing {
    real: Option<(f64, i32)>,
    imag: Option<(f64, i32)>,
}
impl Pairing {
    fn project(&self, value: Complex64) -> Complex64 {
        let materialize = |component: Option<(f64, i32)>| {
            component.map_or(0., |(part, power)| libm::scalbn(part, power))
        };
        let real = materialize(self.real);
        let imag = materialize(self.imag);
        if real.is_finite() && imag.is_finite() && (real != 0. || imag != 0.) {
            // Preserve each original numerator component; the canonical value
            // divider already handles adversarial mixed component scales.
            let (re, im) = runmat_value::divide_complex_f64((value.re, value.im), (real, -imag));
            return Complex64::new(re, im);
        }
        if value == Complex64::new(0., 0.) {
            return value;
        }
        let exponent = self
            .real
            .into_iter()
            .chain(self.imag)
            .map(|(_, power)| power)
            .max()
            .expect("nonzero pairing");
        let component = |value: Option<(f64, i32)>| {
            value.map_or(0., |(part, power)| libm::scalbn(part, power - exponent))
        };
        let direction = Complex64::new(component(self.real), component(self.imag));
        let Some((part, power)) = decompose(value) else {
            return Complex64::new(f64::NAN, f64::NAN);
        };
        let (re, im) =
            runmat_value::divide_complex_f64((part.re, part.im), (direction.re, -direction.im));
        scale(Complex64::new(re, im), power - exponent)
    }
}

pub(super) fn left(left: &mut [Complex64], right: &[Complex64], n: usize, metric: Metric<'_>) {
    for column in 0..n {
        if matches!(&metric, Metric::Pencil { beta, .. } if beta[column] == Complex64::new(0.,0.)) {
            // Infinite/indeterminate pairs have no W^H B V = 1 contract.
            continue;
        }
        let lo = column * n;
        let mut dot = Dot::default();
        for row in 0..n {
            let w = left[lo + row].conj();
            match &metric {
                Metric::Identity => dot.product2(w, right[lo + row]),
                Metric::Pencil { mass, .. } => {
                    for j in 0..n {
                        dot.product3(w, mass[row + j * n], right[lo + j]);
                    }
                }
            }
        }
        let Some(pairing) = dot.finish() else {
            continue;
        };
        let mut nonzero = false;
        let representable = left[lo..lo + n].iter().copied().all(|value| {
            let projected = pairing.project(value);
            nonzero |= projected != Complex64::new(0., 0.);
            finite(projected)
        });
        if !representable || !nonzero {
            continue;
        }
        for value in &mut left[lo..lo + n] {
            *value = pairing.project(*value);
        }
    }
}
