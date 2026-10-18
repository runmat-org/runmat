//! Exact represented-coefficient gain-drop equation.
//!
//! Continuous models use `x = w`. Discrete models use the Cayley coordinate
//! `x = tan(w * Ts / 2)`, so the full nonnegative Nyquist interval maps to
//! the positive real axis without a sampling grid. This owner is private to
//! bandwidth; it does not replace the canonical physical-z model evaluator.

use num_complex::Complex64;
use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::BuiltinResult;

use super::polynomial::{binary, checkpoint, uncertain, CrossingPolynomial};
use crate::context::RuntimeContext;

#[derive(Clone)]
struct ExactComplex {
    re: BigRational,
    im: BigRational,
}

impl ExactComplex {
    fn zero() -> Self {
        Self {
            re: BigRational::zero(),
            im: BigRational::zero(),
        }
    }

    fn one() -> Self {
        Self {
            re: BigRational::one(),
            im: BigRational::zero(),
        }
    }

    fn i() -> Self {
        Self {
            re: BigRational::zero(),
            im: BigRational::one(),
        }
    }

    fn from_binary(value: Complex64) -> BuiltinResult<Self> {
        Ok(Self {
            re: binary(value.re)?,
            im: binary(value.im)?,
        })
    }

    fn conjugate(&self) -> Self {
        Self {
            re: self.re.clone(),
            im: -&self.im,
        }
    }

    fn add_assign(&mut self, other: &Self) {
        self.re += &other.re;
        self.im += &other.im;
    }

    fn multiply(&self, other: &Self) -> Self {
        Self {
            re: &self.re * &other.re - &self.im * &other.im,
            im: &self.re * &other.im + &self.im * &other.re,
        }
    }

    fn norm_squared(&self) -> BigRational {
        &self.re * &self.re + &self.im * &self.im
    }
}

pub(super) fn equation(
    numerator: &[Complex64],
    denominator: &[Complex64],
    discrete: bool,
    drop_ratio: f64,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<CrossingPolynomial> {
    checkpoint(runtime)?;
    if numerator.is_empty()
        || denominator.is_empty()
        || !drop_ratio.is_finite()
        || drop_ratio <= 0.0
        || drop_ratio >= 1.0
    {
        return Err(uncertain(
            "bandwidth: crossing equation has no valid DC reference",
        ));
    }
    let degree = numerator.len().max(denominator.len()) - 1;
    let numerator = exact_coefficients(numerator, runtime)?;
    let denominator = exact_coefficients(denominator, runtime)?;
    let numerator = if discrete {
        cayley(&numerator, degree, runtime)?
    } else {
        continuous(&numerator, runtime)?
    };
    let denominator = if discrete {
        cayley(&denominator, degree, runtime)?
    } else {
        continuous(&denominator, runtime)?
    };
    let dc_n = numerator[0].norm_squared();
    let dc_d = denominator[0].norm_squared();
    if dc_n.is_zero() || dc_d.is_zero() {
        return Err(uncertain(
            "bandwidth: DC crossing equation is indeterminate",
        ));
    }
    let gain_n = squared_magnitude(&numerator, runtime)?;
    let gain_d = squared_magnitude(&denominator, runtime)?;
    let ratio = binary(drop_ratio)?;
    let drop_weight = &ratio * &ratio;
    let mut coefficients = Vec::with_capacity(gain_n.len().max(gain_d.len()));
    for index in 0..gain_n.len().max(gain_d.len()) {
        if index % 128 == 0 {
            checkpoint(runtime)?;
        }
        let n = gain_n.get(index).cloned().unwrap_or_else(BigRational::zero);
        let d = gain_d.get(index).cloned().unwrap_or_else(BigRational::zero);
        coefficients.push(n * &dc_d - &drop_weight * d * &dc_n);
    }
    CrossingPolynomial::new(coefficients)
}

fn exact_coefficients(
    source: &[Complex64],
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<ExactComplex>> {
    let mut result = Vec::with_capacity(source.len());
    for (index, &value) in source.iter().enumerate() {
        if index % 128 == 0 {
            checkpoint(runtime)?;
        }
        result.push(ExactComplex::from_binary(value)?);
    }
    Ok(result)
}

fn continuous(
    source: &[ExactComplex],
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<ExactComplex>> {
    let mut result = Vec::with_capacity(source.len());
    let mut power = ExactComplex::one();
    let i = ExactComplex::i();
    for (index, coefficient) in source.iter().rev().enumerate() {
        if index % 128 == 0 {
            checkpoint(runtime)?;
        }
        result.push(coefficient.multiply(&power));
        power = power.multiply(&i);
    }
    Ok(result)
}

fn times_linear(
    source: &[ExactComplex],
    slope: &ExactComplex,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<ExactComplex>> {
    let mut result = vec![ExactComplex::zero(); source.len() + 1];
    for (index, coefficient) in source.iter().enumerate() {
        if index % 128 == 0 {
            checkpoint(runtime)?;
        }
        result[index].add_assign(coefficient);
        result[index + 1].add_assign(&coefficient.multiply(slope));
    }
    Ok(result)
}

fn cayley(
    source: &[ExactComplex],
    degree: usize,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<ExactComplex>> {
    let i = ExactComplex::i();
    let minus_i = ExactComplex {
        re: BigRational::zero(),
        im: -BigRational::one(),
    };
    let padding = degree + 1 - source.len();
    let mut descending =
        std::iter::repeat_n(ExactComplex::zero(), padding).chain(source.iter().cloned());
    let mut result = vec![descending.next().expect("nonempty polynomial")];
    let mut denominator_power = vec![ExactComplex::one()];
    for coefficient in descending {
        checkpoint(runtime)?;
        result = times_linear(&result, &i, runtime)?;
        denominator_power = times_linear(&denominator_power, &minus_i, runtime)?;
        for (target, power) in result.iter_mut().zip(&denominator_power) {
            target.add_assign(&coefficient.multiply(power));
        }
    }
    Ok(result)
}

fn squared_magnitude(
    source: &[ExactComplex],
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<BigRational>> {
    let length = source
        .len()
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| uncertain("bandwidth: crossing polynomial size overflows"))?;
    let mut result = vec![BigRational::zero(); length];
    for left in 0..source.len() {
        checkpoint(runtime)?;
        for right in left..source.len() {
            if (right - left) % 128 == 0 {
                checkpoint(runtime)?;
            }
            let product = source[left].multiply(&source[right].conjugate());
            let term = if left == right {
                product.re
            } else {
                &product.re + &product.re
            };
            result[left + right] += term;
        }
    }
    Ok(result)
}
