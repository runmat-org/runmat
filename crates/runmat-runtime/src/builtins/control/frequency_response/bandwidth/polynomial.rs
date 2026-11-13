//! Exact root count and first-root bracket for the represented gain equation.
//!
//! Every input scalar is a finite binary64 value. Converting those values to
//! rationals before polynomial construction makes the crossing coefficients
//! exact for the *represented* model and drop ratio. A Sturm sequence then
//! distinguishes no positive root from an unresolved numerical bracket;
//! neither condition is inferred from a floating grid or companion eigenvalue.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

use crate::builtins::control::tf_model::control_error;
use crate::context::host_allocation::HostAllocationReceipt;
use crate::context::RuntimeContext;
use crate::BuiltinResult;

use super::super::{admission, FrequencyApi};

const NAME: &str = "bandwidth";

pub(super) fn uncertain(message: &'static str) -> crate::RuntimeError {
    control_error(NAME, "RunMat:bandwidth:UnsupportedModel", message)
}

pub(super) fn checkpoint(runtime: Option<&RuntimeContext>) -> BuiltinResult<()> {
    HostAllocationReceipt::policy(runtime)
        .map_err(|error| admission(FrequencyApi::Bandwidth, runtime, error))?;
    Ok(())
}

pub(super) fn binary(value: f64) -> BuiltinResult<BigRational> {
    BigRational::from_float(value).ok_or_else(|| uncertain("bandwidth: coefficient is not finite"))
}

#[derive(Clone, Debug)]
pub(super) struct RootBracket {
    pub(super) lower: BigRational,
    pub(super) upper: BigRational,
}

pub(super) struct CrossingPolynomial {
    // Ascending powers. Their rational values include every represented
    // binary64 coefficient/conversion exactly, not a rounded floating proxy.
    coefficients: Vec<BigRational>,
}

impl CrossingPolynomial {
    pub(super) fn new(mut coefficients: Vec<BigRational>) -> BuiltinResult<Self> {
        trim(&mut coefficients);
        if coefficients.is_empty() || coefficients[0] <= BigRational::zero() {
            return Err(uncertain(
                "bandwidth: DC crossing equation is indeterminate",
            ));
        }
        Ok(Self { coefficients })
    }

    pub(super) fn sign_at(
        &self,
        point: &BigRational,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<i8> {
        Ok(sign(&evaluate(&self.coefficients, point, runtime)?))
    }

    /// None proves that the represented gain never falls below its DC
    /// threshold. Repeated contacts and removable common factors are skipped
    /// unless the sign on their right is negative.
    pub(super) fn first_positive_root(
        &self,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<Option<RootBracket>> {
        checkpoint(runtime)?;
        if self.coefficients.len() == 1 {
            return Ok(None);
        }
        if self.coefficients.len() == 2 {
            let root = -&self.coefficients[0] / &self.coefficients[1];
            return Ok((root > BigRational::zero()).then(|| RootBracket {
                lower: root.clone(),
                upper: root,
            }));
        }
        if self.coefficients.len() == 3 {
            return self.first_quadratic_root(runtime);
        }
        let square_free = square_free(&self.coefficients, runtime)?;
        let sequence = sturm(&square_free, runtime)?;
        let at_zero = variations(&sequence, Some(&BigRational::zero()), runtime)?;
        let at_infinity = variations(&sequence, None, runtime)?;
        if at_zero < at_infinity {
            return Err(uncertain("bandwidth: crossing root count is inconsistent"));
        }
        if at_zero == at_infinity {
            return Ok(None);
        }

        let total = at_zero - at_infinity;
        let mut lower = BigRational::zero();
        for ordinal in 1..=total {
            checkpoint(runtime)?;
            let bracket = isolate_root(&sequence, at_zero, ordinal, lower, runtime)?;
            if right_sign(self, &sequence, at_zero, ordinal, &bracket, runtime)? < 0 {
                return Ok(Some(bracket));
            }
            lower = bracket.upper;
        }
        Ok(None)
    }

    // With positive DC, a downward quadratic crossing either is the only
    // positive root (negative leading coefficient), or precedes the exact
    // positive vertex (positive leading coefficient). These signs prove the
    // first-root bracket without constructing a Sturm sequence.
    fn first_quadratic_root(
        &self,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<Option<RootBracket>> {
        let linear = &self.coefficients[1];
        let quadratic = &self.coefficients[2];
        if quadratic.is_positive() && !linear.is_negative() {
            return Ok(None);
        }
        let mut upper = if quadratic.is_positive() {
            let vertex = -linear / (quadratic + quadratic);
            match self.sign_at(&vertex, runtime)? {
                1 => return Ok(None),
                0 => return Ok(None),
                _ => vertex,
            }
        } else {
            let mut candidate = BigRational::from_integer(1.into());
            loop {
                checkpoint(runtime)?;
                if self.sign_at(&candidate, runtime)? <= 0 {
                    break candidate;
                }
                let doubled = &candidate + &candidate;
                if !doubled.to_f64().is_some_and(f64::is_finite) {
                    let maximum = binary(f64::MAX)?;
                    if self.sign_at(&maximum, runtime)? <= 0 {
                        break maximum;
                    }
                    return Err(uncertain(
                        "bandwidth: first crossing lies outside finite frequency range",
                    ));
                }
                candidate = doubled;
            }
        };
        let mut lower = BigRational::zero();
        for _ in 0..=2048 {
            checkpoint(runtime)?;
            let lower_f = lower.to_f64().unwrap_or(0.0);
            let upper_f = upper.to_f64().ok_or_else(|| {
                uncertain("bandwidth: crossing bracket exceeds finite frequency range")
            })?;
            if upper_f > 0.0 && (upper_f == lower_f || upper_f - lower_f <= upper_f * 1.0e-12) {
                return Ok(Some(RootBracket { lower, upper }));
            }
            let midpoint = (&lower + &upper) / BigInt::from(2);
            if self.sign_at(&midpoint, runtime)? <= 0 {
                upper = midpoint;
            } else {
                lower = midpoint;
            }
        }
        Err(uncertain(
            "bandwidth: first crossing cannot be isolated to representable precision",
        ))
    }
}

fn square_free(
    coefficients: &[BigRational],
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<BigRational>> {
    let mut left = coefficients.to_vec();
    let mut right = derivative(coefficients, runtime)?;
    while !right.is_empty() {
        checkpoint(runtime)?;
        let remainder = negative_remainder(&left, &right, runtime)?;
        left = right;
        right = remainder;
    }
    divide_exact(coefficients, &left, runtime)
}

fn derivative(
    coefficients: &[BigRational],
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<BigRational>> {
    let mut result = Vec::with_capacity(coefficients.len().saturating_sub(1));
    for (power, value) in coefficients.iter().enumerate().skip(1) {
        if power % 128 == 0 {
            checkpoint(runtime)?;
        }
        result.push(value * BigRational::from_integer(power.into()));
    }
    Ok(result)
}

fn divide_exact(
    dividend: &[BigRational],
    divisor: &[BigRational],
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<BigRational>> {
    let mut remainder = dividend.to_vec();
    let mut quotient = vec![BigRational::zero(); dividend.len() - divisor.len() + 1];
    let leading = divisor.last().expect("nonempty polynomial gcd");
    while remainder.len() >= divisor.len() {
        checkpoint(runtime)?;
        let offset = remainder.len() - divisor.len();
        let coefficient = remainder.last().expect("nonempty remainder") / leading;
        quotient[offset] = coefficient.clone();
        for (index, value) in divisor.iter().enumerate() {
            if index % 128 == 0 {
                checkpoint(runtime)?;
            }
            remainder[offset + index] -= &coefficient * value;
        }
        trim(&mut remainder);
    }
    if !remainder.is_empty() {
        return Err(uncertain("bandwidth: exact gain factorization failed"));
    }
    trim(&mut quotient);
    Ok(quotient)
}

fn isolate_root(
    sequence: &[Vec<BigRational>],
    at_zero: usize,
    ordinal: usize,
    mut lower: BigRational,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<RootBracket> {
    let mut upper = if lower.is_zero() {
        BigRational::from_integer(1.into())
    } else {
        &lower + &lower
    };
    let mut bracketed = false;
    for _ in 0..=1024 {
        checkpoint(runtime)?;
        if roots_through(sequence, at_zero, &upper, runtime)? >= ordinal {
            bracketed = true;
            break;
        }
        let doubled = &upper + &upper;
        if !doubled.to_f64().is_some_and(f64::is_finite) {
            let maximum = binary(f64::MAX)?;
            if roots_through(sequence, at_zero, &maximum, runtime)? >= ordinal {
                upper = maximum;
                bracketed = true;
                break;
            }
            return Err(uncertain(
                "bandwidth: crossing lies outside finite frequency range",
            ));
        }
        upper = doubled;
    }
    if !bracketed {
        return Err(uncertain("bandwidth: crossing could not be bracketed"));
    }
    for _ in 0..=2048 {
        checkpoint(runtime)?;
        let lower_f = lower.to_f64().unwrap_or(0.0);
        let upper_f = upper
            .to_f64()
            .ok_or_else(|| uncertain("bandwidth: crossing exceeds finite frequency range"))?;
        if upper_f > 0.0
            && (upper_f == lower_f || upper_f - lower_f <= upper_f * 1.0e-12)
            && roots_through(sequence, at_zero, &upper, runtime)? == ordinal
        {
            return Ok(RootBracket { lower, upper });
        }
        let midpoint = (&lower + &upper) / BigInt::from(2);
        if roots_through(sequence, at_zero, &midpoint, runtime)? >= ordinal {
            upper = midpoint;
        } else {
            lower = midpoint;
        }
    }
    Err(uncertain(
        "bandwidth: crossing cannot be isolated to representable precision",
    ))
}

fn right_sign(
    polynomial: &CrossingPolynomial,
    sequence: &[Vec<BigRational>],
    at_zero: usize,
    ordinal: usize,
    bracket: &RootBracket,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<i8> {
    let sign = polynomial.sign_at(&bracket.upper, runtime)?;
    if sign != 0 {
        return Ok(sign);
    }
    let mut step = if bracket.upper == bracket.lower {
        BigRational::from_integer(1.into())
    } else {
        &bracket.upper - &bracket.lower
    };
    for _ in 0..=2048 {
        checkpoint(runtime)?;
        let sample = &bracket.upper + &step;
        if roots_through(sequence, at_zero, &sample, runtime)? == ordinal {
            let sign = polynomial.sign_at(&sample, runtime)?;
            if sign != 0 {
                return Ok(sign);
            }
        }
        step /= BigInt::from(2);
    }
    Err(uncertain(
        "bandwidth: repeated contact cannot be separated from the next root",
    ))
}

fn roots_through(
    sequence: &[Vec<BigRational>],
    at_zero: usize,
    point: &BigRational,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<usize> {
    let at_point = variations(sequence, Some(point), runtime)?;
    at_zero
        .checked_sub(at_point)
        .ok_or_else(|| uncertain("bandwidth: crossing root count is inconsistent"))
}

fn sturm(
    coefficients: &[BigRational],
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<Vec<BigRational>>> {
    let mut sequence = vec![coefficients.to_vec()];
    let derivative = derivative(coefficients, runtime)?;
    sequence.push(derivative);
    while sequence
        .last()
        .is_some_and(|polynomial| polynomial.len() > 1)
    {
        checkpoint(runtime)?;
        let length = sequence.len();
        let next = negative_remainder(&sequence[length - 2], &sequence[length - 1], runtime)?;
        if next.is_empty() {
            return Err(uncertain(
                "bandwidth: square-free root count is inconsistent",
            ));
        }
        sequence.push(next);
    }
    Ok(sequence)
}

fn negative_remainder(
    dividend: &[BigRational],
    divisor: &[BigRational],
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<BigRational>> {
    let mut remainder = dividend.to_vec();
    let leading = divisor
        .last()
        .ok_or_else(|| uncertain("bandwidth: invalid crossing derivative"))?;
    while remainder.len() >= divisor.len() {
        checkpoint(runtime)?;
        let offset = remainder.len() - divisor.len();
        let quotient = remainder.last().expect("nonempty remainder") / leading;
        for (index, coefficient) in divisor.iter().enumerate() {
            if index % 128 == 0 {
                checkpoint(runtime)?;
            }
            remainder[offset + index] -= &quotient * coefficient;
        }
        trim(&mut remainder);
    }
    for coefficient in &mut remainder {
        *coefficient = -coefficient.clone();
    }
    // Positive rescaling preserves every Sturm sign variation while keeping
    // the next exact division from inheriting an arbitrary leading fraction.
    if let Some(leading) = remainder.last().map(Signed::abs) {
        for coefficient in &mut remainder {
            *coefficient /= &leading;
        }
    }
    Ok(remainder)
}

fn trim(coefficients: &mut Vec<BigRational>) {
    while coefficients.last().is_some_and(Zero::is_zero) {
        coefficients.pop();
    }
}

fn evaluate(
    coefficients: &[BigRational],
    point: &BigRational,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<BigRational> {
    let mut result = BigRational::zero();
    for (index, coefficient) in coefficients.iter().rev().enumerate() {
        if index % 128 == 0 {
            checkpoint(runtime)?;
        }
        result = result * point + coefficient;
    }
    Ok(result)
}

fn variations(
    sequence: &[Vec<BigRational>],
    point: Option<&BigRational>,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<usize> {
    let mut previous = 0;
    let mut changes = 0;
    for (index, polynomial) in sequence.iter().enumerate() {
        if index % 16 == 0 {
            checkpoint(runtime)?;
        }
        let current = match point {
            Some(point) => sign(&evaluate(polynomial, point, runtime)?),
            None => sign(polynomial.last().expect("nonempty Sturm polynomial")),
        };
        if current != 0 {
            if previous != 0 && previous != current {
                changes += 1;
            }
            previous = current;
        }
    }
    Ok(changes)
}

fn sign(value: &BigRational) -> i8 {
    if value.is_positive() {
        1
    } else if value.is_negative() {
        -1
    } else {
        0
    }
}
