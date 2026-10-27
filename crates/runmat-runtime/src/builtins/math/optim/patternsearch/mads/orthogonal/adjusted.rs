//! Largest admissible integer rounding along an exact Halton ray.
use super::super::super::{buffer, invalid};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

async fn rounded(
    ray: &[BigRational],
    scale: &BigRational,
    target: &mut Vec<BigInt>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<BigInt> {
    target.clear();
    let mut norm = BigInt::zero();
    for value in ray {
        let scaled = value * scale;
        let magnitude =
            (scaled.abs() + BigRational::new(BigInt::one(), BigInt::from(2))).to_integer();
        norm += &magnitude * &magnitude;
        target.push(if scaled.is_negative() {
            -magnitude
        } else {
            magnitude
        });
        cooperation.completed(1).await?;
    }
    Ok(norm)
}

pub(super) async fn direction(
    ray: &[BigRational],
    level: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<BigInt>> {
    let mut maximum = BigRational::zero();
    for value in ray {
        maximum = maximum.max(value.abs());
        cooperation.completed(1).await?;
    }
    if maximum.is_zero() {
        return Err(invalid("OrthoMADS Halton direction is zero"));
    }
    let limit = BigInt::one() << level;
    let upper_coordinate = (BigInt::one() << level.div_ceil(2)) + BigInt::one();
    let mut upper = BigRational::from_integer(upper_coordinate) / maximum;
    let mut lower = BigRational::zero();
    let mut accepted = buffer(ray.len())?;
    accepted.resize(ray.len(), BigInt::zero());
    let mut proposal = buffer(ray.len())?;
    loop {
        // Exact next rounding threshold means termination certifies the LAST
        // feasible step, rather than accepting an arbitrary bisection iterate.
        let mut threshold: Option<BigRational> = None;
        for (value, current) in ray.iter().zip(&accepted) {
            if !value.is_zero() {
                let next = BigRational::new(current.abs() * 2 + BigInt::one(), BigInt::from(2))
                    / value.abs();
                threshold = Some(match threshold {
                    Some(prior) => prior.min(next),
                    None => next,
                });
            }
            cooperation.completed(1).await?;
        }
        let threshold =
            threshold.ok_or_else(|| invalid("OrthoMADS rounding direction is absent"))?;
        let next_norm = rounded(ray, &threshold, &mut proposal, cooperation).await?;
        if next_norm > limit {
            if accepted.iter().all(BigInt::is_zero) {
                return Err(invalid("OrthoMADS adjusted direction is zero"));
            }
            return Ok(accepted);
        }
        let midpoint = (&lower + &upper) / BigInt::from(2);
        if midpoint <= threshold {
            lower = threshold;
            std::mem::swap(&mut accepted, &mut proposal);
        } else {
            let norm = rounded(ray, &midpoint, &mut proposal, cooperation).await?;
            if norm <= limit {
                lower = midpoint;
                std::mem::swap(&mut accepted, &mut proposal);
            } else {
                upper = midpoint;
            }
        }
    }
}
