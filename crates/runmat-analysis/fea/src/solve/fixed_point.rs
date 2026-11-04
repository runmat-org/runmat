//! Bounded, physics-independent Anderson depth-one fixed-point proposals.
//!
//! A proposal is not an accepted solution. The caller retains ownership of its
//! original equations, admissible set, convergence gate, and iteration budget.

use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory};

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AndersonOnePolicy {
    pub maximum_absolute_coefficient: f64,
}

impl Default for AndersonOnePolicy {
    fn default() -> Self {
        Self {
            maximum_absolute_coefficient: 32.0,
        }
    }
}

impl AndersonOnePolicy {
    pub fn validate(self) -> Result<(), FixedPointError> {
        if !self.maximum_absolute_coefficient.is_finite()
            || self.maximum_absolute_coefficient <= 0.0
        {
            return Err(FixedPointError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AndersonOneProposal {
    pub coefficient: f64,
    pub image: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FixedPointError {
    #[error("Anderson depth-one numerical accounting failed: {0}")]
    Accounting(#[from] LedgerError),
    #[error("invalid Anderson depth-one coefficient policy")]
    InvalidPolicy,
    #[error("invalid Anderson depth-one vector inventory")]
    InvalidInventory,
    #[error("nonfinite Anderson depth-one arithmetic")]
    NonfiniteArithmetic,
    #[error("fixed-point proposal cancelled")]
    Cancelled,
}

/// For residuals `r = g(x) - x`, propose
/// `g_new - gamma * (g_new - g_previous)`, where
/// `gamma = <r_new, r_new-r_previous> / ||r_new-r_previous||²`.
///
/// Empty or identical residual differences have no proposal. No dimensional
/// floor repairs a degenerate denominator, and an excessive coefficient is
/// declined rather than clipped. Cancellation is an explicit sampled input;
/// this numerical helper never calls a runtime or solver callback.
pub fn propose(
    previous_image: &[f64],
    previous_residual: &[f64],
    image: &[f64],
    residual: &[f64],
    policy: AndersonOnePolicy,
    cancelled: bool,
) -> Result<Option<AndersonOneProposal>, FixedPointError> {
    if cancelled {
        return Err(FixedPointError::Cancelled);
    }
    policy.validate()?;
    let n = image.len();
    if previous_image.len() != n
        || previous_residual.len() != n
        || residual.len() != n
        || previous_image
            .iter()
            .chain(previous_residual)
            .chain(image)
            .chain(residual)
            .any(|v| !v.is_finite())
    {
        return Err(FixedPointError::InvalidInventory);
    }
    if n == 0 {
        return Ok(None);
    }
    let length = u64::try_from(n).map_err(|_| LedgerError::CountOverflow)?;
    // The reference schedule is n residual subtractions, two n-term dots
    // (4n-2), one coefficient division when the denominator is nonzero, and
    // 3n affine image operations only when that coefficient is admissible.
    // Each completed logical batch is settled before the next gate; neither
    // a degenerate denominator nor a declined coefficient charges later work.
    let difference = crate::algebra::capacity::measured_sync(
        UsageCategory::Solve,
        OperationCount::from_operations(length),
        || -> Result<Vec<f64>, FixedPointError> {
            Ok(residual
                .iter()
                .zip(previous_residual)
                .map(|(new, previous)| new - previous)
                .collect())
        },
    )?;
    if difference.iter().any(|v| !v.is_finite()) {
        return Err(FixedPointError::NonfiniteArithmetic);
    }
    let dot_work = OperationCount::direct_real_dot(length)
        .map_err(LedgerError::from)?
        .checked_mul(2)
        .map_err(LedgerError::from)?;
    let (denominator, numerator) = crate::algebra::capacity::measured_sync(
        UsageCategory::Solve,
        dot_work,
        || -> Result<_, FixedPointError> {
            Ok((
                crate::algebra::dot_canonical(&difference, &difference),
                crate::algebra::dot_canonical(residual, &difference),
            ))
        },
    )?;
    if !denominator.is_finite() || !numerator.is_finite() || denominator < 0.0 {
        return Err(FixedPointError::NonfiniteArithmetic);
    }
    if denominator == 0.0 {
        return Ok(None);
    }
    let coefficient = crate::algebra::capacity::measured_sync(
        UsageCategory::Solve,
        OperationCount::from_operations(1),
        || -> Result<_, FixedPointError> { Ok(numerator / denominator) },
    )?;
    if !coefficient.is_finite() {
        return Err(FixedPointError::NonfiniteArithmetic);
    }
    if coefficient.abs() > policy.maximum_absolute_coefficient {
        return Ok(None);
    }
    let image_work = OperationCount::from_operations(length)
        .checked_mul(3)
        .map_err(LedgerError::from)?;
    let image = crate::algebra::capacity::measured_sync(
        UsageCategory::Solve,
        image_work,
        || -> Result<Vec<f64>, FixedPointError> {
            Ok(image
                .iter()
                .zip(previous_image)
                .map(|(new, previous)| new - coefficient * (new - previous))
                .collect())
        },
    )?;
    if image.iter().any(|v| !v.is_finite()) {
        return Err(FixedPointError::NonfiniteArithmetic);
    }
    Ok(Some(AndersonOneProposal { coefficient, image }))
}

#[cfg(test)]
mod tests;
