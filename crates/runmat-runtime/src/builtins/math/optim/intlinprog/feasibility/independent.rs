//! Exact independent bound/integer projection when there are no linear rows.
//! This operates on the GA proposal, never on an LP relaxation or its status.
#[derive(Clone, Copy)]
pub(super) enum Refusal {
    Infeasible,
    Unrepresentable,
}
pub(super) fn coordinate(
    value: f64,
    lower: f64,
    upper: f64,
    integer: bool,
) -> Result<f64, Refusal> {
    if !value.is_finite() || lower.is_nan() || upper.is_nan() {
        return Err(Refusal::Unrepresentable);
    }
    if lower > upper || lower == f64::INFINITY || upper == f64::NEG_INFINITY {
        return Err(Refusal::Infeasible);
    }
    let (lower, upper) = if integer {
        (lower.ceil(), upper.floor())
    } else {
        (lower, upper)
    };
    if lower > upper {
        return Err(Refusal::Infeasible);
    }
    if value >= lower && value <= upper && (!integer || value.fract() == 0.0) {
        return Ok(value);
    }
    let candidate = if integer { value.round() } else { value };
    let candidate = candidate.max(lower).min(upper);
    if !candidate.is_finite()
        || candidate < lower
        || candidate > upper
        || (integer && candidate.fract() != 0.0)
    {
        Err(Refusal::Unrepresentable)
    } else {
        Ok(candidate)
    }
}

use super::{invalid, Repair, ScalarCooperation};
use crate::BuiltinResult;
pub(super) struct Domain {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub integers: Vec<usize>,
}
impl Domain {
    pub async fn repair(
        &self,
        proposal: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Repair> {
        // Independent domains have an exact coordinate-wise projection.
        // Avoid any rational scratch/factorization for the ordinary
        // unconstrained/bounds-only population, and never call this a
        // rounded relaxation witness.
        let (lower, upper) = (&self.lower, &self.upper);
        let mut changed = false;
        let mut integer_cursor = 0;
        for (column, &value) in proposal.iter().enumerate() {
            let integer = self.integers.get(integer_cursor) == Some(&column);
            if integer {
                integer_cursor += 1;
            }
            match coordinate(value, lower[column], upper[column], integer) {
                Ok(next) => changed |= next.to_bits() != value.to_bits(),
                Err(Refusal::Infeasible) => return Ok(Repair::Infeasible),
                Err(Refusal::Unrepresentable) => return Ok(Repair::Unverified),
            }
            cooperation.completed(1).await?;
        }
        if !changed {
            return Ok(Repair::Unchanged);
        }
        let mut values =
            runmat_value::exact_capacity_vec(self.lower.len()).map_err(super::super::allocation)?;
        let mut integer_cursor = 0;
        for (column, &value) in proposal.iter().enumerate() {
            let integer = self.integers.get(integer_cursor) == Some(&column);
            if integer {
                integer_cursor += 1;
            }
            let next = coordinate(value, lower[column], upper[column], integer).map_err(|_| {
                invalid("independent feasibility generation changed during publication")
            })?;
            values.push(next);
            cooperation.completed(1).await?;
        }
        Ok(Repair::Point(values))
    }
}
