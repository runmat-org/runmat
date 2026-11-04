//! Recovery of independent constraint-row multipliers from generalized reactions.

use crate::algebra::coercivity::NumericalCapacityError;
use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ConstraintMultiplierError {
    #[error(transparent)]
    Accounting(#[from] LedgerError),
    #[error("constraint reaction recovery cancelled")]
    Cancelled,
}

#[cfg(test)]
pub(crate) fn solve_constraint_multipliers(
    rows: &[BTreeMap<usize, f64>],
    right_hand_side: &[f64],
) -> Result<Option<Vec<f64>>, ConstraintMultiplierError> {
    use futures::FutureExt;
    let result = solve_constraint_multipliers_with_owner(rows, right_hand_side, |_work| {
        std::future::ready(Ok::<_, std::convert::Infallible>(()))
    })
    .now_or_never()
    .expect("native constraint recovery has an immediately ready owner");
    result.map_err(|error| match error {
        NumericalCapacityError::Numerical(error) => error,
        NumericalCapacityError::Admission(never) => match never {},
    })
}

pub(crate) async fn solve_constraint_multipliers_with_owner<O, F, A>(
    rows: &[BTreeMap<usize, f64>],
    right_hand_side: &[f64],
    mut before: O,
) -> Result<Option<Vec<f64>>, NumericalCapacityError<ConstraintMultiplierError, A>>
where
    O: FnMut(OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
{
    if rows.len() != right_hand_side.len() {
        return Ok(None);
    }
    if rows.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let load_scale = right_hand_side
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if !load_scale.is_finite() {
        return Ok(None);
    }
    let usage = UsageContext::capture();
    let mut inverse_diagonal = Vec::with_capacity(rows.len());
    for row in rows {
        let work = real_dot_work(row.len())?;
        let diagonal = measured(usage.as_ref(), &mut before, work, || {
            row.values().map(|value| value * value).sum::<f64>()
        })
        .await?;
        if diagonal <= 1.0e-18 || !diagonal.is_finite() {
            return Ok(None);
        }
        inverse_diagonal.push(
            measured(usage.as_ref(), &mut before, operations(1), || {
                diagonal.recip()
            })
            .await?,
        );
    }
    if load_scale == 0.0 {
        return Ok(Some(vec![0.0; rows.len()]));
    }
    let mut solution = vec![0.0; rows.len()];
    // Normalize the right-hand side so that the convergence and breakdown
    // checks are independent of reaction units or the overall force scale.
    let length = operations(
        u64::try_from(rows.len())
            .map_err(|_| ConstraintMultiplierError::Accounting(LedgerError::CountOverflow))?,
    );
    let dot_work = real_dot_work(rows.len())?;
    let norm_work = dot_work
        .checked_add(operations(1))
        .map_err(accounting_overflow)?;
    let mut residual = measured(usage.as_ref(), &mut before, length, || {
        right_hand_side
            .iter()
            .map(|value| value / load_scale)
            .collect::<Vec<_>>()
    })
    .await?;
    let scale = measured(usage.as_ref(), &mut before, norm_work, || {
        dot(&residual, &residual).sqrt()
    })
    .await?;
    let mut preconditioned = measured(usage.as_ref(), &mut before, length, || {
        residual
            .iter()
            .zip(&inverse_diagonal)
            .map(|(residual, inverse)| residual * inverse)
            .collect::<Vec<_>>()
    })
    .await?;
    let mut direction = preconditioned.clone();
    let mut rz = measured(usage.as_ref(), &mut before, dot_work, || {
        dot(&residual, &preconditioned)
    })
    .await?;
    let mut gram_work = OperationCount::ZERO;
    for row in rows {
        let scatter = operations(
            u64::try_from(row.len())
                .map_err(|_| accounting_overflow(runmat_simulation_usage::CountOverflow))?,
        )
        .checked_mul(2)
        .map_err(accounting_overflow)?;
        gram_work = gram_work
            .checked_add(scatter)
            .map_err(accounting_overflow)?
            .checked_add(real_dot_work(row.len())?)
            .map_err(accounting_overflow)?;
    }
    for _ in 0..rows.len().saturating_mul(4).max(256) {
        if crate::is_fea_cancelled() {
            return Err(ConstraintMultiplierError::Cancelled.into());
        }
        let applied = measured(usage.as_ref(), &mut before, gram_work, || {
            apply_constraint_gram(rows, &direction)
        })
        .await?;
        let denominator = measured(usage.as_ref(), &mut before, dot_work, || {
            dot(&direction, &applied)
        })
        .await?;
        // The Gram quadratic form is positive for an independent direction.
        // A fixed absolute cutoff rejects valid small reactions even when the
        // row coefficients are well scaled (for example, an affine patch with
        // support reactions near solver roundoff).
        if !denominator.is_finite() || denominator <= 0.0 {
            return Ok(None);
        }
        let alpha = measured(usage.as_ref(), &mut before, operations(1), || {
            rz / denominator
        })
        .await?;
        measured(
            usage.as_ref(),
            &mut before,
            length.checked_mul(4).map_err(accounting_overflow)?,
            || {
                for index in 0..rows.len() {
                    solution[index] += alpha * direction[index];
                    residual[index] -= alpha * applied[index];
                }
            },
        )
        .await?;
        let relative_residual = measured(
            usage.as_ref(),
            &mut before,
            norm_work
                .checked_add(operations(1))
                .map_err(accounting_overflow)?,
            || dot(&residual, &residual).sqrt() / scale,
        )
        .await?;
        if relative_residual <= 1.0e-10 {
            let solution = measured(usage.as_ref(), &mut before, length, || {
                solution
                    .into_iter()
                    .map(|value| value * load_scale)
                    .collect::<Vec<_>>()
            })
            .await?;
            return Ok(solution
                .iter()
                .all(|value| value.is_finite())
                .then_some(solution));
        }
        preconditioned = measured(usage.as_ref(), &mut before, length, || {
            residual
                .iter()
                .zip(&inverse_diagonal)
                .map(|(residual, inverse)| residual * inverse)
                .collect()
        })
        .await?;
        let next_rz = measured(usage.as_ref(), &mut before, dot_work, || {
            dot(&residual, &preconditioned)
        })
        .await?;
        let beta = measured(usage.as_ref(), &mut before, operations(1), || next_rz / rz).await?;
        measured(
            usage.as_ref(),
            &mut before,
            length.checked_mul(2).map_err(accounting_overflow)?,
            || {
                for (direction, preconditioned) in direction.iter_mut().zip(&preconditioned) {
                    *direction = preconditioned + beta * *direction;
                }
            },
        )
        .await?;
        rz = next_rz;
    }
    Ok(None)
}

fn operations(count: u64) -> OperationCount {
    OperationCount::from_operations(count)
}
fn accounting_overflow(_: runmat_simulation_usage::CountOverflow) -> ConstraintMultiplierError {
    LedgerError::CountOverflow.into()
}
fn real_dot_work(length: usize) -> Result<OperationCount, ConstraintMultiplierError> {
    OperationCount::direct_real_dot(
        u64::try_from(length)
            .map_err(|_| accounting_overflow(runmat_simulation_usage::CountOverflow))?,
    )
    .map_err(accounting_overflow)
}
async fn measured<T, O, F, A>(
    usage: Option<&UsageContext>,
    before: &mut O,
    work: OperationCount,
    run: impl FnOnce() -> T,
) -> Result<T, NumericalCapacityError<ConstraintMultiplierError, A>>
where
    O: FnMut(OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
{
    before(work)
        .await
        .map_err(NumericalCapacityError::Admission)?;
    let mut reservation = usage
        .map(|usage| usage.ledger().reserve(UsageCategory::Solve, work))
        .transpose()
        .map_err(ConstraintMultiplierError::from)?;
    let result = run();
    if let Some(reservation) = &mut reservation {
        reservation
            .complete(work)
            .map_err(ConstraintMultiplierError::from)?;
    }
    Ok(result)
}

fn apply_constraint_gram(rows: &[BTreeMap<usize, f64>], values: &[f64]) -> Vec<f64> {
    let mut dofs = BTreeMap::<usize, f64>::new();
    for (row, value) in rows.iter().zip(values) {
        for (dof, coefficient) in row {
            *dofs.entry(*dof).or_insert(0.0) += coefficient * value;
        }
    }
    rows.iter()
        .map(|row| {
            row.iter()
                .map(|(dof, coefficient)| coefficient * dofs.get(dof).copied().unwrap_or(0.0))
                .sum()
        })
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_independent_reactions_do_not_trigger_a_fixed_breakdown_cutoff() {
        let rows = vec![BTreeMap::from([(0, 1.0)]), BTreeMap::from([(1, 1.0)])];
        for magnitude in [1.0e-10, 1.0e-20] {
            let rhs = [4.0 * magnitude, -2.0 * magnitude];
            let multipliers = solve_constraint_multipliers(&rows, &rhs)
                .expect("constraint reaction accounting")
                .expect("positive small Gram denominator is not a breakdown");
            for (actual, expected) in multipliers.iter().zip(rhs) {
                assert!((actual - expected).abs() < magnitude * 1.0e-12);
            }
        }
    }
}
