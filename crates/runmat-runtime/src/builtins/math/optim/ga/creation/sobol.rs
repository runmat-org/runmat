//! Sobol creation uses the pinned Gray-code/Joe–Kuo implementation. Only
//! this operator has its direction-table dimension and sequence-length limits.
use super::super::{invalid, ScalarCooperation};
use crate::BuiltinResult;
use sobol::{params::JoeKuoD6, Sobol};

pub(in super::super) async fn fill(
    draft: &mut super::Draft,
    ranges: &[(f64, f64)],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let variables = draft.variables();
    if variables == 0 || variables > 21_201 || ranges.len() != variables {
        return Err(invalid(
            "Sobol creation requires between 1 and 21201 variables",
        ));
    }
    for &(lower, upper) in ranges {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(invalid("Sobol creation requires finite ordered ranges"));
        }
        cooperation.completed(1).await?;
    }
    // The numerical dependency owns its direction table and iterator scratch;
    // it does not expose an allocator receipt or a finite workspace quote.
    crate::context::host_allocation::require_unbounded_workspace(
        cooperation.runtime(),
        "ga",
        "finite host-memory execution is unsupported for Sobol direction tables and iterator workspaces",
    )?;
    cooperation.completed(0).await?;
    let parameters = if variables <= 100 {
        JoeKuoD6::minimal()
    } else if variables <= 1_000 {
        JoeKuoD6::standard()
    } else {
        JoeKuoD6::extended()
    };
    let mut sequence = Sobol::<f64>::new(variables, &parameters);
    let missing = draft.missing();
    if missing.len() as u128 > u128::from(sequence.max_len) {
        return Err(invalid(
            "Sobol creation population exceeds its sequence length",
        ));
    }
    cooperation.completed(variables as u64).await?;
    // Start at the canonical zero point. No scramble, arbitrary skip or
    // uniform fallback is substituted for a different sequence contract.
    for row in missing {
        let mut point = sequence
            .next()
            .ok_or_else(|| invalid("Sobol creation sequence exhausted"))?;
        for (column, value) in point.iter_mut().enumerate() {
            let (lower, upper) = ranges[column];
            *value = super::super::operators::crossover::interpolate(lower, upper, *value)
                .ok_or_else(|| invalid("Sobol creation coordinate cannot be represented"))?;
            cooperation.completed(1).await?;
        }
        draft.set(row, &point, false, cooperation).await?;
    }
    Ok(())
}
