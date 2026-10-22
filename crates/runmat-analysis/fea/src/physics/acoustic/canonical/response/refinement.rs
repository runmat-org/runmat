//! One certified direct solve, accepted only when the original assembled power balance closes.
use super::{power, reduced, AcousticAssembly, AcousticHarmonicError, AcousticHarmonicPolicy};

pub(super) fn solve(
    assembly: &AcousticAssembly,
    policy: AcousticHarmonicPolicy,
    power_tolerance: f64,
) -> Result<(reduced::ReducedSolution, power::PowerClosure), AcousticHarmonicError> {
    if !power_tolerance.is_finite() || power_tolerance <= 0.0 || power_tolerance >= 1.0 {
        return Err(AcousticHarmonicError::InvalidInput(
            "power closure tolerance".into(),
        ));
    }
    let solution = reduced::PreparedReduced::new(assembly, policy)?.solve(assembly, policy)?;
    // The refined residual is necessary, not sufficient: the original K/M/B power balance
    // must close too. A direct solve has no further refinement to request.
    let closure = power::evaluate(assembly, &solution.real, &solution.imaginary)?;
    if closure.power_relative_error > power_tolerance {
        return Err(AcousticHarmonicError::PowerNotConverged {
            iterations: solution.iterations,
            relative_residual: solution.relative_residual,
            power_relative_error: closure.power_relative_error,
            power_tolerance,
        });
    }
    Ok((solution, closure))
}

#[cfg(test)]
mod tests;
