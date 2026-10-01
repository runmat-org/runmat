//! Residual and power acceptance share one finite authored Krylov budget.
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
    let prepared = reduced::PreparedReduced::new(assembly, policy)?;
    let mut initial = None;
    let mut tolerance = policy.relative_tolerance;
    let mut iterations = 0_u32;
    loop {
        let remaining = policy
            .maximum_iterations
            .checked_sub(iterations as usize)
            .filter(|remaining| *remaining > 0)
            .ok_or(AcousticHarmonicError::NotConverged)?;
        let mut solution = prepared.solve(
            assembly,
            AcousticHarmonicPolicy {
                relative_tolerance: tolerance,
                maximum_iterations: remaining,
                ..policy
            },
            initial.as_deref(),
        )?;
        iterations = iterations
            .checked_add(solution.iterations)
            .ok_or(AcousticHarmonicError::NotConverged)?;
        let closure = power::evaluate(assembly, &solution.real, &solution.imaginary)?;
        if closure.power_relative_error <= power_tolerance {
            solution.iterations = iterations;
            return Ok((solution, closure));
        }
        // A smaller linear residual is a refinement request, not a certificate of
        // power closure. Only reevaluating the original assembled physical balance
        // accepts the next iterate. No mesh, matrix, load or authored bound changes.
        let next = tolerance.min(solution.relative_residual)
            * 0.5
            * (power_tolerance / closure.power_relative_error).min(1.0);
        if solution.iterations == 0
            || iterations as usize >= policy.maximum_iterations
            || !next.is_finite()
            || next <= 0.0
            || next >= tolerance
        {
            return Err(AcousticHarmonicError::PowerNotConverged {
                iterations,
                relative_residual: solution.relative_residual,
                power_relative_error: closure.power_relative_error,
                power_tolerance,
            });
        }
        tolerance = next;
        // Drop full-pressure and power workspaces before restarting. Retain only
        // the reduced coordinates; the shared GMRES implementation owns warm starts.
        initial = Some(solution.coordinates);
    }
}

#[cfg(test)]
mod tests;
