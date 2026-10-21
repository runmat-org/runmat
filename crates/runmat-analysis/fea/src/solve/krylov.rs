//! Matrix-free Krylov authority shared by indefinite finite-element solve domains.


mod envelope;
pub(crate) mod right_ilu;
pub(crate) use envelope::GmresWorkEnvelope;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GmresPolicy {
    pub relative_tolerance: f64,
    pub restart: usize,
    pub maximum_iterations: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GmresResult {
    pub solution: Vec<f64>,
    pub iterations: u32,
    pub residual_norm: f64,
    pub relative_residual: Option<f64>,
    pub converged: bool,
    pub cancelled: bool,
    pub breakdown: bool,
}

/// Restarted GMRES on the shared sparse layer for operators supplied as closures (callers that
/// precondition compose `A M⁻¹` themselves). Operator failures propagate unchanged; cancellation
/// and breakdown are reported in the result.
pub(crate) fn restarted_gmres<E>(
    right_hand_side: &[f64],
    initial: &[f64],
    policy: GmresPolicy,
    apply: impl Fn(&[f64]) -> Result<Vec<f64>, E>,
    cancelled: impl Fn() -> bool,
) -> Result<Option<GmresResult>, E> {
    use runmat_sparse_solvers::krylov::{
        block_on, gmres_async, gmres_workspace_len, ApplyStatus, GmresOptions, KrylovOps,
        KrylovOptions, PreconditionerSide, StopFlag,
    };
    use runmat_sparse_solvers::{SolverError, SolverResult};
    let size = right_hand_side.len();
    if size == 0
        || initial.len() != size
        || GmresWorkEnvelope::derive(size, policy).is_err()
        || right_hand_side
            .iter()
            .chain(initial)
            .any(|value| !value.is_finite())
    {
        return Ok(None);
    }
    struct ClosureOps<'a, E, A: Fn(&[f64]) -> Result<Vec<f64>, E>, C: Fn() -> bool> {
        apply: &'a A,
        cancelled: &'a C,
        failure: Option<E>,
        observed_cancellation: bool,
    }
    impl<E, A: Fn(&[f64]) -> Result<Vec<f64>, E>, C: Fn() -> bool> KrylovOps<f64> for ClosureOps<'_, E, A, C> {
        async fn apply_operator(&mut self, x: &[f64], y: &mut [f64]) -> SolverResult<ApplyStatus> {
            match (self.apply)(x) {
                Ok(values) if values.len() == y.len() => {
                    y.copy_from_slice(&values);
                    Ok(ApplyStatus::Applied)
                }
                Ok(_) => Err(SolverError::Dimension("operator output length".into())),
                Err(error) => {
                    self.failure = Some(error);
                    Err(SolverError::Operator("operator application failed".into()))
                }
            }
        }
        async fn apply_preconditioner(&mut self, r: &[f64], z: &mut [f64]) -> SolverResult<ApplyStatus> {
            z.copy_from_slice(r);
            Ok(ApplyStatus::Applied)
        }
        fn has_preconditioner(&self) -> bool {
            false
        }
        async fn checkpoint(&mut self, _iteration: usize, _relative: f64) -> SolverResult<()> {
            if (self.cancelled)() {
                self.observed_cancellation = true;
                return Err(SolverError::Cancelled);
            }
            Ok(())
        }
    }
    let options = KrylovOptions {
        rtol: policy.relative_tolerance,
        max_iterations: policy.maximum_iterations,
        ..Default::default()
    };
    let gmres = GmresOptions {
        restart: policy.restart.min(size).max(1),
        side: PreconditionerSide::Right,
    };
    let mut work = vec![0.0; gmres_workspace_len(size, &options, &gmres)];
    let mut solution = initial.to_vec();
    let mut ops = ClosureOps {
        apply: &apply,
        cancelled: &cancelled,
        failure: None,
        observed_cancellation: false,
    };
    let outcome = block_on(gmres_async(&mut ops, right_hand_side, &mut solution, &options, &gmres, &mut work));
    if let Some(error) = ops.failure.take() {
        return Err(error);
    }
    let reference_norm = runmat_sparse_solvers::par::norm2(right_hand_side);
    let residual_of = |solution: &[f64]| -> Result<f64, E> {
        let applied = apply(solution)?;
        Ok(right_hand_side
            .iter()
            .zip(&applied)
            .map(|(b, a)| (b - a) * (b - a))
            .sum::<f64>()
            .sqrt())
    };
    let (iterations, converged, breakdown) = match &outcome {
        Ok(report) => (
            report.iterations,
            report.converged(),
            report.flag == StopFlag::Breakdown,
        ),
        Err(_) => (0, false, false),
    };
    let residual_norm = match &outcome {
        Ok(report) => report.residual_norm,
        Err(_) => residual_of(&solution)?,
    };
    if !residual_norm.is_finite() {
        return Ok(None);
    }
    let reference = if reference_norm > 0.0 { reference_norm } else { residual_norm };
    Ok(Some(GmresResult {
        solution,
        iterations: u32::try_from(iterations).unwrap_or(u32::MAX),
        residual_norm,
        relative_residual: crate::solve::convergence::relative_residual(residual_norm, reference),
        converged: converged && !ops.observed_cancellation,
        cancelled: ops.observed_cancellation,
        breakdown,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restarted_solver_handles_nonsymmetric_system_and_cancellation() {
        let apply = |x: &[f64]| {
            Ok::<_, std::convert::Infallible>(vec![4.0 * x[0] + x[1], -2.0 * x[0] + 3.0 * x[1]])
        };
        let policy = GmresPolicy {
            relative_tolerance: 1.0e-12,
            restart: 2,
            maximum_iterations: 20,
        };
        let result = restarted_gmres(&[6.0, 4.0], &[0.0; 2], policy, apply, || false)
            .unwrap()
            .unwrap();
        assert!(result.converged);
        assert!((result.solution[0] - 1.0).abs() <= 1.0e-12);
        assert!((result.solution[1] - 2.0).abs() <= 1.0e-12);
        let cancelled = restarted_gmres(&[6.0, 4.0], &[0.0; 2], policy, apply, || true)
            .unwrap()
            .unwrap();
        assert!(cancelled.cancelled);
        assert!(!cancelled.converged);
    }

    #[test]
    fn operator_failure_is_not_converted_into_a_breakdown() {
        let result = restarted_gmres(
            &[1.0],
            &[0.0],
            GmresPolicy {
                relative_tolerance: 1.0e-12,
                restart: 1,
                maximum_iterations: 2,
            },
            |_value| Err::<Vec<f64>, _>("scoped worker failed"),
            || false,
        );
        assert_eq!(result, Err("scoped worker failed"));
    }
}
