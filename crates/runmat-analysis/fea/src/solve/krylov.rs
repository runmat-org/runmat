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

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum GmresFailure<E> {
    Operator(E),
    Usage(runmat_simulation_usage::LedgerError),
}

/// Restarted GMRES on the shared sparse recurrence for operators supplied as closures (callers
/// that precondition compose `A M⁻¹` themselves). The true relative residual is formed when the
/// Arnoldi estimate passes and certifies convergence. Operator failures propagate unchanged;
/// cancellation and breakdown are reported in the result.
pub(crate) fn restarted_gmres<E>(
    right_hand_side: &[f64],
    initial: &[f64],
    policy: GmresPolicy,
    apply: impl Fn(&[f64]) -> Result<Vec<f64>, E>,
    cancelled: impl Fn() -> bool,
) -> Result<Option<GmresResult>, GmresFailure<E>> {
    use runmat_sparse_solvers::krylov::gmres::{
        self, GmresGeometry, GmresMode, GmresOperations, GmresParameters, PreconditionerSide,
        ResidualCheck,
    };
    use runmat_sparse_solvers::krylov::{
        block_on, ApplyStatus, HostKernels, KrylovFailure, ScalarArithmetic,
    };
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
    enum Stop<E> {
        Operator(E),
        Usage(runmat_simulation_usage::LedgerError),
        Dimension,
        Cancelled,
    }
    struct ClosureOperations<'a, E, A: Fn(&[f64]) -> Result<Vec<f64>, E>, C: Fn() -> bool> {
        apply: &'a A,
        cancelled: &'a C,
    }
    impl<E, A: Fn(&[f64]) -> Result<Vec<f64>, E>, C: Fn() -> bool> GmresOperations<f64>
        for ClosureOperations<'_, E, A, C>
    {
        type Error = Stop<E>;
        async fn measured<R>(
            &mut self,
            work: runmat_sparse_solvers::krylov::usage::VectorWork,
            run: impl FnOnce(&mut Self) -> (R, runmat_sparse_solvers::krylov::usage::BatchCompletion),
        ) -> Result<R, Stop<E>> {
            let Some(context) = runmat_simulation_usage::UsageContext::capture() else {
                return Ok(run(self).0);
            };
            let count = work
                .operations()
                .map_err(|error| Stop::Usage(error.into()))?;
            let mut reservation = context
                .ledger()
                .reserve(runmat_simulation_usage::UsageCategory::Solve, count)
                .map_err(Stop::Usage)?;
            let (result, completed) = run(self);
            match completed {
                runmat_sparse_solvers::krylov::usage::BatchCompletion::Skipped => {}
                runmat_sparse_solvers::krylov::usage::BatchCompletion::Complete => {
                    reservation.complete(count).map_err(Stop::Usage)?
                }
                runmat_sparse_solvers::krylov::usage::BatchCompletion::Prefix(prefix) => {
                    reservation.complete(prefix).map_err(Stop::Usage)?
                }
            }
            Ok(result)
        }
        async fn coefficient(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Stop<E>> {
            let values = (self.apply)(input).map_err(Stop::Operator)?;
            if values.len() != output.len() {
                return Err(Stop::Dimension);
            }
            output.copy_from_slice(&values);
            Ok(())
        }
        async fn precondition(
            &mut self,
            input: &[f64],
            output: &mut [f64],
            _intermediate: &mut [f64],
        ) -> Result<ApplyStatus, Stop<E>> {
            output.copy_from_slice(input);
            Ok(ApplyStatus::Applied)
        }
        fn checkpoint(&mut self) -> Result<(), Stop<E>> {
            if (self.cancelled)() {
                return Err(Stop::Cancelled);
            }
            Ok(())
        }
        async fn cooperate(&mut self, _completed: u64) -> Result<(), Stop<E>> {
            Ok(())
        }
        // The shared host kernels (deterministic blocked reductions on the admitted pool).
        fn native_norm(&mut self, values: &[f64]) -> Option<f64> {
            HostKernels::norm(values)
        }
        fn native_dot(
            &mut self,
            left: &[f64],
            right: &[f64],
        ) -> Option<runmat_sparse_solvers::c64> {
            HostKernels::dot(left, right)
        }
        fn native_axpy(
            &mut self,
            target: &mut [f64],
            scale: runmat_sparse_solvers::c64,
            source: &[f64],
        ) -> bool {
            HostKernels::axpy(target, scale, source)
        }
        fn native_divide(&mut self, target: &mut [f64], source: &[f64], divisor: f64) -> bool {
            HostKernels::divide(target, source, divisor)
        }
        fn native_project(
            &mut self,
            basis: &[f64],
            rows: usize,
            w: &mut [f64],
            coefficients: &mut [runmat_sparse_solvers::c64],
        ) -> bool {
            HostKernels::project(basis, rows, w, coefficients)
        }
        fn native_residual(
            &mut self,
            rhs: &[f64],
            product: &[f64],
            residual: &mut [f64],
        ) -> Option<bool> {
            HostKernels::residual(rhs, product, residual)
        }
        fn native_equal(&mut self, left: &[f64], right: &[f64]) -> Option<bool> {
            HostKernels::equal(left, right)
        }
    }
    if right_hand_side.iter().all(|value| *value == 0.0) {
        // The exact solution of the nonsingular system, whatever the initial guess.
        return Ok(Some(GmresResult {
            solution: vec![0.0; size],
            iterations: 0,
            residual_norm: 0.0,
            relative_residual: crate::solve::convergence::relative_residual(0.0, 0.0),
            converged: true,
            cancelled: false,
            breakdown: false,
        }));
    }
    let restart = policy.restart.min(size).max(1);
    let max_outer = policy.maximum_iterations.max(1).div_ceil(restart);
    let Some(geometry) = GmresGeometry::checked(size, restart, max_outer) else {
        return Ok(None);
    };
    let mut native = vec![0.0; geometry.native_elements];
    let mut complex = vec![
        runmat_sparse_solvers::c64::new(0.0, 0.0);
        geometry.hessenberg_elements + geometry.complex_small_elements
    ];
    let mut real = vec![0.0; geometry.real_small_elements + geometry.history_elements];
    let Some(mut work) = geometry.project(&mut native, &mut complex, &mut real) else {
        return Ok(None);
    };
    let mut operations = ClosureOperations {
        apply: &apply,
        cancelled: &cancelled,
    };
    let outcome = block_on(gmres::solve_with::<f64, ScalarArithmetic, _>(
        right_hand_side,
        Some(initial),
        GmresParameters {
            restart,
            max_outer,
            tolerance: policy.relative_tolerance,
        },
        GmresMode {
            side: PreconditionerSide::Right,
            residual_check: ResidualCheck::OnEstimate,
            step_limit: Some(policy.maximum_iterations.max(1)),
        },
        &mut work,
        &mut [],
        &mut operations,
    ));
    let solution = work.x.to_vec();
    let reference_norm = runmat_sparse_solvers::par::norm2(right_hand_side);
    let (iterations, converged, breakdown, cancelled, residual_norm) = match outcome {
        Ok(outcome) => (
            outcome.steps,
            outcome.flag == 0,
            outcome.flag == 4,
            false,
            if reference_norm > 0.0 {
                outcome.relres * reference_norm
            } else {
                outcome.relres
            },
        ),
        Err(KrylovFailure::Operation(Stop::Operator(error))) => {
            return Err(GmresFailure::Operator(error))
        }
        Err(KrylovFailure::Operation(Stop::Usage(error))) => {
            return Err(GmresFailure::Usage(error))
        }
        Err(KrylovFailure::Operation(Stop::Dimension) | KrylovFailure::Invalid(_)) => {
            return Ok(None)
        }
        Err(KrylovFailure::Operation(Stop::Cancelled)) => {
            let applied = apply(&solution).map_err(GmresFailure::Operator)?;
            let residual = runmat_sparse_solvers::par::norm2(
                &right_hand_side
                    .iter()
                    .zip(&applied)
                    .map(|(b, a)| b - a)
                    .collect::<Vec<_>>(),
            );
            (0, false, false, true, residual)
        }
    };
    if !residual_norm.is_finite() {
        return Ok(None);
    }
    let reference = if reference_norm > 0.0 {
        reference_norm
    } else {
        residual_norm
    };
    Ok(Some(GmresResult {
        solution,
        iterations: u32::try_from(iterations).unwrap_or(u32::MAX),
        residual_norm,
        relative_residual: crate::solve::convergence::relative_residual(residual_norm, reference),
        converged,
        cancelled,
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
        assert_eq!(result, Err(GmresFailure::Operator("scoped worker failed")));
    }
}
