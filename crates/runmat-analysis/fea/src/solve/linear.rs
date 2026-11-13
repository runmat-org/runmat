use serde::{Deserialize, Serialize};

mod admission;
mod input;
pub(crate) use input::{solve_structural_system_reusing, StructuralLinearSystemView};
mod diagnostics;
mod settings;
use diagnostics::SuccessfulDiagnostic;
pub(crate) use diagnostics::{successful_diagnostic_record_count, successful_diagnostics_profile};
pub(crate) use settings::accelerator_iteration_limit;
pub(crate) use settings::ScopedLinearSolveSettings;

use admission::{incompatible_system_result, invalid_system_result, rejected_system_result};

use crate::{
    algebra::{
        validate_prepared_solver_compatibility, AssembledRealSystem, RealPreconditioner,
        RealSolver, ScopedSpmvError,
    },
    assembly::AssemblySummary,
    contracts::{FeaRunError, SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    planning::resource::{LocalExecutionPlan, SolverAdmissionResourcePolicy},
    solve::{
        accelerator::{prepare_accelerator_linear_system, solve_prepared_accelerator_system},
        backend::{kind::LinearAlgebraBackendKind, linear_algebra::LinearAlgebraBackend},
    },
    solve::{preconditioner::SpdPreconditionerKind, shared_sparse},
};

struct PreparedSolveSettings<'a> {
    plan: LocalExecutionPlan,
    admission: Option<&'a SolverAdmissionResourcePolicy>,
    relative_tolerance: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearSolveResult<StateStorage = Vec<f64>> {
    pub iterations: u32,
    pub residual_norm: f64,
    /// Relative residual in the actual solved equation space; absent after admission failure.
    pub relative_residual: Option<f64>,
    pub converged: bool,
    pub host_sync_count: u32,
    pub solver_backend: LinearAlgebraBackendKind,
    pub device_apply_k_count: u32,
    pub device_apply_k_attempt_count: u32,
    pub accelerator_evidence: Option<crate::solve::accelerator::AcceleratorSolveEvidence>,
    pub solution: StateStorage,
    pub solver_method: SolverMethod,
    pub preconditioner: SolverPreconditioner,
    pub diagnostics: Vec<FeaDiagnostic>,
    /// PCG only: Lanczos λ_min estimate and the forward error it implies (never a proof).
    pub spectral_estimate: Option<crate::contracts::SpectralErrorEstimate>,
}

impl<S> LinearSolveResult<S> {
    pub fn map_solution<T>(self, map: impl FnOnce(S) -> T) -> LinearSolveResult<T> {
        self.try_map_solution(|storage| Ok::<_, std::convert::Infallible>(map(storage)))
            .unwrap_or_else(|never| match never {})
    }
    /// Move the solver's numerical state into a checked physical representation
    /// without duplicating its buffer or its backend/admission evidence.
    pub fn try_map_solution<T, E>(
        self,
        map: impl FnOnce(S) -> Result<T, E>,
    ) -> Result<LinearSolveResult<T>, E> {
        let LinearSolveResult {
            iterations,
            residual_norm,
            relative_residual,
            converged,
            host_sync_count,
            solver_backend,
            device_apply_k_count,
            device_apply_k_attempt_count,
            accelerator_evidence,
            solution,
            solver_method,
            preconditioner,
            diagnostics,
            spectral_estimate,
        } = self;
        Ok(LinearSolveResult {
            iterations,
            residual_norm,
            relative_residual,
            converged,
            host_sync_count,
            solver_backend,
            device_apply_k_count,
            device_apply_k_attempt_count,
            accelerator_evidence,
            solution: map(solution)?,
            solver_method,
            preconditioner,
            diagnostics,
            spectral_estimate,
        })
    }
}

pub fn solve_linear_system(
    summary: &AssemblySummary,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    input::solve_structural_system(
        &input::StructuralLinearSystemView::from(summary),
        preconditioner_kind,
        backend_kind,
        algebra_backend,
        initial_guess,
    )
}

/// Uses the admitted local CPU plan for ordinary, affine-reduced, and multiplier static stiffness
/// application. Multiplier GMRES propagates scoped execution failure without reporting breakdown.
pub fn solve_linear_system_with_execution(
    summary: &AssemblySummary,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    plan: LocalExecutionPlan,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    solve_linear_system_with_execution_admitted(
        summary,
        preconditioner_kind,
        backend_kind,
        algebra_backend,
        initial_guess,
        plan,
        None,
        None,
        None,
    )
}

pub(crate) fn solve_linear_system_with_execution_admitted(
    summary: &AssemblySummary,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    plan: LocalExecutionPlan,
    admission: Option<&SolverAdmissionResourcePolicy>,
    planned_reduced_support: Option<&crate::algebra::SparsePattern>,
    preparation: Option<&mut crate::algebra::LinearPreparationReuse>,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    input::solve_structural_system_with_execution(
        &input::StructuralLinearSystemView {
            planned_reduced_support,
            ..input::StructuralLinearSystemView::from(summary)
        },
        preconditioner_kind,
        backend_kind,
        algebra_backend,
        initial_guess,
        plan,
        admission,
        preparation,
    )
}

pub(crate) fn solve_operator_system_scoped(
    operator: &AssembledRealSystem,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    _algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    settings: ScopedLinearSolveSettings<'_>,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    solve_operator_system_with_settings(
        operator,
        preconditioner_kind,
        backend_kind,
        initial_guess,
        settings,
    )
}

pub fn solve_operator_system(
    operator: &AssembledRealSystem,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    _algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    solve_operator_system_with_settings(
        operator,
        preconditioner_kind,
        backend_kind,
        initial_guess,
        ScopedLinearSolveSettings::with_default_tolerance(LocalExecutionPlan::serial(), None),
    )
}

fn solve_operator_system_with_settings(
    operator: &AssembledRealSystem,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    initial_guess: Option<&[f64]>,
    settings: ScopedLinearSolveSettings<'_>,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    let tol = settings.relative_tolerance;
    if !tol.is_finite() || tol <= 0.0 {
        return Ok(rejected_system_result(
            operator,
            backend_kind,
            preconditioner_kind,
            "FEA_INVALID_SOLVER_TOLERANCE",
            "relative solver tolerance must be finite and positive".into(),
        ));
    }
    let effective_backend_kind = backend_kind;
    let has_dofs = operator.dof_count() > 0;
    if !has_dofs {
        return Ok(LinearSolveResult {
            spectral_estimate: None,
            iterations: 0,
            residual_norm: 0.0,
            relative_residual: None,
            converged: false,
            host_sync_count: 0,
            solver_backend: effective_backend_kind,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            accelerator_evidence: None,
            solution: Vec::new(),
            solver_method: admission::requested_provenance(preconditioner_kind).0,
            preconditioner: admission::requested_provenance(preconditioner_kind).1,
            diagnostics: vec![FeaDiagnostic {
                code: "FEA_EMPTY_SYSTEM".to_string(),
                severity: FeaDiagnosticSeverity::Warning,
                message: "linear solve skipped because assembled system has zero DOFs".to_string(),
            }],
        });
    }

    let plan = settings.plan;
    let reuse = settings.preparation;
    // The coercivity proof factors the operator; run it on the admitted plan.
    let preparation = shared_sparse::with_plan(plan, move || {
        operator.prepare_linear_with_reuse(reuse, settings.admission)
    });
    let prepared = match preparation {
        Ok(prepared) => prepared,
        Err(error) => {
            return Ok(invalid_system_result(
                operator,
                effective_backend_kind,
                preconditioner_kind,
                error,
            ))
        }
    };
    solve_prepared_operator_system(
        operator,
        preconditioner_kind,
        backend_kind,
        initial_guess,
        PreparedSolveSettings {
            plan,
            admission: settings.admission,
            relative_tolerance: tol,
        },
        prepared,
    )
}

/// The owned browser path awaits the exact symbolic factor grant before numerical
/// factorization. The resulting prepared system enters the same solver as the sync path.
pub(crate) async fn solve_operator_system_with_settings_owned<O, F, A>(
    operator: &AssembledRealSystem,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    initial_guess: Option<&[f64]>,
    settings: ScopedLinearSolveSettings<'_>,
    mut before_factor: O,
) -> Result<LinearSolveResult, crate::algebra::coercivity::NumericalCapacityError<FeaRunError, A>>
where
    O: FnMut(
        crate::algebra::coercivity::NumericalCapacityPhase,
        runmat_simulation_usage::OperationCount,
    ) -> F,
    F: std::future::Future<Output = Result<(), A>>,
{
    use crate::algebra::coercivity::NumericalCapacityError;
    let tol = settings.relative_tolerance;
    if !tol.is_finite() || tol <= 0.0 || operator.dof_count() == 0 {
        return solve_operator_system_with_settings(
            operator,
            preconditioner_kind,
            backend_kind,
            initial_guess,
            settings,
        )
        .map_err(|error| NumericalCapacityError::Numerical(scoped_solve_error(error)));
    }
    let plan = settings.plan;
    let admission = settings.admission;
    let prepared = match operator
        .prepare_linear_with_reuse_owned(settings.preparation, admission, &mut before_factor)
        .await
    {
        Ok(prepared) => prepared,
        Err(NumericalCapacityError::Numerical(error)) => {
            return Ok(invalid_system_result(
                operator,
                backend_kind,
                preconditioner_kind,
                error,
            ));
        }
        Err(NumericalCapacityError::Admission(error)) => {
            return Err(NumericalCapacityError::Admission(error));
        }
    };
    solve_prepared_operator_system_owned(
        operator,
        preconditioner_kind,
        backend_kind,
        initial_guess,
        PreparedSolveSettings {
            plan,
            admission,
            relative_tolerance: tol,
        },
        prepared,
        before_factor,
    )
    .await
}

fn scoped_solve_error(error: ScopedSpmvError) -> FeaRunError {
    match error {
        ScopedSpmvError::Usage(cause) => FeaRunError::Usage(cause),
        other => FeaRunError::Solve(other.to_string()),
    }
}

async fn solve_prepared_operator_system_owned<O, F, A>(
    operator: &AssembledRealSystem,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    initial_guess: Option<&[f64]>,
    settings: PreparedSolveSettings<'_>,
    prepared: crate::algebra::PreparedRealLinearSystem,
    before_capacity: O,
) -> Result<LinearSolveResult, crate::algebra::coercivity::NumericalCapacityError<FeaRunError, A>>
where
    O: FnMut(crate::algebra::NumericalCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
{
    use crate::algebra::coercivity::NumericalCapacityError;
    let PreparedSolveSettings {
        admission,
        relative_tolerance: tol,
        ..
    } = settings;
    if let Some(admission) = admission {
        if !prepared
            .admissibility()
            .is_some_and(|evidence| admission.admits_certificate(evidence.coercivity()))
        {
            return Ok(rejected_system_result(
                operator,
                backend_kind,
                preconditioner_kind,
                "FEA_SOLVER_WORK_EXCEEDS_ADMISSION",
                "exact prepared-solver work exceeds the admitted resource envelope".into(),
            ));
        }
    }
    if let Err(error) = validate_prepared_solver_compatibility(
        &prepared,
        RealSolver::ConjugateGradient,
        match preconditioner_kind {
            SpdPreconditionerKind::Jacobi => RealPreconditioner::Jacobi,
            SpdPreconditionerKind::Ilu0 => RealPreconditioner::Ilu0,
            _ => RealPreconditioner::None,
        },
    ) {
        return Ok(incompatible_system_result(
            operator,
            backend_kind,
            preconditioner_kind,
            error,
        ));
    }
    let factor = prepared
        .admissibility()
        .and_then(|evidence| evidence.factor())
        .map(|factor| factor.0.clone());
    let strategy = shared_sparse::resolve_spd_strategy(
        preconditioner_kind,
        prepared.dof_count(),
        prepared.matrix().values().len(),
        1,
        factor.is_some(),
    );
    if backend_kind != LinearAlgebraBackendKind::CpuReference {
        return Err(NumericalCapacityError::Numerical(FeaRunError::Solve(
            "awaited structural solve requires the CPU reference backend".into(),
        )));
    }
    let b = prepared.rhs();
    let mut x = match initial_guess {
        Some(values)
            if values.len() == operator.dof_count()
                && values.iter().all(|value| value.is_finite()) =>
        {
            values.to_vec()
        }
        Some(_) => {
            return Ok(rejected_system_result(
                operator,
                backend_kind,
                preconditioner_kind,
                "FEA_INVALID_INITIAL_GUESS",
                "initial guess must contain one finite value per solver DOF".into(),
            ))
        }
        None => vec![0.0; operator.dof_count()],
    };
    for (index, constrained) in prepared.constrained_dofs().iter().copied().enumerate() {
        if constrained {
            x[index] = b[index];
        }
    }
    let arrays = shared_sparse::CsrArrays::of(prepared.matrix());
    let outcome = match strategy {
        shared_sparse::SpdStrategy::Cholesky => {
            let factor = factor.as_ref().ok_or_else(|| {
                NumericalCapacityError::Numerical(FeaRunError::Solve(
                    "awaited direct solve requires the coercivity-proving factor".into(),
                ))
            })?;
            shared_sparse::solve_cholesky_with_owner(
                arrays.view(),
                b,
                Some(&x),
                tol,
                Some(factor.as_ref()),
                before_capacity,
            )
            .await
            .map_err(|error| error.map_numerical(FeaRunError::Usage))?
        }
        shared_sparse::SpdStrategy::Pcg(
            choice @ (runmat_sparse_solvers::policy::PreconditionerChoice::Jacobi
            | runmat_sparse_solvers::policy::PreconditionerChoice::Ilu0
            | runmat_sparse_solvers::policy::PreconditionerChoice::Amg),
        ) => shared_sparse::solve_pcg_with_owner(
            arrays.view(),
            b,
            Some(&x),
            tol,
            choice,
            before_capacity,
        )
        .await
        .map_err(|error| {
            error.map_numerical(|error| match error {
                runmat_sparse_solvers::SolverError::Usage(cause) => FeaRunError::Usage(cause),
                other => FeaRunError::Solve(other.to_string()),
            })
        })?,
        _ => {
            return Err(NumericalCapacityError::Numerical(FeaRunError::Solve(
                "awaited PCG preconditioner has no resumable reference-work owner".into(),
            )))
        }
    };
    let (solver_method, preconditioner) = strategy.provenance();
    let mut diagnostics = vec![SuccessfulDiagnostic::CpuMethod(strategy).diagnostic()];
    if outcome.cancelled {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_SOLVER_CANCELLED".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: "linear solve cancelled".to_string(),
        });
    } else if !outcome.converged {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_CG_NOT_CONVERGED".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: format!(
                "{} did not reach tolerance {tol}: {}",
                strategy.label(),
                outcome
                    .failure
                    .as_deref()
                    .unwrap_or("residual above tolerance")
            ),
        });
    }
    Ok(LinearSolveResult {
        spectral_estimate: outcome.spectral_estimate,
        iterations: outcome.iterations,
        residual_norm: outcome.residual_norm,
        relative_residual: crate::solve::convergence::relative_residual(
            outcome.residual_norm,
            outcome.reference_norm,
        ),
        converged: outcome.converged && !outcome.cancelled,
        host_sync_count: 0,
        solver_backend: backend_kind,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution: outcome.solution,
        solver_method,
        preconditioner,
        diagnostics,
    })
}

fn solve_prepared_operator_system(
    operator: &AssembledRealSystem,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    initial_guess: Option<&[f64]>,
    settings: PreparedSolveSettings<'_>,
    prepared: crate::algebra::PreparedRealLinearSystem,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    let PreparedSolveSettings {
        plan,
        admission,
        relative_tolerance: tol,
    } = settings;
    let mut effective_backend_kind = backend_kind;
    let mut accelerator_fallback = false;
    if let Some(admission) = admission {
        let accepted = prepared
            .admissibility()
            .is_some_and(|evidence| admission.admits_certificate(evidence.coercivity()));
        if !accepted {
            return Ok(rejected_system_result(
                operator,
                effective_backend_kind,
                preconditioner_kind,
                "FEA_SOLVER_WORK_EXCEEDS_ADMISSION",
                "exact prepared-solver work exceeds the admitted resource envelope".into(),
            ));
        }
    }
    if let Err(error) = validate_prepared_solver_compatibility(
        &prepared,
        RealSolver::ConjugateGradient,
        match preconditioner_kind {
            SpdPreconditionerKind::Jacobi => RealPreconditioner::Jacobi,
            SpdPreconditionerKind::Ilu0 => RealPreconditioner::Ilu0,
            SpdPreconditionerKind::Auto
            | SpdPreconditionerKind::Direct
            | SpdPreconditionerKind::Amg => RealPreconditioner::None,
        },
    ) {
        return Ok(incompatible_system_result(
            operator,
            effective_backend_kind,
            preconditioner_kind,
            error,
        ));
    }
    // Compatibility is a solver-independent admission boundary. A runtime provider must never
    // receive a system that the reference path would reject, and provider fallback must not alter
    // that decision.
    // One canonical SPD path: the shared sparse layer solves with the coercivity-proving
    // Cholesky factor when one exists, otherwise with the policy's strategy.
    let factor = prepared
        .admissibility()
        .and_then(|evidence| evidence.factor())
        .map(|factor| factor.0.clone());
    let strategy = shared_sparse::resolve_spd_strategy(
        preconditioner_kind,
        prepared.dof_count(),
        prepared.matrix().values().len(),
        1,
        factor.is_some(),
    );
    // A multigrid strategy on an accelerator runs mixed precision: host hierarchy, device inner
    // CG, host f64 flexible CG with the f64 true-residual certificate. Other strategies use the
    // provider's resident Jacobi CG.
    let device = shared_sparse::multigrid_device(backend_kind, strategy);
    if backend_kind == LinearAlgebraBackendKind::Accelerator && device.is_none() {
        let maximum_iterations = accelerator_iteration_limit(operator, 64);
        match prepare_accelerator_linear_system(prepared.clone(), maximum_iterations).and_then(
            |accelerator| {
                solve_prepared_accelerator_system(&accelerator, prepared.rhs(), initial_guess, tol)
            },
        ) {
            Ok(output) => {
                // The provider path is Jacobi-preconditioned CG.
                return Ok(linear_result_from_accelerator_output(
                    output,
                    SpdPreconditionerKind::Jacobi,
                ));
            }
            Err(error) if error.permits_cpu_fallback() => {
                effective_backend_kind = LinearAlgebraBackendKind::CpuReference;
                accelerator_fallback = true;
            }
            Err(error) => {
                return Ok(linear_result_from_accelerator_error(
                    operator.dof_count(),
                    operator.load(),
                    SpdPreconditionerKind::Jacobi,
                    error,
                ))
            }
        }
    }
    let mut initial_diagnostics = Vec::new();
    if accelerator_fallback {
        initial_diagnostics.push(SuccessfulDiagnostic::AdmissionFallback.diagnostic());
    }
    let b = prepared.rhs();
    let mut x = match initial_guess {
        Some(values)
            if values.len() == operator.dof_count()
                && values.iter().all(|value| value.is_finite()) =>
        {
            values.to_vec()
        }
        Some(_) => {
            return Ok(rejected_system_result(
                operator,
                effective_backend_kind,
                preconditioner_kind,
                "FEA_INVALID_INITIAL_GUESS",
                "initial guess must contain one finite value per solver DOF".to_string(),
            ))
        }
        None => vec![0.0; operator.dof_count()],
    };
    for (index, constrained) in prepared.constrained_dofs().iter().copied().enumerate() {
        if constrained {
            x[index] = b[index];
        }
    }
    let outcome = shared_sparse::solve_spd(shared_sparse::SpdSolveRequest {
        matrix: prepared.matrix(),
        rhs: b,
        initial_guess: Some(&x),
        relative_tolerance: tol,
        plan,
        strategy,
        block_size: 1,
        factor: factor.as_deref(),
        device,
    })
    .map_err(ScopedSpmvError::Usage)?;
    let mut diagnostics = initial_diagnostics;
    diagnostics.push(SuccessfulDiagnostic::CpuMethod(strategy).diagnostic());
    if outcome.device_preconditioner {
        effective_backend_kind = LinearAlgebraBackendKind::Accelerator;
        diagnostics.push(FeaDiagnostic {
            code: "FEA_SOLVER_DEVICE_PRECONDITIONER".to_string(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "solver=flexible_cg preconditioner=device_amg_inner_cg inner_steps={} inner_precision=f32 outer_precision=f64",
                shared_sparse::DEVICE_INNER_CG_STEPS
            ),
        });
    } else if device.is_some() {
        // The provider could not hold the hierarchy: the host multigrid ran instead.
        effective_backend_kind = LinearAlgebraBackendKind::CpuReference;
        diagnostics.push(SuccessfulDiagnostic::AdmissionFallback.diagnostic());
    }
    if outcome.cancelled {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_SOLVER_CANCELLED".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: "linear solve cancelled".to_string(),
        });
    } else if !outcome.converged {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_CG_NOT_CONVERGED".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: format!(
                "{} did not reach tolerance {tol}: {}",
                strategy.label(),
                outcome
                    .failure
                    .as_deref()
                    .unwrap_or("residual above tolerance"),
            ),
        });
    }
    let (solver_method, preconditioner) = strategy.provenance();
    Ok(LinearSolveResult {
        spectral_estimate: outcome.spectral_estimate,
        iterations: outcome.iterations,
        residual_norm: outcome.residual_norm,
        relative_residual: crate::solve::convergence::relative_residual(
            outcome.residual_norm,
            outcome.reference_norm,
        ),
        converged: outcome.converged && !outcome.cancelled,
        host_sync_count: 0,
        solver_backend: effective_backend_kind,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution: outcome.solution,
        solver_method,
        preconditioner,
        diagnostics,
    })
}

pub(crate) fn linear_result_from_accelerator_output(
    output: crate::solve::accelerator::AcceleratorSolveOutput,
    preconditioner_kind: SpdPreconditionerKind,
) -> LinearSolveResult {
    let provider_evidence = output.evidence.provider();
    let host_sync_count = provider_evidence.host_synchronizations.saturating_add(1);
    let device_apply_k_count = u32::try_from(provider_evidence.spmv_dispatches).unwrap_or(u32::MAX);
    let mut diagnostics =
        vec![SuccessfulDiagnostic::AcceleratorMethod(preconditioner_kind).diagnostic()];
    if !output.converged {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_ACCELERATOR_TOLERANCE_NOT_SATISFIED".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: "native sparse solve did not satisfy the independently recomputed residual tolerance"
                .to_string(),
        });
    }
    LinearSolveResult {
        spectral_estimate: None,
        iterations: output.iterations,
        residual_norm: output
            .evidence
            .independent_residual_norm()
            .expect("completed accelerator evidence has a residual"),
        relative_residual: output.evidence.independent_relative_residual(),
        converged: output.converged,
        host_sync_count,
        solver_backend: LinearAlgebraBackendKind::Accelerator,
        device_apply_k_count,
        device_apply_k_attempt_count: device_apply_k_count,
        accelerator_evidence: Some(output.evidence),
        solution: output.solution,
        solver_method: SolverMethod::AssembledPcg,
        preconditioner: SolverPreconditioner::from_spd(preconditioner_kind),
        diagnostics,
    }
}

pub(crate) fn linear_result_from_accelerator_error(
    dof_count: usize,
    right_hand_side: &[f64],
    preconditioner_kind: SpdPreconditionerKind,
    error: crate::solve::accelerator::AcceleratorAdapterError,
) -> LinearSolveResult {
    let terminal = error.phase;
    LinearSolveResult {
        spectral_estimate: None,
        iterations: 0,
        residual_norm: right_hand_side
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt(),
        relative_residual: None,
        converged: false,
        host_sync_count: error
            .evidence
            .as_ref()
            .map_or(0, |evidence| evidence.provider().host_synchronizations),
        solver_backend: LinearAlgebraBackendKind::Accelerator,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: error.evidence.map(|evidence| *evidence),
        solution: vec![0.0; dof_count],
        solver_method: SolverMethod::AssembledPcg,
        preconditioner: SolverPreconditioner::from_spd(preconditioner_kind),
        diagnostics: vec![FeaDiagnostic {
            code: match terminal {
                crate::solve::accelerator::AcceleratorAdapterPhase::Cancellation => {
                    "FEA_SOLVER_CANCELLED"
                }
                crate::solve::accelerator::AcceleratorAdapterPhase::ProviderLoss => {
                    "FEA_ACCELERATOR_PROVIDER_LOST"
                }
                _ => "FEA_ACCELERATOR_SOLVE_FAILED",
            }
            .to_string(),
            severity: match terminal {
                crate::solve::accelerator::AcceleratorAdapterPhase::Cancellation => {
                    FeaDiagnosticSeverity::Warning
                }
                _ => FeaDiagnosticSeverity::Error,
            },
            message: error.message,
        }],
    }
}

#[cfg(test)]
#[path = "linear/tests.rs"]
mod tests;
