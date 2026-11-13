use crate::{
    algebra::{PreparedRealLinearSystem, ScopedSpmvError},
    constraints::AffineConstraintReduction,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    planning::resource::{LocalExecutionPlan, SolverAdmissionResourcePolicy},
    solve::linear::{ScopedLinearSolveSettings, StructuralLinearSystemView},
    solve::{
        backend::kind::LinearAlgebraBackendKind, linear::LinearSolveResult,
        preconditioner::SpdPreconditionerKind, shared_sparse,
    },
};

use super::backend::linear_algebra::LinearAlgebraBackend;
use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};

pub(crate) struct AffineOwnedSolveOptions<'a> {
    settings: ScopedLinearSolveSettings<'a>,
    kind: SpdPreconditionerKind,
    allow_sync_other_strategy: bool,
}

impl<'a> AffineOwnedSolveOptions<'a> {
    pub(crate) fn new(
        settings: ScopedLinearSolveSettings<'a>,
        kind: SpdPreconditionerKind,
        allow_sync_other_strategy: bool,
    ) -> Self {
        Self {
            settings,
            kind,
            allow_sync_other_strategy,
        }
    }
}

fn affine_count(value: usize) -> Result<OperationCount, ScopedSpmvError> {
    Ok(OperationCount::from_operations(
        u64::try_from(value).map_err(|_| ScopedSpmvError::Usage(LedgerError::CountOverflow))?,
    ))
}

fn measured_affine<R>(
    work: OperationCount,
    run: impl FnOnce() -> Result<R, ScopedSpmvError>,
) -> Result<R, ScopedSpmvError> {
    let Some(context) = UsageContext::capture() else {
        return run();
    };
    let mut reservation = context
        .ledger()
        .reserve(UsageCategory::Solve, work)
        .map_err(ScopedSpmvError::Usage)?;
    let result = run()?;
    reservation.complete(work).map_err(ScopedSpmvError::Usage)?;
    Ok(result)
}

fn measured_affine_result<R, E>(
    work: OperationCount,
    run: impl FnOnce() -> Result<R, E>,
) -> Result<Result<R, E>, ScopedSpmvError> {
    let Some(context) = UsageContext::capture() else {
        return Ok(run());
    };
    let mut reservation = context
        .ledger()
        .reserve(UsageCategory::Solve, work)
        .map_err(ScopedSpmvError::Usage)?;
    let result = run();
    if result.is_ok() {
        reservation.complete(work).map_err(ScopedSpmvError::Usage)?;
    }
    Ok(result)
}

pub(crate) fn affine_rhs_work(
    reduction: &AffineConstraintReduction,
    stiffness: &crate::algebra::RealSparseMatrix,
) -> Result<OperationCount, ScopedSpmvError> {
    let dot_work = stiffness.pattern().row_offsets().windows(2).try_fold(
        OperationCount::ZERO,
        |total, row| {
            let entries = row[1] - row[0];
            let work = OperationCount::direct_real_dot(entries)
                .map_err(|error| ScopedSpmvError::Usage(error.into()))?;
            total
                .checked_add(work)
                .map_err(|error| ScopedSpmvError::Usage(error.into()))
        },
    )?;
    let shift_work = affine_count(reduction.full_dof_count)?;
    let transpose_work = affine_transpose_work(reduction)?;
    dot_work
        .checked_add(shift_work)
        .and_then(|work| work.checked_add(transpose_work))
        .map_err(|error| ScopedSpmvError::Usage(error.into()))
}

/// `Tᵀ f`: one multiply and one add per stored recovery coefficient.
pub(crate) fn affine_transpose_work(
    reduction: &AffineConstraintReduction,
) -> Result<OperationCount, ScopedSpmvError> {
    affine_count(reduction.recovery_coefficients.len())?
        .checked_mul(2)
        .map_err(|error| ScopedSpmvError::Usage(error.into()))
}

pub(crate) fn affine_scatter_work(
    reduction: &AffineConstraintReduction,
    stiffness: &crate::algebra::RealSparseMatrix,
) -> Result<OperationCount, ScopedSpmvError> {
    let pattern = stiffness.pattern();
    let mut work = OperationCount::ZERO;
    for row in 0..reduction.full_dof_count {
        let left = reduction.recovery_row_offsets[row + 1] - reduction.recovery_row_offsets[row];
        for entry in pattern.row_range(row) {
            let column = pattern.column(entry);
            let right =
                reduction.recovery_row_offsets[column + 1] - reduction.recovery_row_offsets[column];
            let pair_work = affine_count(left)?
                .checked_mul(
                    u64::try_from(right)
                        .map_err(|_| ScopedSpmvError::Usage(LedgerError::CountOverflow))?,
                )
                .and_then(|work| work.checked_mul(3))
                .map_err(|error| ScopedSpmvError::Usage(error.into()))?;
            work = work
                .checked_add(pair_work)
                .map_err(|error| ScopedSpmvError::Usage(error.into()))?;
        }
    }
    Ok(work)
}

pub(super) fn solve_affine_constraint_system(
    summary: &StructuralLinearSystemView<'_>,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    _algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    solve_affine_constraint_system_with_settings(
        summary,
        reduction,
        requested_backend,
        initial_guess,
        ScopedLinearSolveSettings::with_default_tolerance(LocalExecutionPlan::serial(), None),
    )
}

pub(super) fn solve_affine_constraint_system_with_execution(
    summary: &StructuralLinearSystemView<'_>,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    _algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    plan: LocalExecutionPlan,
    admission: Option<&SolverAdmissionResourcePolicy>,
    preconditioner: SpdPreconditionerKind,
    preparation: Option<&mut crate::algebra::LinearPreparationReuse>,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    let mut settings = ScopedLinearSolveSettings::with_default_tolerance(plan, admission);
    settings.preparation = preparation;
    // A planned proof without a factor (a supplied preparation) solves with the planned
    // preconditioner; otherwise the proving factor, when one exists, solves.
    let kind = if settings.preparation.is_some() {
        preconditioner
    } else {
        SpdPreconditionerKind::Auto
    };
    solve_affine_constraint_system_with_kind(
        summary,
        reduction,
        requested_backend,
        initial_guess,
        settings,
        kind,
    )
}

/// Solves the affine-reduced SPD system `Tᵀ K T u = Tᵀ (f − K c)` through the shared sparse layer
/// (the coercivity-proving Cholesky factor of the reduced operator is the direct solver) and
/// expands `x = T u + c`.
fn solve_affine_constraint_system_with_settings(
    summary: &StructuralLinearSystemView<'_>,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    initial_guess: Option<&[f64]>,
    settings: ScopedLinearSolveSettings<'_>,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    solve_affine_constraint_system_with_kind(
        summary,
        reduction,
        requested_backend,
        initial_guess,
        settings,
        SpdPreconditionerKind::Auto,
    )
}

fn solve_affine_constraint_system_with_kind(
    summary: &StructuralLinearSystemView<'_>,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    initial_guess: Option<&[f64]>,
    settings: ScopedLinearSolveSettings<'_>,
    kind: SpdPreconditionerKind,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    shared_sparse::with_plan(settings.plan, move || {
        let future = solve_affine_constraint_system_with_kind_owned(
            summary,
            reduction,
            requested_backend,
            initial_guess,
            AffineOwnedSolveOptions::new(settings, kind, true),
            |_, _| std::future::ready(Ok::<(), std::convert::Infallible>(())),
        );
        let mut future = std::pin::pin!(future);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match std::future::Future::poll(future.as_mut(), &mut context) {
            std::task::Poll::Ready(result) => result.map_err(|error| match error {
                crate::algebra::coercivity::NumericalCapacityError::Numerical(error) => error,
                crate::algebra::coercivity::NumericalCapacityError::Admission(never) => {
                    match never {}
                }
            }),
            std::task::Poll::Pending => unreachable!("immediate affine owner cannot suspend"),
        }
    })
}

pub(crate) async fn solve_affine_constraint_system_with_kind_owned<O, F, A>(
    summary: &StructuralLinearSystemView<'_>,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    initial_guess: Option<&[f64]>,
    options: AffineOwnedSolveOptions<'_>,
    mut before_capacity: O,
) -> Result<LinearSolveResult, crate::algebra::coercivity::NumericalCapacityError<ScopedSpmvError, A>>
where
    O: FnMut(crate::algebra::NumericalCapacityPhase, OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
{
    use crate::algebra::{coercivity::NumericalCapacityError, NumericalCapacityPhase};
    let AffineOwnedSolveOptions {
        settings,
        kind,
        allow_sync_other_strategy,
    } = options;
    let admission = settings.admission;
    let reuse = settings.preparation;
    let plan = settings.plan;
    let tolerance = settings.relative_tolerance;
    let expected_reduction = summary
        .constraint_matrix
        .as_ref()
        .and_then(|matrix| matrix.affine_reduction().ok());
    if expected_reduction.as_ref() != Some(reduction) {
        return reject_affine_with_owner(
            summary,
            "affine reduction differs from the canonical resolved constraint matrix".into(),
            &mut before_capacity,
        )
        .await;
    }
    let stiffness = summary.operator.operators().stiffness();
    let derived_support;
    let reduced_support = match summary.planned_reduced_support {
        Some(planned) => planned,
        None => match reduction.reduce_sparse_pattern(stiffness.pattern()) {
            Ok(pattern) => {
                derived_support = pattern;
                &derived_support
            }
            Err(error) => {
                return reject_affine_with_owner(summary, error.to_string(), &mut before_capacity)
                    .await
            }
        },
    };
    let rhs_work = affine_rhs_work(reduction, stiffness)?;
    before_capacity(NumericalCapacityPhase::AffineRhsReduction, rhs_work)
        .await
        .map_err(NumericalCapacityError::Admission)?;
    let reduced_rhs = match measured_affine_result(rhs_work, || {
        reduction.reduce_rhs(stiffness, summary.operator.load())
    })? {
        Ok(rhs) => rhs,
        Err(error) => {
            return reject_affine_with_owner(summary, error.to_string(), &mut before_capacity).await
        }
    };
    let scatter_work = affine_scatter_work(reduction, stiffness)?;
    before_capacity(NumericalCapacityPhase::AffineOperatorScatter, scatter_work)
        .await
        .map_err(NumericalCapacityError::Admission)?;
    let reduced_operator = match measured_affine_result(scatter_work, || {
        reduction.reduce_sparse_operator_planned(stiffness, reduced_support)
    })? {
        Ok(operator) => operator,
        Err(error) => {
            return reject_affine_with_owner(summary, error.to_string(), &mut before_capacity).await
        }
    };
    let reduced_guess = initial_guess.map(|guess| {
        reduction
            .free_dof_indices
            .iter()
            .map(|dof| guess.get(*dof).copied().unwrap_or(0.0))
            .collect::<Vec<_>>()
    });
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_AFFINE_CONSTRAINT_REDUCTION".into(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "full_dofs={} reduced_dofs={} constraint_rank={}",
            reduction.full_dof_count,
            reduction.reduced_dof_count,
            reduction.pivot_dof_indices.len()
        ),
    }];
    if requested_backend == LinearAlgebraBackendKind::Accelerator {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_AFFINE_CPU_REFERENCE".into(),
            severity: FeaDiagnosticSeverity::Warning,
            message: "affine constraint reduction is executing on the CPU reference backend".into(),
        });
    }

    if reduction.reduced_dof_count == 0 {
        return Ok(LinearSolveResult {
            spectral_estimate: None,
            iterations: 0,
            residual_norm: 0.0,
            relative_residual: Some(0.0),
            converged: true,
            host_sync_count: 0,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            accelerator_evidence: None,
            solution: reduction.offset.clone(),
            solver_method: SolverMethod::SparseDirectCholesky,
            preconditioner: SolverPreconditioner::None,
            diagnostics,
        });
    }

    let reduced_dofs = reduction.reduced_dof_count;
    let reduced_pattern_entries = reduced_operator.pattern().column_indices().len();
    let preparation = PreparedRealLinearSystem::try_new_admitted_with_owner(
        reduced_operator,
        reduced_rhs,
        vec![false; reduced_dofs],
        reuse,
        admission,
        &mut before_capacity,
    )
    .await;
    let prepared = match preparation {
        Ok(prepared) => prepared,
        Err(NumericalCapacityError::Numerical(error)) => {
            return reject_affine_with_owner(summary, error.to_string(), &mut before_capacity).await
        }
        Err(NumericalCapacityError::Admission(error)) => {
            return Err(NumericalCapacityError::Admission(error))
        }
    };
    if let Some(admission) = admission {
        let accepted = prepared
            .admissibility()
            .is_some_and(|evidence| admission.admits_certificate(evidence.coercivity()));
        if !accepted {
            let exact = prepared
                .admissibility()
                .map(|evidence| evidence.coercivity().work_counts());
            return reject_affine_with_owner(
                summary,
                format!(
                    "exact reduced-solver work (dimension, factor nonzeros, updates, bytes) {exact:?} on a reduced pattern of {} entries (planned support: {}) exceeds the admitted resource envelope {admission:?}",
                    reduced_pattern_entries,
                    summary.planned_reduced_support.is_some(),
                ),
                &mut before_capacity,
            ).await;
        }
    }
    let factor = prepared
        .admissibility()
        .and_then(|evidence| evidence.factor())
        .map(|factor| factor.0.clone());
    // The elastic rigidity proof guarantees a node-major three-DOF solid; when the reduction keeps
    // whole nodes, multigrid aggregates nodal 3×3 blocks.
    let elastic = prepared
        .admissibility()
        .is_some_and(|evidence| evidence.coercivity().proved_by_elastic_rigidity());
    let block_size = if elastic && keeps_whole_nodes(reduction) {
        3
    } else {
        1
    };
    let strategy = shared_sparse::resolve_spd_strategy(
        kind,
        prepared.dof_count(),
        prepared.matrix().values().len(),
        block_size,
        factor.is_some(),
    );
    let device = shared_sparse::multigrid_device(requested_backend, strategy);
    let outcome = if requested_backend == LinearAlgebraBackendKind::CpuReference
        && strategy == shared_sparse::SpdStrategy::Cholesky
        && factor.is_some()
    {
        let arrays = shared_sparse::CsrArrays::of(prepared.matrix());
        shared_sparse::solve_cholesky_with_owner(
            arrays.view(),
            prepared.rhs(),
            reduced_guess.as_deref(),
            tolerance,
            factor.as_deref(),
            &mut before_capacity,
        )
        .await
        .map_err(|error| error.map_numerical(ScopedSpmvError::Usage))?
    } else if allow_sync_other_strategy {
        shared_sparse::solve_spd(shared_sparse::SpdSolveRequest {
            matrix: prepared.matrix(),
            rhs: prepared.rhs(),
            initial_guess: reduced_guess.as_deref(),
            relative_tolerance: tolerance,
            plan,
            strategy,
            block_size,
            factor: factor.as_deref(),
            device,
        })
        .map_err(ScopedSpmvError::Usage)?
    } else {
        return reject_affine_with_owner(
            summary,
            "awaited affine solve requires a CPU Cholesky factor".into(),
            &mut before_capacity,
        )
        .await;
    };
    if outcome.cancelled {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_SOLVER_CANCELLED".into(),
            severity: FeaDiagnosticSeverity::Warning,
            message: "affine reduced solve cancelled".into(),
        });
    } else if !outcome.converged {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_AFFINE_CG_NOT_CONVERGED".into(),
            severity: FeaDiagnosticSeverity::Warning,
            message: format!(
                "affine reduced {} did not reach tolerance {tolerance}: {}",
                strategy.label(),
                outcome
                    .failure
                    .as_deref()
                    .unwrap_or("residual above tolerance"),
            ),
        });
    }
    diagnostics.push(FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".into(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "{} space=affine_reduced block_size={block_size}",
            strategy.label()
        ),
    });
    if outcome.device_preconditioner {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_SOLVER_DEVICE_PRECONDITIONER".into(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "solver=flexible_cg preconditioner=device_amg_inner_cg inner_steps={} inner_precision=f32 outer_precision=f64",
                shared_sparse::DEVICE_INNER_CG_STEPS
            ),
        });
    } else if device.is_some() {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_SOLVER_DEVICE_FALLBACK".into(),
            severity: FeaDiagnosticSeverity::Info,
            message: "the provider could not hold the multigrid hierarchy; the host multigrid ran"
                .into(),
        });
    }
    let (solver_method, preconditioner) = strategy.provenance();
    let expansion_work = affine_expand_work(reduction, Some(plan))?;
    before_capacity(NumericalCapacityPhase::AffineExpansion, expansion_work)
        .await
        .map_err(NumericalCapacityError::Admission)?;
    let solution = expand(reduction, &outcome.solution, Some(plan))?;
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
        solver_backend: if outcome.device_preconditioner {
            LinearAlgebraBackendKind::Accelerator
        } else {
            LinearAlgebraBackendKind::CpuReference
        },
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution,
        solver_method,
        preconditioner,
        diagnostics,
    })
}

/// The reduction keeps every node's three DOFs or none, so reduced DOFs form nodal triples.
fn keeps_whole_nodes(reduction: &AffineConstraintReduction) -> bool {
    reduction.free_dof_indices.len() % 3 == 0
        && reduction.free_dof_indices.chunks_exact(3).all(|triple| {
            triple[0] % 3 == 0 && triple[1] == triple[0] + 1 && triple[2] == triple[0] + 2
        })
}

async fn reject_affine_with_owner<O, F, A>(
    summary: &StructuralLinearSystemView<'_>,
    message: String,
    before_capacity: &mut O,
) -> Result<LinearSolveResult, crate::algebra::coercivity::NumericalCapacityError<ScopedSpmvError, A>>
where
    O: FnMut(crate::algebra::NumericalCapacityPhase, OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
{
    let work = affine_rejection_norm_work(summary)?;
    before_capacity(
        crate::algebra::NumericalCapacityPhase::AffineRejectedNorm,
        work,
    )
    .await
    .map_err(crate::algebra::coercivity::NumericalCapacityError::Admission)?;
    Ok(rejected_affine_system(summary, &message)?)
}

fn affine_rejection_norm_work(
    summary: &StructuralLinearSystemView<'_>,
) -> Result<OperationCount, ScopedSpmvError> {
    affine_count(summary.operator.load().len())?
        .checked_mul(2)
        .map_err(|error| ScopedSpmvError::Usage(error.into()))
}

fn rejected_affine_system(
    summary: &StructuralLinearSystemView<'_>,
    message: &str,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    let norm_work = affine_rejection_norm_work(summary)?;
    let residual_norm = measured_affine(norm_work, || {
        Ok(summary
            .operator
            .load()
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt())
    })?;
    Ok(LinearSolveResult {
        spectral_estimate: None,
        iterations: 0,
        residual_norm,
        relative_residual: None,
        converged: false,
        host_sync_count: 0,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution: vec![0.0; summary.dof_count],
        solver_method: SolverMethod::SparseDirectCholesky,
        preconditioner: SolverPreconditioner::None,
        diagnostics: vec![FeaDiagnostic {
            code: "FEA_INVALID_AFFINE_SOLVER_ADMISSION".into(),
            severity: FeaDiagnosticSeverity::Error,
            message: message.into(),
        }],
    })
}

fn expand(
    reduction: &AffineConstraintReduction,
    reduced: &[f64],
    plan: Option<LocalExecutionPlan>,
) -> Result<Vec<f64>, ScopedSpmvError> {
    let work = affine_expand_work(reduction, plan)?;
    measured_affine(work, || {
        let mut full = expand_linear(reduction, reduced, plan)?;
        if let Some(plan) = plan.filter(|plan| !plan.is_serial()) {
            crate::algebra::axpy_scoped(1.0, &reduction.offset, &mut full, plan)?;
        } else {
            for (value, offset) in full.iter_mut().zip(&reduction.offset) {
                *value += offset;
            }
        }
        Ok(full)
    })
}

pub(crate) fn affine_expand_work(
    reduction: &AffineConstraintReduction,
    plan: Option<LocalExecutionPlan>,
) -> Result<OperationCount, ScopedSpmvError> {
    let dot_work = affine_recovery_work(reduction)?;
    let offset_work = affine_count(reduction.full_dof_count)?
        .checked_mul(if plan.is_some_and(|plan| !plan.is_serial()) {
            2
        } else {
            1
        })
        .map_err(|error| ScopedSpmvError::Usage(error.into()))?;
    let work = dot_work
        .checked_add(offset_work)
        .map_err(|error| ScopedSpmvError::Usage(error.into()))?;
    Ok(work)
}

/// `T u` without the affine offset: one recovery dot per full DOF.
pub(crate) fn affine_recovery_work(
    reduction: &AffineConstraintReduction,
) -> Result<OperationCount, ScopedSpmvError> {
    reduction
        .recovery_row_offsets
        .windows(2)
        .try_fold(OperationCount::ZERO, |total, row| {
            let terms = u64::try_from(row[1] - row[0])
                .map_err(|_| ScopedSpmvError::Usage(LedgerError::CountOverflow))?;
            let row_work = OperationCount::direct_real_dot(terms)
                .map_err(|error| ScopedSpmvError::Usage(error.into()))?;
            total
                .checked_add(row_work)
                .map_err(|error| ScopedSpmvError::Usage(error.into()))
        })
}

/// Unmetered `T u`; callers reserve [`affine_recovery_work`] around it.
pub(crate) fn expand_linear(
    reduction: &AffineConstraintReduction,
    reduced: &[f64],
    plan: Option<LocalExecutionPlan>,
) -> Result<Vec<f64>, ScopedSpmvError> {
    let mut full = vec![0.0; reduction.full_dof_count];
    if let Some(plan) = plan.filter(|plan| !plan.is_serial()) {
        let cancellation = crate::progress::cancellation_token();
        crate::planning::resource::execute_scoped_slices_mut(plan, &mut full, |chunk, output| {
            for (offset, value) in output.iter_mut().enumerate() {
                if cancellation.is_cancelled() {
                    return Err(crate::algebra::SpmvWorkError::Cancelled);
                }
                *value = recovery_row(reduction, reduced, chunk.start() as usize + offset);
            }
            Ok(())
        })?;
    } else {
        for (row, value) in full.iter_mut().enumerate() {
            *value = recovery_row(reduction, reduced, row);
        }
    }
    Ok(full)
}

fn recovery_row(reduction: &AffineConstraintReduction, reduced: &[f64], row: usize) -> f64 {
    let start = reduction.recovery_row_offsets[row];
    let end = reduction.recovery_row_offsets[row + 1];
    (start..end)
        .map(|entry| {
            reduction.recovery_coefficients[entry]
                * reduced[reduction.recovery_column_indices[entry]]
        })
        .sum()
}

pub(super) fn solve_affine_constraint_system_reusing(
    summary: &StructuralLinearSystemView<'_>,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    _algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    reuse: &mut crate::algebra::LinearPreparationReuse,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    let mut settings =
        ScopedLinearSolveSettings::with_default_tolerance(LocalExecutionPlan::serial(), None);
    settings.preparation = Some(reuse);
    solve_affine_constraint_system_with_settings(
        summary,
        reduction,
        requested_backend,
        initial_guess,
        settings,
    )
}

#[cfg(test)]
mod tests;
