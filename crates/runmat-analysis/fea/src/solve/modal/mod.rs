use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    planning::resource::LocalExecutionPlan,
    solve::backend::kind::LinearAlgebraBackendKind,
    ComputeBackend,
};

mod capacity;
mod diagnostics;
mod math;
mod truncation;

use diagnostics::push_modal_quality_diagnostics;
mod preparation;
pub use preparation::solve_modal_system;
use truncation::resolve_truncation_boundary;

pub(crate) const MODAL_BOUNDARY_PROBE_WIDTH: usize = 4;
/// Block width of the modal shift-invert Lanczos.
pub(crate) const MODAL_LANCZOS_BLOCK_SIZE: usize = runmat_sparse_solvers::eigen::DEFAULT_BLOCK_SIZE;
/// Modes past the last requested one that the admitted subspace keeps room for when they share
/// its eigenvalue. A Lanczos block of `p` vectors resolves an eigenvalue of multiplicity up to
/// `p`, so a resolved cluster at the truncation boundary adds at most `p − 1` modes.
pub(crate) const MODAL_BOUNDARY_CLUSTER_ALLOWANCE: usize = MODAL_LANCZOS_BLOCK_SIZE - 1;

/// Eigensolver options for one modal search of `subspace_dimension` vectors. The resource
/// planner sizes the search's working set from these same options.
pub(crate) fn modal_eigen_options(
    subspace_dimension: usize,
    convergence: crate::pipeline::modal_governance::ModalConvergencePolicy,
) -> runmat_sparse_solvers::eigen::EigenOptions {
    runmat_sparse_solvers::eigen::EigenOptions {
        count: subspace_dimension,
        shift: Some(0.0),
        block_size: MODAL_LANCZOS_BLOCK_SIZE,
        tolerance: (0.1 * convergence.relative_residual_tolerance()).min(1.0e-10),
        max_restarts: convergence.maximum_subspace_iterations().max(1),
        sturm_check: true,
        ..Default::default()
    }
}
pub(crate) const MODAL_MASS_ORTHOGONALIZATION_PASSES: usize = 2;
type ModalCandidate = (f64, Vec<f64>, f64);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalSolveResult {
    pub converged: bool,
    pub frequencies_hz: Vec<f64>,
    pub mode_shapes: Vec<Vec<f64>>,
    pub residual_norms: Vec<f64>,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub solver_method: SolverMethod,
    pub solver_backend: LinearAlgebraBackendKind,
    pub preconditioner: SolverPreconditioner,
    pub solver_host_sync_count: u32,
    pub device_apply_k_count: u32,
    pub device_apply_k_attempt_count: u32,
    pub accelerator_evidence: Vec<crate::solve::accelerator::AcceleratorSolveEvidence>,
    pub available_mode_count: usize,
    pub boundary_probe_count: usize,
    pub boundary_probe_frequency_hz: Option<f64>,
    /// The admitted subspace dimension when the search reached it with the truncation boundary
    /// still unresolved: the boundary cluster needs more vectors than the plan reserved.
    #[serde(default)]
    pub boundary_unresolved_at_capacity: Option<usize>,
}

pub(crate) fn solve_modal_system_reusing(
    summary: &AssemblySummary,
    requested_mode_count: usize,
    eigenspace_relative_tolerance: f64,
    convergence: crate::pipeline::modal_governance::ModalConvergencePolicy,
    backend: ComputeBackend,
    reuse: &mut crate::algebra::LinearPreparationReuse,
) -> ModalSolveResult {
    if summary.dof_count == 0 || requested_mode_count == 0 {
        return ModalSolveResult {
            converged: false,
            frequencies_hz: Vec::new(),
            mode_shapes: Vec::new(),
            residual_norms: Vec::new(),
            diagnostics: vec![FeaDiagnostic {
                code: "FEA_MODAL_EMPTY_SYSTEM".to_string(),
                severity: FeaDiagnosticSeverity::Warning,
                message:
                    "modal solve skipped because assembled system has zero DOFs or requested mode_count is zero"
                        .to_string(),
            }],
            solver_method: SolverMethod::ShiftInvertBlockLanczos,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            preconditioner: SolverPreconditioner::None,
            solver_host_sync_count: 0,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            accelerator_evidence: Vec::new(),
            available_mode_count: summary.dof_count,
            boundary_probe_count: 0,
            boundary_probe_frequency_hz: None,
            boundary_unresolved_at_capacity: None,
        };
    }

    let prepared_modal_result = match summary.constraint_matrix.as_ref() {
        Some(constraints)
            if constraints
                .right_hand_side
                .iter()
                .all(|value| *value == 0.0) =>
        {
            constraints.affine_reduction().and_then(|reduction| {
                summary
                    .operator
                    .prepare_modal_with_reduction(reduction)
                    .map_err(|error| {
                        crate::constraints::ConstraintAlgebraError::InvalidMatrix(error.to_string())
                    })
            })
        }
        Some(_) => Err(crate::constraints::ConstraintAlgebraError::InvalidMatrix(
            "modal constraints must be homogeneous".into(),
        )),
        None => summary.operator.prepare_modal().map_err(|error| {
            crate::constraints::ConstraintAlgebraError::InvalidMatrix(error.to_string())
        }),
    };
    let prepared_modal = match prepared_modal_result {
        Ok(prepared) => prepared,
        Err(error) => {
            return ModalSolveResult {
                converged: false,
                frequencies_hz: Vec::new(),
                mode_shapes: Vec::new(),
                residual_norms: Vec::new(),
                diagnostics: vec![FeaDiagnostic {
                    code: "FEA_MODAL_PREPARATION_FAILED".to_string(),
                    severity: FeaDiagnosticSeverity::Error,
                    message: format!("modal free-DOF reduction failed: {error:?}"),
                }],
                solver_method: SolverMethod::ShiftInvertBlockLanczos,
                solver_backend: LinearAlgebraBackendKind::CpuReference,
                preconditioner: SolverPreconditioner::None,
                solver_host_sync_count: 0,
                device_apply_k_count: 0,
                device_apply_k_attempt_count: 0,
                accelerator_evidence: Vec::new(),
                available_mode_count: summary.dof_count,
                boundary_probe_count: 0,
                boundary_probe_frequency_hz: None,
                boundary_unresolved_at_capacity: None,
            };
        }
    };
    let reduced_operator = match prepared_modal.as_assembled() {
        Ok(operator) => operator,
        Err(error) => {
            return ModalSolveResult {
                converged: false,
                frequencies_hz: Vec::new(),
                mode_shapes: Vec::new(),
                residual_norms: Vec::new(),
                diagnostics: vec![FeaDiagnostic {
                    code: "FEA_MODAL_PREPARATION_FAILED".to_string(),
                    severity: FeaDiagnosticSeverity::Error,
                    message: format!("modal reduced system validation failed: {error:?}"),
                }],
                solver_method: SolverMethod::ShiftInvertBlockLanczos,
                solver_backend: LinearAlgebraBackendKind::CpuReference,
                preconditioner: SolverPreconditioner::None,
                solver_host_sync_count: 0,
                device_apply_k_count: 0,
                device_apply_k_attempt_count: 0,
                accelerator_evidence: Vec::new(),
                available_mode_count: summary.dof_count,
                boundary_probe_count: 0,
                boundary_probe_frequency_hz: None,
                boundary_unresolved_at_capacity: None,
            };
        }
    };
    let prepared = match reduced_operator.prepare_linear_reusing(reuse) {
        Ok(prepared) => prepared,
        Err(error) => {
            return ModalSolveResult {
                converged: false,
                frequencies_hz: Vec::new(),
                mode_shapes: Vec::new(),
                residual_norms: Vec::new(),
                diagnostics: vec![FeaDiagnostic {
                    code: "FEA_MODAL_SOLVER_ADMISSION_FAILED".into(),
                    severity: FeaDiagnosticSeverity::Error,
                    message: format!(
                        "modal stiffness admission failed on the constraint space: {error}"
                    ),
                }],
                solver_method: SolverMethod::ShiftInvertBlockLanczos,
                solver_backend: LinearAlgebraBackendKind::CpuReference,
                preconditioner: SolverPreconditioner::None,
                solver_host_sync_count: 0,
                device_apply_k_count: 0,
                device_apply_k_attempt_count: 0,
                accelerator_evidence: Vec::new(),
                available_mode_count: reduced_operator.dof_count(),
                boundary_probe_count: 0,
                boundary_probe_frequency_hz: None,
                boundary_unresolved_at_capacity: None,
            }
        }
    };
    let mut linear_diagnostics = Vec::new();
    // Strict placement fails closed at the modal entry; here the shared admission only supplies
    // the fallback warning for a GPU request.
    if let Ok(admission) = crate::contracts::accelerator_admission::admit_accelerator_with_policy(
        crate::contracts::accelerator_admission::AcceleratorFamily::StructuralModal,
        backend,
        crate::contracts::accelerator_admission::AcceleratorAdmissionPolicy::FallbackToCpu,
    ) {
        linear_diagnostics.extend(admission.fallback);
    }
    let unconstrained_count = reduced_operator.dof_count();
    let target_mode_count = requested_mode_count.min(unconstrained_count);
    let maximum_subspace_dimension =
        match convergence.admitted_subspace_dimension(unconstrained_count, target_mode_count) {
            Ok(capacity) => capacity,
            Err(error) => return capacity::failure(error, unconstrained_count),
        };
    // The Cholesky factor that proved K coercive is the σ = 0 shift-invert operator.
    let factor = prepared
        .admissibility()
        .and_then(|evidence| evidence.factor())
        .map(|factor| factor.0.clone());
    let stiffness =
        crate::solve::shared_sparse::CsrArrays::of(reduced_operator.operators().stiffness());
    let mass = crate::solve::shared_sparse::CsrArrays::of(reduced_operator.operators().mass());
    let mut modes: Vec<ModalCandidate> = Vec::with_capacity(target_mode_count.saturating_add(4));
    let mut boundary_probe_count = 0usize;
    let mut complete_boundary_eigenspace = false;
    let mut capacity_reached = false;
    let mut boundary_probe_frequency_hz = None;
    let mut subspace_dimension = target_mode_count
        .saturating_add(MODAL_BOUNDARY_PROBE_WIDTH)
        .min(maximum_subspace_dimension);
    let mut subspace_iteration_count = 0usize;
    let mut subspace_maximum_residual = f64::INFINITY;
    let mut subspace_converged = false;
    let mut sturm = None;
    let tolerance = convergence.relative_residual_tolerance();
    while subspace_dimension > 0 && !crate::is_fea_cancelled() {
        let options = modal_eigen_options(subspace_dimension, convergence);
        let solved = crate::solve::shared_sparse::with_plan(LocalExecutionPlan::serial(), || {
            let (k, m) = (stiffness.view(), mass.view());
            match factor.as_deref() {
                Some(solver) => runmat_sparse_solvers::eigen::modal_with_factor(
                    k,
                    m,
                    solver,
                    0.0,
                    &options,
                    crate::solve::shared_sparse::control(),
                ),
                None => runmat_sparse_solvers::eigen::modal(
                    k,
                    m,
                    &options,
                    crate::solve::shared_sparse::control(),
                ),
            }
        });
        let result = match solved {
            Ok(result) => result,
            Err(error) => {
                linear_diagnostics.push(FeaDiagnostic {
                    code: "FEA_MODAL_EIGENSOLVER_FAILED".into(),
                    severity: FeaDiagnosticSeverity::Error,
                    message: error.to_string(),
                });
                break;
            }
        };
        subspace_iteration_count = subspace_iteration_count.saturating_add(result.restarts.max(1));
        subspace_maximum_residual = result.residuals.iter().copied().fold(0.0, f64::max);
        subspace_converged = result.converged && subspace_maximum_residual <= tolerance;
        sturm = result.sturm;
        let n = unconstrained_count;
        let mut candidates = Vec::with_capacity(result.values.len());
        for (index, eigenvalue) in result.values.iter().copied().enumerate() {
            let vector = &result.vectors[index * n..(index + 1) * n];
            let Ok(expanded) = prepared_modal.expand(vector) else {
                candidates.clear();
                break;
            };
            candidates.push((
                eigenvalue.max(0.0).sqrt() / (2.0 * std::f64::consts::PI),
                expanded,
                result.residuals[index],
            ));
        }
        let exhausted = subspace_dimension == unconstrained_count;
        if !subspace_converged {
            modes = candidates;
            break;
        }
        if let Some(boundary) = resolve_truncation_boundary(
            &candidates,
            target_mode_count,
            eigenspace_relative_tolerance,
            exhausted,
        ) {
            boundary_probe_count = boundary.probe_count;
            boundary_probe_frequency_hz = boundary.probe_frequency_hz;
            modes = boundary.retained;
            complete_boundary_eigenspace = true;
            break;
        }
        if exhausted || subspace_dimension == maximum_subspace_dimension {
            capacity_reached = !exhausted;
            modes = candidates;
            break;
        }
        subspace_dimension = subspace_dimension
            .saturating_add(MODAL_BOUNDARY_PROBE_WIDTH)
            .min(maximum_subspace_dimension);
    }
    if let Some(sturm) = sturm {
        linear_diagnostics.push(FeaDiagnostic {
            code: "FEA_MODAL_STURM_CHECK".into(),
            severity: if sturm.complete() {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Error
            },
            message: format!(
                "eigenvalues_below_shift={} returned_below_shift={} shift={} complete={}",
                sturm.counted,
                sturm.returned,
                sturm.shift,
                sturm.complete()
            ),
        });
        if !sturm.complete() {
            subspace_converged = false;
        }
    }
    let solver_backend = LinearAlgebraBackendKind::CpuReference;
    let (solver_host_sync_count, device_apply_k_count, device_apply_k_attempt_count) =
        (0u32, 0u32, 0u32);
    let fallback_apply_count = 0u32;
    let accelerator_evidence = Vec::new();
    let unconstrained: Vec<usize> = (0..unconstrained_count).collect();

    modes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut frequencies_hz = Vec::with_capacity(modes.len());
    let mut mode_shapes = Vec::with_capacity(modes.len());
    let mut residual_norms = Vec::with_capacity(modes.len());
    for (freq_hz, shape, residual) in modes {
        frequencies_hz.push(freq_hz);
        mode_shapes.push(shape);
        residual_norms.push(residual);
    }

    let converged = subspace_converged
        && !crate::is_fea_cancelled()
        && frequencies_hz.len() >= target_mode_count
        && complete_boundary_eigenspace;
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_MODAL_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message:
            "solver=shift_invert_block_lanczos shift=0 factor=sparse_cholesky sturm_check=true"
                .to_string(),
    }];
    diagnostics.extend(linear_diagnostics);
    diagnostics.push(FeaDiagnostic {
        code: "FEA_MODAL_CONVERGENCE".to_string(),
        severity: if converged {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "mode_count_requested={} mode_count_solved={} subspace_dimension={} iteration_count={} maximum_generalized_residual={} converged={}",
            requested_mode_count,
            frequencies_hz.len(),
            subspace_dimension,
            subspace_iteration_count,
            subspace_maximum_residual,
            converged
        ),
    });

    if !complete_boundary_eigenspace
        && subspace_dimension == maximum_subspace_dimension
        && maximum_subspace_dimension < unconstrained.len()
    {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_MODAL_SUBSPACE_CAPACITY".into(),
            severity: FeaDiagnosticSeverity::Error,
            message: format!(
                "modal truncation boundary remains unproven at admitted subspace capacity: requested={target_mode_count} capacity={maximum_subspace_dimension} available={}",
                unconstrained.len(),
            ),
        });
    }
    push_modal_quality_diagnostics(
        &mut diagnostics,
        &summary.operator,
        &frequencies_hz,
        &mode_shapes,
        &residual_norms,
        convergence.relative_residual_tolerance(),
    );
    diagnostics.push(FeaDiagnostic {
        code: "FEA_MODAL_COST".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "subspace_iteration_count={} fallback_apply_count={} device_apply_k_count={} device_apply_k_attempt_count={} host_sync_count={}",
            subspace_iteration_count,
            fallback_apply_count,
            device_apply_k_count,
            device_apply_k_attempt_count,
            solver_host_sync_count,
        ),
    });

    ModalSolveResult {
        converged,
        frequencies_hz,
        mode_shapes,
        residual_norms,
        diagnostics,
        solver_method: SolverMethod::ShiftInvertBlockLanczos,
        solver_backend,
        preconditioner: SolverPreconditioner::SparseCholesky,
        solver_host_sync_count,
        device_apply_k_count,
        device_apply_k_attempt_count,
        accelerator_evidence,
        available_mode_count: unconstrained.len(),
        boundary_probe_count,
        boundary_probe_frequency_hz,
        boundary_unresolved_at_capacity: capacity_reached.then_some(maximum_subspace_dimension),
    }
}
