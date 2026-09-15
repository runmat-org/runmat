use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::{
        accelerator::prepare_accelerator_linear_system,
        backend::kind::LinearAlgebraBackendKind,
        preconditioner::{build_spd_preconditioner, SpdPreconditionerKind},
    },
    ComputeBackend,
};

mod diagnostics;
mod linear_solve;
mod math;
mod subspace;
mod truncation;

use diagnostics::push_modal_quality_diagnostics;
use linear_solve::{solve_k_system_cg, CgSolveOptions};
use subspace::solve_lowest_eigenspace;
use truncation::resolve_truncation_boundary;

pub(crate) const MODAL_BOUNDARY_PROBE_WIDTH: usize = 4;
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
}

pub fn solve_modal_system(
    summary: &AssemblySummary,
    requested_mode_count: usize,
    eigenspace_relative_tolerance: f64,
    convergence: crate::pipeline::modal_governance::ModalConvergencePolicy,
    backend: ComputeBackend,
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
            solver_method: SolverMethod::MatrixFreeSubspaceIteration,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            preconditioner: SolverPreconditioner::None,
            solver_host_sync_count: 0,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            accelerator_evidence: Vec::new(),
            available_mode_count: summary.dof_count,
            boundary_probe_count: 0,
            boundary_probe_frequency_hz: None,
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
                solver_method: SolverMethod::MatrixFreeSubspaceIteration,
                solver_backend: LinearAlgebraBackendKind::CpuReference,
                preconditioner: SolverPreconditioner::None,
                solver_host_sync_count: 0,
                device_apply_k_count: 0,
                device_apply_k_attempt_count: 0,
                accelerator_evidence: Vec::new(),
                available_mode_count: summary.dof_count,
                boundary_probe_count: 0,
                boundary_probe_frequency_hz: None,
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
                solver_method: SolverMethod::MatrixFreeSubspaceIteration,
                solver_backend: LinearAlgebraBackendKind::CpuReference,
                preconditioner: SolverPreconditioner::None,
                solver_host_sync_count: 0,
                device_apply_k_count: 0,
                device_apply_k_attempt_count: 0,
                accelerator_evidence: Vec::new(),
                available_mode_count: summary.dof_count,
                boundary_probe_count: 0,
                boundary_probe_frequency_hz: None,
            };
        }
    };
    if let Err(error) = reduced_operator.prepare_linear() {
        return ModalSolveResult {
            converged: false,
            frequencies_hz: Vec::new(),
            mode_shapes: Vec::new(),
            residual_norms: Vec::new(),
            diagnostics: vec![FeaDiagnostic {
                code: "FEA_MODAL_SOLVER_ADMISSION_FAILED".into(),
                severity: FeaDiagnosticSeverity::Error,
                message: format!(
                    "modal stiffness is not coercive on the constraint space: {error}"
                ),
            }],
            solver_method: SolverMethod::MatrixFreeSubspaceIteration,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            preconditioner: SolverPreconditioner::None,
            solver_host_sync_count: 0,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            accelerator_evidence: Vec::new(),
            available_mode_count: reduced_operator.dof_count(),
            boundary_probe_count: 0,
            boundary_probe_frequency_hz: None,
        };
    }
    let mut solve_summary = summary.clone();
    solve_summary.dof_count = reduced_operator.dof_count();
    solve_summary.constrained_dof_count = 0;
    solve_summary.operator = reduced_operator;
    let accelerator_requested = backend == ComputeBackend::Gpu;
    let mut solver_backend = LinearAlgebraBackendKind::CpuReference;
    let mut solver_host_sync_count = 0u32;
    let mut device_apply_k_count = 0u32;
    let mut device_apply_k_attempt_count = 0u32;

    let unconstrained: Vec<usize> = (0..solve_summary.operator.dof_count()).collect();
    let target_mode_count = requested_mode_count.min(unconstrained.len());

    let mut modes: Vec<ModalCandidate> = Vec::with_capacity(target_mode_count.saturating_add(4));
    let mut boundary_probe_count = 0usize;
    let mut complete_boundary_eigenspace = false;
    let mut boundary_probe_frequency_hz = None;
    let mut accelerator_admission_fallback = None;
    let prepared_accelerator_system = if accelerator_requested {
        let prepared = prepare_accelerator_linear_system(
            solve_summary
                .operator
                .prepare_linear()
                .expect("modal coercivity was checked before accelerator preparation"),
            64,
        );
        match prepared {
            Ok(prepared) => Some(prepared),
            Err(error) if error.permits_cpu_fallback() => {
                accelerator_admission_fallback = Some(error.message);
                None
            }
            Err(error) => {
                return ModalSolveResult {
                    converged: false,
                    frequencies_hz: Vec::new(),
                    mode_shapes: Vec::new(),
                    residual_norms: Vec::new(),
                    diagnostics: vec![FeaDiagnostic {
                        code: "FEA_MODAL_ACCELERATOR_PREPARATION_FAILED".to_string(),
                        severity: FeaDiagnosticSeverity::Error,
                        message: error.message,
                    }],
                    solver_method: SolverMethod::MatrixFreeSubspaceIteration,
                    solver_backend: LinearAlgebraBackendKind::Accelerator,
                    preconditioner: SolverPreconditioner::Jacobi,
                    solver_host_sync_count: 0,
                    device_apply_k_count: 0,
                    device_apply_k_attempt_count: 0,
                    accelerator_evidence: error
                        .evidence
                        .into_iter()
                        .map(|evidence| *evidence)
                        .collect(),
                    available_mode_count: unconstrained.len(),
                    boundary_probe_count: 0,
                    boundary_probe_frequency_hz: None,
                };
            }
        }
    } else {
        None
    };
    let mut linear_diagnostics = Vec::new();
    let (cpu_preconditioner_kind, cpu_preconditioner) = if prepared_accelerator_system.is_none() {
        let prepared = solve_summary
            .operator
            .prepare_linear()
            .expect("modal coercivity was checked before preconditioner construction");
        match build_spd_preconditioner(&prepared, SpdPreconditionerKind::Ilu0) {
            Ok(preconditioner) => (Some(SolverPreconditioner::Ilu0), Some(preconditioner)),
            Err(error) => {
                linear_diagnostics.push(FeaDiagnostic {
                    code: "FEA_MODAL_ILU0_FALLBACK".into(),
                    severity: FeaDiagnosticSeverity::Warning,
                    message: format!(
                        "ILU(0) modal preconditioner was unavailable; using Jacobi: {error}"
                    ),
                });
                match build_spd_preconditioner(&prepared, SpdPreconditionerKind::Jacobi) {
                    Ok(preconditioner) => {
                        (Some(SolverPreconditioner::Jacobi), Some(preconditioner))
                    }
                    Err(error) => {
                        linear_diagnostics.push(FeaDiagnostic {
                            code: "FEA_MODAL_PRECONDITIONER_FAILED".into(),
                            severity: FeaDiagnosticSeverity::Error,
                            message: error.to_string(),
                        });
                        (None, None)
                    }
                }
            }
        }
    } else {
        (Some(SolverPreconditioner::Jacobi), None)
    };
    if prepared_accelerator_system.is_none() && cpu_preconditioner.is_none() {
        return ModalSolveResult {
            converged: false,
            frequencies_hz: Vec::new(),
            mode_shapes: Vec::new(),
            residual_norms: Vec::new(),
            diagnostics: linear_diagnostics,
            solver_method: SolverMethod::MatrixFreeSubspaceIteration,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            preconditioner: SolverPreconditioner::None,
            solver_host_sync_count: 0,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            accelerator_evidence: Vec::new(),
            available_mode_count: unconstrained.len(),
            boundary_probe_count: 0,
            boundary_probe_frequency_hz: None,
        };
    }
    let mut fallback_apply_count = 0u32;
    let mut accelerator_evidence = Vec::new();
    let mut subspace_dimension = target_mode_count
        .saturating_add(MODAL_BOUNDARY_PROBE_WIDTH)
        .min(unconstrained.len());
    let mut subspace_iteration_count = 0usize;
    let mut subspace_maximum_residual = f64::INFINITY;
    let mut subspace_converged = false;
    while subspace_dimension > 0 && !crate::is_fea_cancelled() {
        let result = solve_lowest_eigenspace(
            &solve_summary.operator,
            subspace_dimension,
            target_mode_count.saturating_add(1),
            convergence.relative_residual_tolerance(),
            convergence.maximum_subspace_iterations(),
            |rhs, initial_guess| {
                if crate::is_fea_cancelled() {
                    return None;
                }
                let attempt = solve_k_system_cg(
                    &solve_summary.operator,
                    rhs,
                    CgSolveOptions {
                        max_iters: 64,
                        tol: 1.0e-8,
                        prepared_accelerator_system: prepared_accelerator_system.as_ref(),
                        cpu_preconditioner: cpu_preconditioner.as_deref(),
                        initial_guess,
                    },
                );
                if let Some(solve) = &attempt.accelerator {
                    solver_backend = solve.solver_backend;
                    solver_host_sync_count =
                        solver_host_sync_count.saturating_add(solve.host_sync_count);
                    device_apply_k_count =
                        device_apply_k_count.saturating_add(solve.device_apply_k_count);
                    device_apply_k_attempt_count = device_apply_k_attempt_count
                        .saturating_add(solve.device_apply_k_attempt_count);
                    fallback_apply_count = fallback_apply_count.saturating_add(
                        solve
                            .device_apply_k_attempt_count
                            .saturating_sub(solve.device_apply_k_count),
                    );
                    linear_diagnostics.extend(solve.diagnostics.clone());
                    accelerator_evidence.extend(solve.accelerator_evidence.clone());
                }
                attempt.converged.then_some(attempt.vector)
            },
        );
        subspace_iteration_count = subspace_iteration_count.saturating_add(result.iteration_count);
        subspace_maximum_residual = result.maximum_residual;
        subspace_converged = result.converged;
        let mut candidates = Vec::with_capacity(result.candidates.len());
        for candidate in result.candidates {
            let Ok(expanded) = prepared_modal.expand(&candidate.vector) else {
                candidates.clear();
                break;
            };
            candidates.push((
                candidate.eigenvalue.sqrt() / (2.0 * std::f64::consts::PI),
                expanded,
                candidate.residual,
            ));
        }
        let exhausted = subspace_dimension == unconstrained.len();
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
        if exhausted {
            modes = candidates;
            break;
        }
        subspace_dimension = subspace_dimension
            .saturating_add(MODAL_BOUNDARY_PROBE_WIDTH)
            .min(unconstrained.len());
    }

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
        message: "solver=matrix_free_subspace_iteration inverse_k=true projection=rayleigh_ritz"
            .to_string(),
    }];
    if let Some(reason) = accelerator_admission_fallback {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_MODAL_ACCELERATOR_ADMISSION_FALLBACK".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: format!(
                "native sparse execution was rejected before provider allocation; using cpu_reference: {reason}"
            ),
        });
    }
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
        solver_method: SolverMethod::MatrixFreeSubspaceIteration,
        solver_backend,
        preconditioner: if solver_backend == LinearAlgebraBackendKind::Accelerator {
            SolverPreconditioner::Jacobi
        } else {
            cpu_preconditioner_kind.unwrap_or(SolverPreconditioner::None)
        },
        solver_host_sync_count,
        device_apply_k_count,
        device_apply_k_attempt_count,
        accelerator_evidence,
        available_mode_count: unconstrained.len(),
        boundary_probe_count,
        boundary_probe_frequency_hz,
    }
}
