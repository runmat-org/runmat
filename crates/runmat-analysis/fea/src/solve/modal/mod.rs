use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::{
    algebra::{apply_k, apply_m},
    assembly::AssemblySummary,
    contracts::SolverMethod,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::{
        accelerator::prepare_accelerator_linear_system, backend::kind::LinearAlgebraBackendKind,
    },
    ComputeBackend,
};

mod diagnostics;
mod linear_solve;
mod math;

use diagnostics::push_modal_quality_diagnostics;
use linear_solve::{solve_k_system_cg, CgSolveOptions};
use math::{dot, normalize_mass, orthonormalize_mass, relative_l2_update};

const BOUNDARY_PROBE_STABILITY_COUNT: usize = 4;
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

    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(target_mode_count.saturating_add(1));
    let mut modes: Vec<ModalCandidate> = Vec::with_capacity(target_mode_count.saturating_add(1));
    let mut boundary_probe_count = 0usize;
    let mut complete_boundary_eigenspace = false;
    let mut boundary_probe_frequency_hz = None;
    let mut prepared_build_ms = 0.0_f64;
    let mut accelerator_admission_fallback = None;
    let prepared_accelerator_system = if accelerator_requested {
        let prepared_start = Instant::now();
        let prepared = prepare_accelerator_linear_system(
            solve_summary
                .operator
                .prepare_linear()
                .expect("modal coercivity was checked before accelerator preparation"),
            64,
        );
        prepared_build_ms = prepared_start.elapsed().as_secs_f64() * 1_000.0;
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
    let has_accel_provider = prepared_accelerator_system.is_some();
    let mut solve_ms = 0.0_f64;
    let mut fallback_apply_count = 0u32;
    let mut accelerator_evidence = Vec::new();
    let mut linear_diagnostics = Vec::new();
    let (max_inverse_iters, min_inverse_iters, update_tol) = if has_accel_provider {
        (6usize, 2usize, 5.0e-4)
    } else {
        (8usize, 3usize, 1.0e-4)
    };

    'modes: for mode_idx in 0..unconstrained.len() {
        if crate::is_fea_cancelled() {
            break;
        }
        let mut q = vec![0.0; solve_summary.operator.dof_count()];
        q[unconstrained[mode_idx]] = 1.0;
        normalize_mass(&solve_summary.operator, &mut q);
        let mut linear_guess: Option<Vec<f64>> = None;

        for iter in 0..max_inverse_iters {
            if crate::is_fea_cancelled() {
                break 'modes;
            }
            let q_prev = q.clone();
            let mq = apply_m(&solve_summary.operator, &q);
            let solve_start = Instant::now();
            let z = solve_k_system_cg(
                &solve_summary.operator,
                &mq,
                CgSolveOptions {
                    max_iters: 64,
                    tol: 1.0e-10,
                    prepared_accelerator_system: prepared_accelerator_system.as_ref(),
                    initial_guess: linear_guess.as_deref(),
                },
            );
            if crate::is_fea_cancelled() {
                break 'modes;
            }
            solve_ms += solve_start.elapsed().as_secs_f64() * 1_000.0;
            if let Some(solve) = z.accelerator {
                solver_backend = solve.solver_backend;
                solver_host_sync_count =
                    solver_host_sync_count.saturating_add(solve.host_sync_count);
                device_apply_k_count =
                    device_apply_k_count.saturating_add(solve.device_apply_k_count);
                device_apply_k_attempt_count =
                    device_apply_k_attempt_count.saturating_add(solve.device_apply_k_attempt_count);
                fallback_apply_count = fallback_apply_count.saturating_add(
                    solve
                        .device_apply_k_attempt_count
                        .saturating_sub(solve.device_apply_k_count),
                );
                linear_diagnostics.extend(solve.diagnostics.clone());
                accelerator_evidence.extend(solve.accelerator_evidence.clone());
                if !solve.converged {
                    break 'modes;
                }
            }
            let mut z_vec = z.vector;
            linear_guess = Some(z_vec.clone());
            orthonormalize_mass(&solve_summary.operator, &mut z_vec, &basis);
            normalize_mass(&solve_summary.operator, &mut z_vec);
            q = z_vec;

            if iter + 1 >= min_inverse_iters {
                let rel_update = relative_l2_update(&q_prev, &q);
                if rel_update <= update_tol {
                    break;
                }
            }
        }

        let kq = apply_k(&solve_summary.operator, &q);
        let mq = apply_m(&solve_summary.operator, &q);
        let lambda = (dot(&q, &kq) / dot(&q, &mq).abs().max(1.0e-12)).max(0.0);
        let freq_hz = lambda.sqrt() / (2.0 * std::f64::consts::PI);

        let residual = kq
            .iter()
            .zip(mq.iter())
            .map(|(k_value, m_value)| {
                let diff = *k_value - lambda * *m_value;
                diff * diff
            })
            .sum::<f64>()
            .sqrt();
        let kq_norm = kq
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(1.0e-12);

        let expanded = match prepared_modal.expand(&q) {
            Ok(shape) => shape,
            Err(_) => break,
        };
        basis.push(q.clone());
        modes.push((freq_hz, expanded, residual / kq_norm));
        let exhausted = mode_idx + 1 == unconstrained.len();
        if let Some(boundary) = resolve_truncation_boundary(
            &modes,
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

    let converged = !crate::is_fea_cancelled()
        && frequencies_hz.len() >= target_mode_count
        && complete_boundary_eigenspace;
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_MODAL_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: "solver=matrix_free_subspace_iteration inverse_k=true".to_string(),
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
            "mode_count_requested={} mode_count_solved={} converged={}",
            requested_mode_count,
            frequencies_hz.len(),
            converged
        ),
    });

    push_modal_quality_diagnostics(
        &mut diagnostics,
        &summary.operator,
        &frequencies_hz,
        &mode_shapes,
        &residual_norms,
    );
    diagnostics.push(FeaDiagnostic {
        code: "FEA_MODAL_COST".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "prepared_build_ms={} solve_ms={} fallback_apply_count={}",
            prepared_build_ms, solve_ms, fallback_apply_count
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
        solver_host_sync_count,
        device_apply_k_count,
        device_apply_k_attempt_count,
        accelerator_evidence,
        available_mode_count: unconstrained.len(),
        boundary_probe_count,
        boundary_probe_frequency_hz,
    }
}

struct ResolvedTruncationBoundary {
    retained: Vec<ModalCandidate>,
    probe_count: usize,
    probe_frequency_hz: Option<f64>,
}

fn resolve_truncation_boundary(
    candidates: &[ModalCandidate],
    requested_count: usize,
    tolerance: f64,
    exhausted: bool,
) -> Option<ResolvedTruncationBoundary> {
    if candidates.len() < requested_count {
        return exhausted.then(|| ResolvedTruncationBoundary {
            retained: candidates.to_vec(),
            probe_count: 0,
            probe_frequency_hz: None,
        });
    }
    let mut ordered = candidates.to_vec();
    ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut retained_count = requested_count;
    while retained_count < ordered.len()
        && crate::pipeline::modal_governance::relative_frequency_separation(
            ordered[retained_count - 1].0,
            ordered[retained_count].0,
        ) <= tolerance
    {
        retained_count += 1;
    }
    let probe_count = ordered.len() - retained_count;
    if !exhausted && probe_count < BOUNDARY_PROBE_STABILITY_COUNT {
        return None;
    }
    let probe_frequency_hz = ordered.get(retained_count).map(|candidate| candidate.0);
    ordered.truncate(retained_count);
    Some(ResolvedTruncationBoundary {
        retained: ordered,
        probe_count,
        probe_frequency_hz,
    })
}

#[cfg(test)]
mod truncation_tests {
    use super::*;

    fn candidate(frequency_hz: f64) -> ModalCandidate {
        (frequency_hz, vec![frequency_hz], 0.0)
    }

    #[test]
    fn boundary_resolution_orders_a_nonmonotonic_candidate_stream() {
        let candidates = [10.0, 1.0, 2.0, 6.0, 3.0, 5.0]
            .into_iter()
            .map(candidate)
            .collect::<Vec<_>>();
        let resolved = resolve_truncation_boundary(&candidates, 2, 1.0e-6, false).unwrap();
        assert_eq!(
            resolved
                .retained
                .iter()
                .map(|candidate| candidate.0)
                .collect::<Vec<_>>(),
            [1.0, 2.0]
        );
        assert_eq!(resolved.probe_count, 4);
        assert_eq!(resolved.probe_frequency_hz, Some(3.0));
    }

    #[test]
    fn boundary_resolution_keeps_a_complete_cluster() {
        let candidates = [8.0, 1.0, 2.0, 2.000_001, 4.0, 6.0, 10.0]
            .into_iter()
            .map(candidate)
            .collect::<Vec<_>>();
        let resolved = resolve_truncation_boundary(&candidates, 2, 1.0e-6, false).unwrap();
        assert_eq!(resolved.retained.len(), 3);
        assert_eq!(resolved.probe_frequency_hz, Some(4.0));
    }
}
