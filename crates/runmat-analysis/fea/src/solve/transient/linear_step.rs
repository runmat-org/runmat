use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use runmat_execution::Digest;

use crate::{
    algebra::{
        apply_k, apply_m, apply_prepared, AssembledRealSystem, CanonicalRealOperators,
        MatrixProperties, OperatorRole, RealSparseMatrix,
    },
    assembly::AssemblySummary,
    contracts::SolverPreconditioner,
    solve::{
        accelerator::{
            prepare_accelerator_linear_system, solve_prepared_accelerator_system,
            AcceleratorAdapterError, AcceleratorPreparedLinearSystem,
        },
        backend::kind::LinearAlgebraBackendKind,
        linear::{linear_result_from_accelerator_error, linear_result_from_accelerator_output},
        preconditioner::SpdPreconditionerKind,
    },
};

use super::TransientSolveOptions;

#[derive(Debug, Clone)]
pub(super) struct LinearStepStats {
    pub(super) solver_backend: LinearAlgebraBackendKind,
    pub(super) host_sync_count: u32,
    pub(super) device_apply_k_count: u32,
    pub(super) device_apply_k_attempt_count: u32,
    pub(super) preconditioner: SolverPreconditioner,
    pub(super) diagnostics: Vec<crate::diagnostics::FeaDiagnostic>,
    pub(super) accelerator_evidence: Option<crate::solve::accelerator::AcceleratorSolveEvidence>,
}

pub(super) struct AcceleratorStepCache<'a> {
    pub(super) prepared_systems_by_dt:
        &'a mut HashMap<AcceleratorStepCacheKey, AcceleratorPreparedLinearSystem>,
    pub(super) prepared_lru: &'a mut VecDeque<AcceleratorStepCacheKey>,
    pub(super) cache_hits: &'a mut usize,
    pub(super) cache_misses: &'a mut usize,
    pub(super) prepared_build_ms: &'a mut f64,
    pub(super) dt_bucket_rel_tolerance: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct AcceleratorStepCacheKey {
    dt_bucket: u64,
    stiffness_numerics: Digest,
    mass_numerics: Digest,
}

pub(super) fn build_step_rhs_with_load(
    summary: &AssemblySummary,
    x_prev: &[f64],
    dt: f64,
    load: &[f64],
) -> Vec<f64> {
    let mut rhs = apply_m(&summary.operator, x_prev);
    for (value, load) in rhs.iter_mut().zip(load) {
        *value += dt * load;
    }
    rhs
}

pub(super) fn solve_implicit_step_system(
    summary: &AssemblySummary,
    rhs: &[f64],
    dt: f64,
    nominal_dt: f64,
    options: &TransientSolveOptions,
    use_accelerator: bool,
    accelerator_cache: AcceleratorStepCache<'_>,
) -> (Vec<f64>, f64, bool, Option<LinearStepStats>) {
    const PREPARED_ACCELERATOR_CACHE_CAPACITY: usize = 12;
    let preconditioner_kind = SpdPreconditionerKind::Jacobi;
    let implicit_summary = match build_implicit_summary(summary, rhs, dt) {
        Ok(prepared) => prepared,
        Err(_) => return (vec![0.0; summary.dof_count], f64::INFINITY, false, None),
    };
    if implicit_summary.constraint_matrix.is_some() {
        let backend = if use_accelerator {
            LinearAlgebraBackendKind::Accelerator
        } else {
            LinearAlgebraBackendKind::CpuReference
        };
        let result = crate::solve::linear::solve_linear_system(
            &implicit_summary,
            preconditioner_kind,
            backend,
            &crate::solve::backend::cpu_reference::CpuReferenceBackend,
            None,
        );
        let rhs_norm = dot(rhs, rhs).sqrt().max(1.0);
        return (
            result.solution,
            result.residual_norm / rhs_norm,
            result.converged,
            Some(LinearStepStats {
                solver_backend: result.solver_backend,
                host_sync_count: result.host_sync_count,
                device_apply_k_count: result.device_apply_k_count,
                device_apply_k_attempt_count: result.device_apply_k_attempt_count,
                preconditioner: result.preconditioner,
                diagnostics: result.diagnostics,
                accelerator_evidence: result.accelerator_evidence,
            }),
        );
    }
    let prepared_cpu_system = match implicit_summary.operator.prepare_linear() {
        Ok(prepared) => prepared,
        Err(_) => return (vec![0.0; summary.dof_count], f64::INFINITY, false, None),
    };
    let mut accelerator_admission_diagnostics = Vec::new();
    if use_accelerator {
        let Ok(stiffness_numerics) = summary
            .operator
            .operators()
            .stiffness()
            .numerical_identity()
        else {
            return (vec![0.0; summary.dof_count], f64::INFINITY, false, None);
        };
        let Ok(mass_numerics) = summary.operator.operators().mass().numerical_identity() else {
            return (vec![0.0; summary.dof_count], f64::INFINITY, false, None);
        };
        let dt_key = AcceleratorStepCacheKey {
            dt_bucket: dt_cache_key(dt, nominal_dt, accelerator_cache.dt_bucket_rel_tolerance),
            stiffness_numerics,
            mass_numerics,
        };
        if accelerator_cache
            .prepared_systems_by_dt
            .contains_key(&dt_key)
        {
            *accelerator_cache.cache_hits += 1;
            accelerator_cache
                .prepared_lru
                .retain(|value| *value != dt_key);
            accelerator_cache.prepared_lru.push_back(dt_key);
        } else {
            *accelerator_cache.cache_misses += 1;
            let prepare_start = Instant::now();
            match prepare_accelerator_linear_system(
                prepared_cpu_system.clone(),
                u32::try_from(options.max_linear_iters).unwrap_or(u32::MAX),
            ) {
                Ok(prepared) => {
                    *accelerator_cache.prepared_build_ms +=
                        prepare_start.elapsed().as_secs_f64() * 1_000.0;
                    if accelerator_cache.prepared_systems_by_dt.len()
                        >= PREPARED_ACCELERATOR_CACHE_CAPACITY
                    {
                        if let Some(evicted_key) = accelerator_cache.prepared_lru.pop_front() {
                            accelerator_cache
                                .prepared_systems_by_dt
                                .remove(&evicted_key);
                        }
                    }
                    accelerator_cache
                        .prepared_systems_by_dt
                        .insert(dt_key, prepared);
                    accelerator_cache.prepared_lru.push_back(dt_key);
                }
                Err(error) if error.permits_cpu_fallback() => {
                    *accelerator_cache.prepared_build_ms +=
                        prepare_start.elapsed().as_secs_f64() * 1_000.0;
                    accelerator_admission_diagnostics.push(
                        crate::diagnostics::FeaDiagnostic {
                            code: "FEA_TRANSIENT_ACCELERATOR_ADMISSION_FALLBACK".to_string(),
                            severity: crate::diagnostics::FeaDiagnosticSeverity::Warning,
                            message: format!(
                                "native sparse execution was rejected before provider allocation; using cpu_reference: {}",
                                error.message
                            ),
                        },
                    );
                }
                Err(error) => {
                    *accelerator_cache.prepared_build_ms +=
                        prepare_start.elapsed().as_secs_f64() * 1_000.0;
                    return accelerator_step_failure(
                        &prepared_cpu_system,
                        preconditioner_kind,
                        error,
                    );
                }
            }
        }

        if let Some(prepared) = accelerator_cache.prepared_systems_by_dt.get(&dt_key) {
            let result = match solve_prepared_accelerator_system(
                prepared,
                prepared_cpu_system.rhs(),
                None,
                options.tolerance,
            ) {
                Ok(output) => linear_result_from_accelerator_output(output, preconditioner_kind),
                Err(error) => linear_result_from_accelerator_error(
                    prepared_cpu_system.dof_count(),
                    prepared_cpu_system.rhs(),
                    preconditioner_kind,
                    error,
                ),
            };
            let rhs_norm = dot(prepared_cpu_system.rhs(), prepared_cpu_system.rhs())
                .sqrt()
                .max(1.0);
            let relative_residual = result.residual_norm / rhs_norm;
            return (
                result.solution.clone(),
                relative_residual,
                result.converged,
                Some(LinearStepStats {
                    solver_backend: result.solver_backend,
                    host_sync_count: result.host_sync_count,
                    device_apply_k_count: result.device_apply_k_count,
                    device_apply_k_attempt_count: result.device_apply_k_attempt_count,
                    preconditioner: SolverPreconditioner::from_spd(preconditioner_kind),
                    diagnostics: result.diagnostics.clone(),
                    accelerator_evidence: result.accelerator_evidence.clone(),
                }),
            );
        }
    }

    let mut x = vec![0.0; prepared_cpu_system.dof_count()];
    for (index, constrained) in prepared_cpu_system
        .constrained_dofs()
        .iter()
        .copied()
        .enumerate()
    {
        if constrained {
            x[index] = prepared_cpu_system.rhs()[index];
        }
    }
    let mut r = prepared_cpu_system
        .rhs()
        .iter()
        .zip(apply_prepared(&prepared_cpu_system, &x))
        .map(|(right_hand_side, applied)| right_hand_side - applied)
        .collect::<Vec<_>>();
    let mut p = r.clone();
    let rhs_norm = dot(prepared_cpu_system.rhs(), prepared_cpu_system.rhs())
        .sqrt()
        .max(1.0);
    let mut rr_old = dot(&r, &r);
    if rr_old.sqrt() / rhs_norm <= options.tolerance {
        return cpu_step_result(
            x,
            rr_old.sqrt() / rhs_norm,
            true,
            accelerator_admission_diagnostics,
        );
    }

    let mut converged = false;
    for _ in 0..options.max_linear_iters {
        let ap = apply_prepared(&prepared_cpu_system, &p);
        let denom = dot(&p, &ap);
        if !denom.is_finite() || denom <= 1.0e-18 {
            break;
        }
        let alpha = rr_old / denom;
        for i in 0..x.len() {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rr_new = dot(&r, &r);
        if rr_new.sqrt() / rhs_norm <= options.tolerance {
            converged = true;
            rr_old = rr_new;
            break;
        }
        let beta = rr_new / rr_old.max(1.0e-18);
        for i in 0..p.len() {
            p[i] = r[i] + beta * p[i];
        }
        rr_old = rr_new;
    }

    cpu_step_result(
        x,
        rr_old.sqrt() / rhs_norm,
        converged,
        accelerator_admission_diagnostics,
    )
}

fn cpu_step_result(
    solution: Vec<f64>,
    residual: f64,
    converged: bool,
    diagnostics: Vec<crate::diagnostics::FeaDiagnostic>,
) -> (Vec<f64>, f64, bool, Option<LinearStepStats>) {
    let stats = (!diagnostics.is_empty()).then_some(LinearStepStats {
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        host_sync_count: 0,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        preconditioner: SolverPreconditioner::None,
        diagnostics,
        accelerator_evidence: None,
    });
    (solution, residual, converged, stats)
}

fn accelerator_step_failure(
    prepared: &crate::algebra::PreparedRealLinearSystem,
    preconditioner_kind: SpdPreconditionerKind,
    error: AcceleratorAdapterError,
) -> (Vec<f64>, f64, bool, Option<LinearStepStats>) {
    let result = linear_result_from_accelerator_error(
        prepared.dof_count(),
        prepared.rhs(),
        preconditioner_kind,
        error,
    );
    (
        result.solution,
        f64::INFINITY,
        false,
        Some(LinearStepStats {
            solver_backend: result.solver_backend,
            host_sync_count: result.host_sync_count,
            device_apply_k_count: result.device_apply_k_count,
            device_apply_k_attempt_count: result.device_apply_k_attempt_count,
            preconditioner: result.preconditioner,
            diagnostics: result.diagnostics,
            accelerator_evidence: result.accelerator_evidence,
        }),
    )
}

fn dt_cache_key(dt: f64, nominal_dt: f64, rel_tol: f64) -> u64 {
    if rel_tol <= 0.0 {
        return dt.to_bits();
    }
    let base = nominal_dt.abs().max(1.0e-12);
    let ratio = (dt.abs().max(1.0e-12)) / base;
    let ln_bucket = (1.0 + rel_tol).ln().max(1.0e-9);
    let bucket_idx = (ratio.ln() / ln_bucket).round() as i64;
    bucket_idx as u64
}

fn build_implicit_summary(
    summary: &AssemblySummary,
    rhs: &[f64],
    dt: f64,
) -> Result<AssemblySummary, String> {
    let mut implicit = summary.clone();
    let effective = RealSparseMatrix::scaled_sum(
        &[
            (1.0, summary.operator.operators().mass()),
            (dt, summary.operator.operators().stiffness()),
        ],
        OperatorRole::EffectiveTangent,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .map_err(|error| error.to_string())?;
    let mut prepared_rhs = rhs.to_vec();
    for (dof, constrained) in summary
        .operator
        .constrained_dofs()
        .iter()
        .copied()
        .enumerate()
    {
        if constrained {
            prepared_rhs[dof] = summary.operator.load()[dof];
        }
    }
    let operators = CanonicalRealOperators::try_new(
        effective,
        summary.operator.operators().mass().clone(),
        summary.operator.operators().damping().clone(),
    )
    .map_err(|error| error.to_string())?;
    implicit.operator = AssembledRealSystem::try_new(
        operators,
        summary.operator.constrained_dofs().to_vec(),
        prepared_rhs,
    )
    .map_err(|error| error.to_string())?;
    Ok(implicit)
}

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()
}

pub(super) fn strain_energy(summary: &AssemblySummary, x: &[f64]) -> f64 {
    let kx = apply_stiffness(summary, x);
    0.5 * dot(x, &kx).abs()
}

fn apply_stiffness(summary: &AssemblySummary, x: &[f64]) -> Vec<f64> {
    apply_k(&summary.operator, x)
}
