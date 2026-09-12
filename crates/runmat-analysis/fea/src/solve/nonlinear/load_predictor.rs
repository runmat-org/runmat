use crate::{
    algebra::{
        apply_m, apply_prepared, AssembledRealSystem, CanonicalRealOperators, MatrixProperties,
        OperatorRole, RealSparseMatrix,
    },
    assembly::AssemblySummary,
    contracts::SolverPreconditioner,
    diagnostics::FeaDiagnostic,
    solve::{
        accelerator::{
            prepare_accelerator_linear_system, solve_prepared_accelerator_system,
            AcceleratorAdapterError, AcceleratorPreparedLinearSystem, AcceleratorSolveEvidence,
        },
        backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
        linear::{
            linear_result_from_accelerator_error, linear_result_from_accelerator_output,
            solve_linear_system,
        },
        preconditioner::SpdPreconditionerKind,
    },
    ComputeBackend,
};
use runmat_execution::Digest;
use std::{
    collections::{HashMap, VecDeque},
    time::Instant,
};

const PREPARED_ACCELERATOR_CACHE_CAPACITY: usize = 12;

mod schedule;

pub(crate) use schedule::LoadIncrementSchedule;

pub(super) struct LoadPredictorOptions {
    pub(super) increment_count: usize,
    pub(super) max_linear_iters: usize,
    pub(super) tolerance: f64,
}

pub(super) struct LoadPredictorResult {
    pub(super) load_factors: Vec<f64>,
    pub(super) displacement_snapshots: Vec<Vec<f64>>,
    pub(super) residual_norms: Vec<f64>,
    pub(super) diagnostics: Vec<FeaDiagnostic>,
    pub(super) solver_backend: LinearAlgebraBackendKind,
    pub(super) solver_host_sync_count: u32,
    pub(super) device_apply_k_count: u32,
    pub(super) device_apply_k_attempt_count: u32,
    pub(super) preconditioner: SolverPreconditioner,
    pub(super) accelerator_evidence: Vec<AcceleratorSolveEvidence>,
    pub(super) prepared_build_ms: f64,
    pub(super) fallback_apply_count: u32,
}

pub(super) fn predict_load_increments(
    summary: &AssemblySummary,
    options: LoadPredictorOptions,
    backend: ComputeBackend,
    start: f64,
    end: f64,
    schedule: &LoadIncrementSchedule,
) -> LoadPredictorResult {
    let nominal_increment = (end - start) / options.increment_count as f64;
    let mut current_factor = start;
    let mut displacement = vec![0.0; summary.dof_count];
    let mut load_factors = Vec::with_capacity(options.increment_count);
    let mut displacement_snapshots = vec![displacement.clone()];
    let mut residual_norms = Vec::with_capacity(options.increment_count);
    let mut diagnostics = Vec::new();
    let mut solver_backend = LinearAlgebraBackendKind::CpuReference;
    let mut solver_host_sync_count = 0_u32;
    let mut device_apply_k_count = 0_u32;
    let mut device_apply_k_attempt_count = 0_u32;
    let mut preconditioner = SolverPreconditioner::None;
    let mut accelerator_evidence = Vec::new();
    let mut prepared_systems = HashMap::new();
    let mut prepared_lru = VecDeque::new();
    let mut prepared_build_ms = 0.0;
    let mut fallback_apply_count = 0_u32;

    while current_factor < end && !crate::is_fea_cancelled() {
        let next_factor = schedule.next_factor(current_factor, nominal_increment, end);
        let increment = next_factor - current_factor;
        let rhs = predictor_rhs(summary, &displacement, increment);
        let solved = solve_predictor_increment(
            summary,
            &rhs,
            increment,
            &options,
            backend == ComputeBackend::Gpu,
            PredictorAcceleratorCache {
                systems: &mut prepared_systems,
                lru: &mut prepared_lru,
                prepared_build_ms: &mut prepared_build_ms,
            },
        );
        displacement = solved.solution;
        current_factor = next_factor;
        load_factors.push(current_factor);
        displacement_snapshots.push(displacement.clone());
        residual_norms.push(solved.relative_residual);
        solver_backend = solved.solver_backend;
        solver_host_sync_count = solver_host_sync_count.saturating_add(solved.host_sync_count);
        device_apply_k_count = device_apply_k_count.saturating_add(solved.device_apply_k_count);
        device_apply_k_attempt_count =
            device_apply_k_attempt_count.saturating_add(solved.device_apply_k_attempt_count);
        fallback_apply_count = fallback_apply_count.saturating_add(
            solved
                .device_apply_k_attempt_count
                .saturating_sub(solved.device_apply_k_count),
        );
        preconditioner = solved.preconditioner;
        diagnostics.extend(solved.diagnostics);
        if let Some(evidence) = solved.accelerator_evidence {
            accelerator_evidence.push(evidence);
        }
    }

    LoadPredictorResult {
        load_factors,
        displacement_snapshots,
        residual_norms,
        diagnostics,
        solver_backend,
        solver_host_sync_count,
        device_apply_k_count,
        device_apply_k_attempt_count,
        preconditioner,
        accelerator_evidence,
        prepared_build_ms,
        fallback_apply_count,
    }
}

struct PredictorIncrementResult {
    solution: Vec<f64>,
    relative_residual: f64,
    solver_backend: LinearAlgebraBackendKind,
    host_sync_count: u32,
    device_apply_k_count: u32,
    device_apply_k_attempt_count: u32,
    preconditioner: SolverPreconditioner,
    diagnostics: Vec<FeaDiagnostic>,
    accelerator_evidence: Option<AcceleratorSolveEvidence>,
}

struct PredictorAcceleratorCache<'a> {
    systems: &'a mut HashMap<PredictorCacheKey, AcceleratorPreparedLinearSystem>,
    lru: &'a mut VecDeque<PredictorCacheKey>,
    prepared_build_ms: &'a mut f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PredictorCacheKey {
    increment_bits: u64,
    stiffness_numerics: Digest,
    mass_numerics: Digest,
}

fn predictor_rhs(summary: &AssemblySummary, previous: &[f64], increment: f64) -> Vec<f64> {
    let mut rhs = apply_m(&summary.operator, previous);
    for (value, load) in rhs.iter_mut().zip(summary.operator.load()) {
        *value += increment * load;
    }
    rhs
}

fn solve_predictor_increment(
    summary: &AssemblySummary,
    rhs: &[f64],
    increment: f64,
    options: &LoadPredictorOptions,
    use_accelerator: bool,
    cache: PredictorAcceleratorCache<'_>,
) -> PredictorIncrementResult {
    let preconditioner_kind = SpdPreconditionerKind::Jacobi;
    let prepared_summary = match build_predictor_summary(summary, rhs, increment) {
        Ok(prepared) => prepared,
        Err(message) => {
            return failed_increment(summary.dof_count, message, use_accelerator);
        }
    };
    if prepared_summary.constraint_matrix.is_some() {
        let backend = if use_accelerator {
            LinearAlgebraBackendKind::Accelerator
        } else {
            LinearAlgebraBackendKind::CpuReference
        };
        let result = solve_linear_system(
            &prepared_summary,
            preconditioner_kind,
            backend,
            &CpuReferenceBackend,
            None,
        );
        let rhs_norm = dot(rhs, rhs).sqrt().max(1.0);
        return PredictorIncrementResult {
            solution: result.solution,
            relative_residual: result.residual_norm / rhs_norm,
            solver_backend: result.solver_backend,
            host_sync_count: result.host_sync_count,
            device_apply_k_count: result.device_apply_k_count,
            device_apply_k_attempt_count: result.device_apply_k_attempt_count,
            preconditioner: result.preconditioner,
            diagnostics: result.diagnostics,
            accelerator_evidence: result.accelerator_evidence,
        };
    }
    let prepared_cpu = match prepared_summary.operator.prepare_linear() {
        Ok(prepared) => prepared,
        Err(error) => {
            return failed_increment(summary.dof_count, error.to_string(), use_accelerator);
        }
    };
    let mut admission_diagnostics = Vec::new();
    if use_accelerator {
        let Some(key) = predictor_cache_key(summary, increment) else {
            return failed_increment(
                summary.dof_count,
                "predictor operator identity could not be computed".into(),
                true,
            );
        };
        if cache.systems.contains_key(&key) {
            cache.lru.retain(|candidate| *candidate != key);
            cache.lru.push_back(key);
        } else {
            let started = Instant::now();
            match prepare_accelerator_linear_system(
                prepared_cpu.clone(),
                u32::try_from(options.max_linear_iters).unwrap_or(u32::MAX),
            ) {
                Ok(prepared) => {
                    *cache.prepared_build_ms += started.elapsed().as_secs_f64() * 1_000.0;
                    if cache.systems.len() >= PREPARED_ACCELERATOR_CACHE_CAPACITY {
                        if let Some(evicted) = cache.lru.pop_front() {
                            cache.systems.remove(&evicted);
                        }
                    }
                    cache.systems.insert(key, prepared);
                    cache.lru.push_back(key);
                }
                Err(error) if error.permits_cpu_fallback() => {
                    *cache.prepared_build_ms += started.elapsed().as_secs_f64() * 1_000.0;
                    admission_diagnostics.push(FeaDiagnostic {
                        code: "FEA_NONLINEAR_PREDICTOR_ACCELERATOR_ADMISSION_FALLBACK".into(),
                        severity: crate::diagnostics::FeaDiagnosticSeverity::Warning,
                        message: format!(
                            "native sparse execution was rejected before provider allocation; using cpu_reference: {}",
                            error.message
                        ),
                    });
                }
                Err(error) => {
                    *cache.prepared_build_ms += started.elapsed().as_secs_f64() * 1_000.0;
                    return accelerator_failure(&prepared_cpu, preconditioner_kind, error);
                }
            }
        }
        if let Some(prepared) = cache.systems.get(&key) {
            let result = match solve_prepared_accelerator_system(
                prepared,
                prepared_cpu.rhs(),
                None,
                options.tolerance,
            ) {
                Ok(output) => linear_result_from_accelerator_output(output, preconditioner_kind),
                Err(error) => linear_result_from_accelerator_error(
                    prepared_cpu.dof_count(),
                    prepared_cpu.rhs(),
                    preconditioner_kind,
                    error,
                ),
            };
            let rhs_norm = dot(prepared_cpu.rhs(), prepared_cpu.rhs()).sqrt().max(1.0);
            return PredictorIncrementResult {
                solution: result.solution,
                relative_residual: result.residual_norm / rhs_norm,
                solver_backend: result.solver_backend,
                host_sync_count: result.host_sync_count,
                device_apply_k_count: result.device_apply_k_count,
                device_apply_k_attempt_count: result.device_apply_k_attempt_count,
                preconditioner: result.preconditioner,
                diagnostics: result.diagnostics,
                accelerator_evidence: result.accelerator_evidence,
            };
        }
    }

    cpu_predictor_increment(&prepared_cpu, options, admission_diagnostics)
}

fn cpu_predictor_increment(
    prepared: &crate::algebra::PreparedRealLinearSystem,
    options: &LoadPredictorOptions,
    diagnostics: Vec<FeaDiagnostic>,
) -> PredictorIncrementResult {
    let mut solution = vec![0.0; prepared.dof_count()];
    for (index, constrained) in prepared.constrained_dofs().iter().copied().enumerate() {
        if constrained {
            solution[index] = prepared.rhs()[index];
        }
    }
    let mut residual = prepared
        .rhs()
        .iter()
        .zip(apply_prepared(prepared, &solution))
        .map(|(rhs, applied)| rhs - applied)
        .collect::<Vec<_>>();
    let mut direction = residual.clone();
    let rhs_norm = dot(prepared.rhs(), prepared.rhs()).sqrt().max(1.0);
    let mut residual_squared = dot(&residual, &residual);
    if residual_squared.sqrt() / rhs_norm > options.tolerance {
        for _ in 0..options.max_linear_iters {
            let applied = apply_prepared(prepared, &direction);
            let denominator = dot(&direction, &applied);
            if !denominator.is_finite() || denominator <= 1.0e-18 {
                break;
            }
            let alpha = residual_squared / denominator;
            for index in 0..solution.len() {
                solution[index] += alpha * direction[index];
                residual[index] -= alpha * applied[index];
            }
            let next_residual_squared = dot(&residual, &residual);
            if next_residual_squared.sqrt() / rhs_norm <= options.tolerance {
                residual_squared = next_residual_squared;
                break;
            }
            let beta = next_residual_squared / residual_squared.max(1.0e-18);
            for index in 0..direction.len() {
                direction[index] = residual[index] + beta * direction[index];
            }
            residual_squared = next_residual_squared;
        }
    }
    PredictorIncrementResult {
        solution,
        relative_residual: residual_squared.sqrt() / rhs_norm,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        host_sync_count: 0,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        preconditioner: SolverPreconditioner::None,
        diagnostics,
        accelerator_evidence: None,
    }
}

fn build_predictor_summary(
    summary: &AssemblySummary,
    rhs: &[f64],
    increment: f64,
) -> Result<AssemblySummary, String> {
    let effective = RealSparseMatrix::scaled_sum(
        &[
            (1.0, summary.operator.operators().mass()),
            (increment, summary.operator.operators().stiffness()),
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
    let operator = AssembledRealSystem::try_new(
        operators,
        summary.operator.constrained_dofs().to_vec(),
        prepared_rhs,
    )
    .map_err(|error| error.to_string())?;
    let mut prepared = summary.clone();
    prepared.operator = operator;
    Ok(prepared)
}

fn predictor_cache_key(summary: &AssemblySummary, increment: f64) -> Option<PredictorCacheKey> {
    Some(PredictorCacheKey {
        increment_bits: increment.to_bits(),
        stiffness_numerics: summary
            .operator
            .operators()
            .stiffness()
            .numerical_identity()
            .ok()?,
        mass_numerics: summary
            .operator
            .operators()
            .mass()
            .numerical_identity()
            .ok()?,
    })
}

fn accelerator_failure(
    prepared: &crate::algebra::PreparedRealLinearSystem,
    preconditioner: SpdPreconditionerKind,
    error: AcceleratorAdapterError,
) -> PredictorIncrementResult {
    let result = linear_result_from_accelerator_error(
        prepared.dof_count(),
        prepared.rhs(),
        preconditioner,
        error,
    );
    PredictorIncrementResult {
        solution: result.solution,
        relative_residual: f64::INFINITY,
        solver_backend: result.solver_backend,
        host_sync_count: result.host_sync_count,
        device_apply_k_count: result.device_apply_k_count,
        device_apply_k_attempt_count: result.device_apply_k_attempt_count,
        preconditioner: result.preconditioner,
        diagnostics: result.diagnostics,
        accelerator_evidence: result.accelerator_evidence,
    }
}

fn failed_increment(
    dof_count: usize,
    message: String,
    accelerator: bool,
) -> PredictorIncrementResult {
    PredictorIncrementResult {
        solution: vec![0.0; dof_count],
        relative_residual: f64::INFINITY,
        solver_backend: if accelerator {
            LinearAlgebraBackendKind::Accelerator
        } else {
            LinearAlgebraBackendKind::CpuReference
        },
        host_sync_count: 0,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        preconditioner: SolverPreconditioner::None,
        diagnostics: vec![FeaDiagnostic {
            code: "FEA_NONLINEAR_PREDICTOR_FAILED".into(),
            severity: crate::diagnostics::FeaDiagnosticSeverity::Error,
            message,
        }],
        accelerator_evidence: None,
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(test)]
mod tests;
