use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::runtime_tensor_solver::RuntimeTensorPreparedLinearSystem,
    thermo::{sample_time_profile_scale, temporal_profile_variation},
    ComputeBackend, FeaElectroThermalContext, FeaPrepContext, FeaThermoMechanicalContext,
};

mod diagnostics;
mod linear_step;

use diagnostics::push_transient_quality_diagnostics;
use linear_step::{build_step_rhs, solve_implicit_step_system, strain_energy, LinearStepStats};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientSolveOptions {
    pub time_step_s: f64,
    pub min_time_step_s: f64,
    pub max_time_step_s: f64,
    pub step_count: usize,
    pub max_linear_iters: usize,
    pub tolerance: f64,
    pub residual_target: f64,
    pub adaptive_time_step: bool,
    pub max_step_retries: usize,
    pub adapt_min_scale: f64,
    pub adapt_max_scale: f64,
    pub adapt_growth_exponent: f64,
    pub adapt_retry_growth_cap: f64,
    pub adapt_nonconverged_shrink: f64,
    pub dt_bucket_rel_tolerance: f64,
    pub prep_context: Option<FeaPrepContext>,
    pub thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
    pub electro_thermal_context: Option<FeaElectroThermalContext>,
}

impl Default for TransientSolveOptions {
    fn default() -> Self {
        Self {
            time_step_s: 1.0e-3,
            min_time_step_s: 1.0e-6,
            max_time_step_s: 2.0e-2,
            step_count: 10,
            max_linear_iters: 128,
            tolerance: 1.0e-8,
            residual_target: 1.0e-6,
            adaptive_time_step: true,
            max_step_retries: 4,
            adapt_min_scale: 0.8,
            adapt_max_scale: 1.25,
            adapt_growth_exponent: 0.35,
            adapt_retry_growth_cap: 1.05,
            adapt_nonconverged_shrink: 0.75,
            dt_bucket_rel_tolerance: 0.0,
            prep_context: None,
            thermo_mechanical_context: None,
            electro_thermal_context: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientSolveResult {
    pub converged_steps: usize,
    pub total_steps: usize,
    pub time_points_s: Vec<f64>,
    pub displacement_snapshots: Vec<Vec<f64>>,
    pub residual_norms: Vec<f64>,
    pub accepted_time_steps_s: Vec<f64>,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub solver_method: String,
    pub solver_backend: String,
    pub solver_host_sync_count: u32,
    pub device_apply_k_count: u32,
    pub device_apply_k_attempt_count: u32,
    pub preconditioner: String,
}

pub fn solve_transient_system(
    summary: &AssemblySummary,
    options: TransientSolveOptions,
    backend: ComputeBackend,
) -> TransientSolveResult {
    if summary.dof_count == 0 || options.step_count == 0 {
        return TransientSolveResult {
            converged_steps: 0,
            total_steps: options.step_count,
            time_points_s: vec![0.0],
            displacement_snapshots: vec![vec![0.0; summary.dof_count]],
            residual_norms: Vec::new(),
            accepted_time_steps_s: Vec::new(),
            diagnostics: vec![FeaDiagnostic {
                code: "FEA_TRANSIENT_EMPTY_SYSTEM".to_string(),
                severity: FeaDiagnosticSeverity::Warning,
                message: "transient solve skipped because assembled system has zero DOFs or step_count is zero"
                    .to_string(),
            }],
            solver_method: "implicit_euler_pcg".to_string(),
            solver_backend: "cpu_reference".to_string(),
            solver_host_sync_count: 0,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            preconditioner: "none".to_string(),
        };
    }

    let use_runtime_tensor = backend == ComputeBackend::Gpu;
    let thermo_severity_base =
        thermo_mechanical_severity(options.thermo_mechanical_context.clone());
    let thermo_temporal_variation = options
        .thermo_mechanical_context
        .as_ref()
        .map(temporal_profile_variation)
        .unwrap_or(0.0);
    let electro_severity_base = electro_thermal_severity(options.electro_thermal_context.clone());
    let electro_temporal_variation =
        electro_temporal_profile_variation(options.electro_thermal_context.clone());
    let mut thermo_severity_sum = 0.0_f64;
    let mut thermo_time_scale_sum = 0.0_f64;
    let mut thermo_time_extrapolated = 0usize;
    let mut thermo_time_clamped = 0usize;
    let mut electro_severity_sum = 0.0_f64;
    let mut electro_time_scale_sum = 0.0_f64;
    let mut electro_severity_peak = 0.0_f64;
    let mut thermo_severity_peak = 0.0_f64;
    let mut effective_residual_target_peak = options.residual_target;
    let mut thermo_growth_limit_min = 1.0_f64;
    let mut thermo_nonconverged_shrink_min = 1.0_f64;
    let min_dt = options.min_time_step_s.max(1.0e-9);
    let max_dt = options.max_time_step_s.max(min_dt);
    let mut dt = options.time_step_s.clamp(min_dt, max_dt);
    let mut x = vec![0.0; summary.dof_count];
    let mut time_points_s = vec![0.0];
    let mut displacement_snapshots = vec![x.clone()];
    let mut residual_norms = Vec::with_capacity(options.step_count);
    let mut accepted_time_steps_s = Vec::with_capacity(options.step_count);
    let mut converged_steps = 0usize;
    let mut retry_budget_hits = 0usize;
    let mut energies = Vec::with_capacity(options.step_count + 1);
    energies.push(strain_energy(summary, &x));
    let mut solver_backend = "cpu_reference".to_string();
    let mut solver_host_sync_count = 0u32;
    let mut device_apply_k_count = 0u32;
    let mut device_apply_k_attempt_count = 0u32;
    let mut selected_preconditioner = "none".to_string();
    let mut prepared_runtime_systems_by_dt: HashMap<u64, RuntimeTensorPreparedLinearSystem> =
        HashMap::new();
    let mut prepared_runtime_system_lru = VecDeque::new();
    let mut prepared_runtime_cache_hits = 0usize;
    let mut prepared_runtime_cache_misses = 0usize;
    let mut prepared_build_ms = 0.0_f64;
    let mut solve_ms = 0.0_f64;
    let mut fallback_apply_count = 0u32;
    let mut adapt_increase_steps = 0usize;
    let mut adapt_decrease_steps = 0usize;
    let mut adapt_hold_steps = 0usize;
    let mut adapt_scale_sum = 0.0_f64;
    let mut adapt_scale_min = f64::INFINITY;
    let mut adapt_scale_max = 0.0_f64;
    let dt_bucket_rel_tolerance = options.dt_bucket_rel_tolerance.max(0.0);

    for step_index in 0..options.step_count {
        let step_progress = if options.step_count <= 1 {
            1.0
        } else {
            step_index as f64 / (options.step_count - 1) as f64
        };
        let thermo_time_sample = options
            .thermo_mechanical_context
            .as_ref()
            .map(|context| sample_time_profile_scale(context, step_progress));
        let thermo_time_scale = thermo_time_sample.map(|sample| sample.scale).unwrap_or(1.0);
        if let Some(sample) = thermo_time_sample {
            if sample.extrapolated {
                thermo_time_extrapolated = thermo_time_extrapolated.saturating_add(1);
            }
            if sample.clamped {
                thermo_time_clamped = thermo_time_clamped.saturating_add(1);
            }
        }
        let electro_time_scale =
            electro_time_scale(options.electro_thermal_context.clone(), step_progress);
        let thermo_severity = (thermo_severity_base * thermo_time_scale).clamp(0.0, 1.0);
        let electro_severity = (electro_severity_base * electro_time_scale).clamp(0.0, 1.0);
        thermo_severity_sum += thermo_severity;
        thermo_time_scale_sum += thermo_time_scale;
        thermo_severity_peak = thermo_severity_peak.max(thermo_severity);
        electro_severity_sum += electro_severity;
        electro_time_scale_sum += electro_time_scale;
        electro_severity_peak = electro_severity_peak.max(electro_severity);
        let thermo_residual_relaxation = 1.0 + 1.5 * thermo_severity;
        let effective_residual_target = options.residual_target * thermo_residual_relaxation;
        let thermo_growth_limit = (1.0 - 0.12 * thermo_severity).clamp(0.75, 1.0);
        let thermo_nonconverged_shrink = (1.0 - 0.20 * thermo_severity).clamp(0.65, 1.0);
        effective_residual_target_peak =
            effective_residual_target_peak.max(effective_residual_target);
        thermo_growth_limit_min = thermo_growth_limit_min.min(thermo_growth_limit);
        thermo_nonconverged_shrink_min =
            thermo_nonconverged_shrink_min.min(thermo_nonconverged_shrink);
        let mut step_dt = dt;
        let mut retries = 0usize;
        let (next_x, residual_norm, converged, step_stats) = loop {
            let rhs = build_step_rhs(summary, &x, step_dt);
            let solve_start = Instant::now();
            let solved = solve_implicit_step_system(
                summary,
                &rhs,
                step_dt,
                &options,
                use_runtime_tensor,
                &mut prepared_runtime_systems_by_dt,
                &mut prepared_runtime_system_lru,
                &mut prepared_runtime_cache_hits,
                &mut prepared_runtime_cache_misses,
                &mut prepared_build_ms,
                dt_bucket_rel_tolerance,
            );
            solve_ms += solve_start.elapsed().as_secs_f64() * 1_000.0;
            if !options.adaptive_time_step {
                break solved;
            }
            let (candidate_x, candidate_residual, candidate_converged, candidate_stats) = solved;
            if candidate_converged && candidate_residual <= effective_residual_target * 4.0 {
                break (
                    candidate_x,
                    candidate_residual,
                    candidate_converged,
                    candidate_stats,
                );
            }
            if retries >= options.max_step_retries || step_dt <= min_dt * 1.01 {
                retry_budget_hits += 1;
                break (
                    candidate_x,
                    candidate_residual,
                    candidate_converged,
                    candidate_stats,
                );
            }
            step_dt = (step_dt * 0.5).clamp(min_dt, max_dt);
            retries += 1;
        };

        if let Some(LinearStepStats {
            solver_backend: step_solver_backend,
            host_sync_count,
            device_apply_k_count: step_device_apply_k_count,
            device_apply_k_attempt_count: step_device_apply_k_attempt_count,
            preconditioner,
        }) = step_stats
        {
            solver_backend = step_solver_backend;
            solver_host_sync_count = solver_host_sync_count.saturating_add(host_sync_count);
            device_apply_k_count = device_apply_k_count.saturating_add(step_device_apply_k_count);
            device_apply_k_attempt_count =
                device_apply_k_attempt_count.saturating_add(step_device_apply_k_attempt_count);
            fallback_apply_count = fallback_apply_count.saturating_add(
                step_device_apply_k_attempt_count.saturating_sub(step_device_apply_k_count),
            );
            selected_preconditioner = preconditioner;
        }

        x = next_x;
        let next_time = time_points_s.last().copied().unwrap_or(0.0) + step_dt;
        time_points_s.push(next_time);
        displacement_snapshots.push(x.clone());
        residual_norms.push(residual_norm);
        accepted_time_steps_s.push(step_dt);
        energies.push(strain_energy(summary, &x));
        if converged {
            converged_steps += 1;
        }

        if options.adaptive_time_step {
            let next_dt = recommend_next_time_step(
                step_dt,
                residual_norm,
                effective_residual_target,
                min_dt,
                max_dt,
                converged,
                retries,
                &options,
                thermo_growth_limit,
                thermo_nonconverged_shrink,
            );
            let scale = next_dt / step_dt.max(1.0e-12);
            adapt_scale_sum += scale;
            adapt_scale_min = adapt_scale_min.min(scale);
            adapt_scale_max = adapt_scale_max.max(scale);
            if scale > 1.01 {
                adapt_increase_steps += 1;
            } else if scale < 0.99 {
                adapt_decrease_steps += 1;
            } else {
                adapt_hold_steps += 1;
            }
            dt = next_dt;
        } else {
            dt = step_dt;
        }
    }

    let mut max_step_l2_jump_ratio = 0.0_f64;
    let mut nonfinite_displacement_count = 0usize;
    for window in displacement_snapshots.windows(2) {
        let prev = &window[0];
        let next = &window[1];
        let prev_norm = prev.iter().map(|value| value * value).sum::<f64>().sqrt();
        let next_norm = next.iter().map(|value| value * value).sum::<f64>().sqrt();
        let mut jump_norm_sq = 0.0_f64;
        for (a, b) in prev.iter().zip(next.iter()) {
            let d = b - a;
            jump_norm_sq += d * d;
            if !b.is_finite() {
                nonfinite_displacement_count += 1;
            }
        }
        let jump_norm = jump_norm_sq.sqrt();
        let jump_ratio = jump_norm / prev_norm.max(next_norm).max(1.0);
        max_step_l2_jump_ratio = max_step_l2_jump_ratio.max(jump_ratio);
    }

    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_TRANSIENT_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: "solver=implicit_euler_pcg matrix_free=true".to_string(),
    }];
    push_transient_quality_diagnostics(
        &mut diagnostics,
        &options,
        dt,
        converged_steps,
        retry_budget_hits,
        &accepted_time_steps_s,
        &residual_norms,
        &energies,
        use_runtime_tensor,
        prepared_runtime_systems_by_dt.len(),
        prepared_runtime_cache_hits,
        prepared_runtime_cache_misses,
        prepared_build_ms,
        solve_ms,
        fallback_apply_count,
        adapt_increase_steps,
        adapt_decrease_steps,
        adapt_hold_steps,
        if accepted_time_steps_s.is_empty() {
            1.0
        } else {
            adapt_scale_sum / accepted_time_steps_s.len() as f64
        },
        if adapt_scale_min.is_finite() {
            adapt_scale_min
        } else {
            1.0
        },
        if adapt_scale_max > 0.0 {
            adapt_scale_max
        } else {
            1.0
        },
        dt_bucket_rel_tolerance,
        max_step_l2_jump_ratio,
        nonfinite_displacement_count,
        if options.step_count == 0 {
            0.0
        } else {
            thermo_severity_sum / options.step_count as f64
        },
        if options.step_count == 0 {
            1.0
        } else {
            thermo_time_scale_sum / options.step_count as f64
        },
        thermo_severity_peak,
        thermo_temporal_variation,
        thermo_time_extrapolated,
        thermo_time_clamped,
        effective_residual_target_peak,
        thermo_growth_limit_min,
        thermo_nonconverged_shrink_min,
    );
    if electro_severity_peak > 0.0 {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_ET_TRANSIENT".to_string(),
            severity: if electro_severity_peak <= 0.6 && electro_temporal_variation <= 0.5 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "severity_mean={} time_scale_mean={} severity_peak={} temporal_variation={}",
                if options.step_count == 0 {
                    0.0
                } else {
                    electro_severity_sum / options.step_count as f64
                },
                if options.step_count == 0 {
                    1.0
                } else {
                    electro_time_scale_sum / options.step_count as f64
                },
                electro_severity_peak,
                electro_temporal_variation,
            ),
        });
    }

    TransientSolveResult {
        converged_steps,
        total_steps: options.step_count,
        time_points_s,
        displacement_snapshots,
        residual_norms,
        accepted_time_steps_s,
        diagnostics,
        solver_method: "implicit_euler_pcg".to_string(),
        solver_backend,
        solver_host_sync_count,
        device_apply_k_count,
        device_apply_k_attempt_count,
        preconditioner: selected_preconditioner,
    }
}

fn recommend_next_time_step(
    step_dt: f64,
    residual_norm: f64,
    residual_target: f64,
    min_dt: f64,
    max_dt: f64,
    converged: bool,
    retries: usize,
    options: &TransientSolveOptions,
    thermo_growth_limit: f64,
    thermo_nonconverged_shrink: f64,
) -> f64 {
    if !converged {
        return (step_dt
            * options.adapt_nonconverged_shrink.clamp(0.2, 1.0)
            * thermo_nonconverged_shrink)
            .clamp(min_dt, max_dt);
    }

    let target = residual_target.max(1.0e-12);
    let ratio = (target / residual_norm.max(1.0e-12)).clamp(0.25, 4.0);
    let mut factor = ratio.powf(options.adapt_growth_exponent.clamp(0.1, 1.0));
    factor = factor.clamp(
        options.adapt_min_scale.clamp(0.2, 1.0),
        options.adapt_max_scale.clamp(1.0, 2.0),
    );
    if retries > 0 {
        factor = factor.min(options.adapt_retry_growth_cap.clamp(1.0, 1.5));
    } else if residual_norm <= target * 0.1 {
        factor = factor.max((1.0 + (options.adapt_max_scale - 1.0) * 0.6).clamp(1.0, 1.5));
    }
    factor = factor.min(thermo_growth_limit.max(0.5));
    (step_dt * factor).clamp(min_dt, max_dt)
}

fn thermo_mechanical_severity(context: Option<FeaThermoMechanicalContext>) -> f64 {
    let Some(context) = context else {
        return 0.0;
    };
    if !context.enabled {
        return 0.0;
    }
    let thermal_strain = (context.thermal_expansion_coefficient
        * context.applied_temperature_delta_k.abs())
    .clamp(0.0, 0.05);
    (thermal_strain / 0.05).clamp(0.0, 1.0)
}

fn electro_thermal_severity(context: Option<FeaElectroThermalContext>) -> f64 {
    let Some(context) = context else {
        return 0.0;
    };
    if !context.enabled {
        return 0.0;
    }
    let joule_proxy = (context.applied_voltage_v.powi(2)
        * context.base_electrical_conductivity_s_per_m.max(1.0e-9)
        * context.resistive_heating_coefficient.max(0.0)
        / 1.0e7)
        .clamp(0.0, 1.0);
    joule_proxy
}

fn electro_time_scale(context: Option<FeaElectroThermalContext>, normalized_time: f64) -> f64 {
    let Some(context) = context else {
        return 1.0;
    };
    if context.time_profile.is_empty() {
        return 1.0;
    }
    let t = normalized_time.clamp(0.0, 1.0);
    let mut points = context.time_profile;
    points.sort_by(|a, b| a.normalized_time.total_cmp(&b.normalized_time));
    if t <= points[0].normalized_time {
        return points[0].current_scale.clamp(0.2, 2.0);
    }
    for pair in points.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        if t >= a.normalized_time && t <= b.normalized_time {
            let span = (b.normalized_time - a.normalized_time).abs().max(1.0e-9);
            let alpha = (t - a.normalized_time) / span;
            return (a.current_scale + (b.current_scale - a.current_scale) * alpha).clamp(0.2, 2.0);
        }
    }
    points
        .last()
        .map(|p| p.current_scale.clamp(0.2, 2.0))
        .unwrap_or(1.0)
}

fn electro_temporal_profile_variation(context: Option<FeaElectroThermalContext>) -> f64 {
    let Some(context) = context else {
        return 0.0;
    };
    if context.time_profile.len() < 2 {
        return 0.0;
    }
    let mut min_scale = f64::INFINITY;
    let mut max_scale = -f64::INFINITY;
    for point in &context.time_profile {
        min_scale = min_scale.min(point.current_scale);
        max_scale = max_scale.max(point.current_scale);
    }
    if !min_scale.is_finite() || !max_scale.is_finite() {
        return 0.0;
    }
    ((max_scale - min_scale).abs() / 2.0).clamp(0.0, 1.0)
}
