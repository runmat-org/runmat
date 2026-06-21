use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel};

use crate::{
    assembly::assemble_linear_system,
    contracts::{
        fea_thermal_boundary_heat_flux_field_id, fea_thermal_heat_flux_field_id,
        fea_thermal_heat_source_field_id, fea_thermal_temperature_field_id,
        fea_thermal_temperature_gradient_field_id, ComputeBackend, FeaRunError, FeaRunResult,
        FeaThermalRunResult, ThermalSolveOptions,
    },
    diagnostics::{
        builders::{material_assignment_diagnostics, thermo_mechanical_diagnostic},
        FeaDiagnostic, FeaDiagnosticSeverity,
    },
    physics::{coupling::thermo_mechanical, thermal::constitutive_stats},
    progress::{check_cancelled, emit_phase, is_cancelled, FeaProgressPhase, FeaProgressStatus},
};

const VECTOR_COMPONENT_COUNT: usize = 3;
const BOUNDARY_HEAT_FLUX_COMPONENT_COUNT: usize = 2;

pub fn run_thermal(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaThermalRunResult, FeaRunError> {
    run_thermal_with_options(model, backend, ThermalSolveOptions::default())
}

pub fn run_thermal_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ThermalSolveOptions,
) -> Result<FeaThermalRunResult, FeaRunError> {
    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating thermal FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_thermal")?;
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "thermal model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_thermal")?;

    let Some(thermo_context) = options.thermo_mechanical_context.clone() else {
        return Err(FeaRunError::InvalidModel(
            "thermal solve requires thermo_mechanical_context".to_string(),
        ));
    };
    if !thermo_context.enabled {
        return Err(FeaRunError::InvalidModel(
            "thermal solve requires enabled thermo_mechanical_context".to_string(),
        ));
    }

    let step_count = options.step_count.max(2);
    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling thermal system",
        Some(1),
        Some(5),
    );
    let summary = assemble_linear_system(
        model,
        options.prep_context,
        Some(thermo_context.clone()),
        None,
    );
    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "thermal system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_thermal")?;
    let node_count = (summary.dof_count / 3).max(1);

    let region_avg_delta = if thermo_context.region_temperature_deltas.is_empty() {
        thermo_context.applied_temperature_delta_k
    } else {
        thermo_context
            .region_temperature_deltas
            .iter()
            .map(|delta| delta.temperature_delta_k)
            .sum::<f64>()
            / thermo_context.region_temperature_deltas.len() as f64
    };

    let constitutive = constitutive_stats(model);

    let mut time_points_s = Vec::with_capacity(step_count);
    let mut temperature_snapshots_raw = Vec::with_capacity(step_count);
    let mut residual_norms = Vec::with_capacity(step_count.saturating_sub(1));
    let mut previous_temperatures = vec![thermo_context.reference_temperature_k; node_count];
    let mut min_temperature_k = f64::INFINITY;
    let mut max_temperature_k = f64::NEG_INFINITY;
    let dt = options.time_step_s.max(1.0e-9);
    let relax_gain = (constitutive.response_rate * dt * 10.0).clamp(0.05, 0.7);
    let diffusion_gain = (constitutive.diffusivity_proxy * dt * 2.0e6).clamp(0.0, 0.2);

    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "solving thermal time steps",
        Some(2),
        Some(5),
    );
    for step in 0..step_count {
        if is_cancelled() {
            emit_phase(
                "fea.run_thermal",
                FeaProgressPhase::Solve,
                FeaProgressStatus::Cancelled,
                "thermal solve cancelled",
                Some(step as u64),
                Some(step_count as u64),
            );
            break;
        }
        emit_phase(
            "fea.run_thermal",
            FeaProgressPhase::Solve,
            FeaProgressStatus::Advanced,
            format!("solving thermal step {}", step + 1),
            Some(step as u64),
            Some(step_count as u64),
        );
        let normalized_time = if step_count <= 1 {
            0.0
        } else {
            step as f64 / (step_count - 1) as f64
        };
        let profile =
            thermo_mechanical::sample_time_profile(Some(&thermo_context), normalized_time);
        let transient_response_scale =
            (1.0 - (-constitutive.response_rate * normalized_time.max(0.0)).exp()).clamp(0.0, 1.0);
        let effective_delta = (0.65 * thermo_context.applied_temperature_delta_k
            + 0.35 * region_avg_delta)
            * profile.scale
            * transient_response_scale.max(0.05);

        let mut target_temperatures = Vec::with_capacity(node_count);
        for i in 0..node_count {
            let spatial_factor = if node_count <= 1 {
                1.0
            } else {
                0.9 + 0.2 * (i as f64 / (node_count - 1) as f64)
            };
            target_temperatures
                .push(thermo_context.reference_temperature_k + effective_delta * spatial_factor);
        }

        let mut temperatures = Vec::with_capacity(node_count);
        let mut residual_sum = 0.0;
        for i in 0..node_count {
            let prev = previous_temperatures[i];
            let left = if i == 0 {
                prev
            } else {
                previous_temperatures[i - 1]
            };
            let right = if i + 1 >= node_count {
                prev
            } else {
                previous_temperatures[i + 1]
            };
            let laplacian = left + right - 2.0 * prev;
            let candidate =
                prev + relax_gain * (target_temperatures[i] - prev) + diffusion_gain * laplacian;
            residual_sum += (candidate - prev).abs();
            min_temperature_k = min_temperature_k.min(candidate);
            max_temperature_k = max_temperature_k.max(candidate);
            temperatures.push(candidate);
        }
        if step > 0 {
            residual_norms.push(residual_sum / node_count as f64);
        }
        previous_temperatures = temperatures.clone();

        time_points_s.push(step as f64 * options.time_step_s.max(1.0e-9));
        temperature_snapshots_raw.push(temperatures);
    }
    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "thermal solve complete",
        Some(temperature_snapshots_raw.len() as u64),
        Some(step_count as u64),
    );
    check_cancelled("fea.run_thermal")?;

    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Started,
        "recovering thermal fields",
        Some(3),
        Some(5),
    );
    let temperature_snapshots = temperature_snapshots_raw
        .iter()
        .enumerate()
        .map(|(step, temperatures)| {
            AnalysisField::host_f64(
                fea_thermal_temperature_field_id(step),
                vec![node_count],
                temperatures.clone(),
            )
        })
        .collect::<Vec<_>>();
    let temperature_gradient_raw = recover_temperature_gradients(&temperature_snapshots_raw);
    let heat_flux_raw =
        recover_heat_flux_snapshots(&temperature_gradient_raw, constitutive.conductivity_mean);
    let heat_source_raw = recover_heat_source_snapshots(
        &temperature_snapshots_raw,
        options.time_step_s.max(1.0e-9),
        constitutive.density_mean,
        constitutive.heat_capacity_mean,
    );
    let boundary_heat_flux_raw = recover_boundary_heat_flux_snapshots(&heat_flux_raw, node_count);
    let heat_balance = thermal_heat_balance(
        &temperature_snapshots_raw,
        &heat_source_raw,
        &boundary_heat_flux_raw,
        options.time_step_s.max(1.0e-9),
        constitutive.density_mean,
        constitutive.heat_capacity_mean,
        thermo_context.reference_temperature_k,
    );
    let temperature_gradient_snapshots = temperature_gradient_raw
        .into_iter()
        .enumerate()
        .map(|(step, values)| {
            AnalysisField::host_f64(
                fea_thermal_temperature_gradient_field_id(step),
                vec![node_count, VECTOR_COMPONENT_COUNT],
                values,
            )
        })
        .collect::<Vec<_>>();
    let heat_flux_snapshots = heat_flux_raw
        .into_iter()
        .enumerate()
        .map(|(step, values)| {
            AnalysisField::host_f64(
                fea_thermal_heat_flux_field_id(step),
                vec![node_count, VECTOR_COMPONENT_COUNT],
                values,
            )
        })
        .collect::<Vec<_>>();
    let heat_source_snapshots = heat_source_raw
        .into_iter()
        .enumerate()
        .map(|(step, values)| {
            AnalysisField::host_f64(
                fea_thermal_heat_source_field_id(step),
                vec![node_count],
                values,
            )
        })
        .collect::<Vec<_>>();
    let boundary_heat_flux_snapshots = boundary_heat_flux_raw
        .into_iter()
        .enumerate()
        .map(|(step, values)| {
            AnalysisField::host_f64(
                fea_thermal_boundary_heat_flux_field_id(step),
                vec![BOUNDARY_HEAT_FLUX_COMPONENT_COUNT],
                values,
            )
        })
        .collect::<Vec<_>>();
    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Completed,
        "thermal result field recovery complete",
        Some(4),
        Some(5),
    );
    check_cancelled("fea.run_thermal")?;

    let final_snapshot = temperature_snapshots_raw
        .last()
        .cloned()
        .unwrap_or_else(|| vec![thermo_context.reference_temperature_k; node_count]);
    let initial_snapshot = temperature_snapshots_raw
        .first()
        .cloned()
        .unwrap_or_else(|| vec![thermo_context.reference_temperature_k; node_count]);
    let final_mean_temperature_k =
        final_snapshot.iter().sum::<f64>() / final_snapshot.len().max(1) as f64;
    let final_mean_delta_k = final_mean_temperature_k - thermo_context.reference_temperature_k;
    let peak_delta_k = max_temperature_k - thermo_context.reference_temperature_k;
    let spatial_gradient_index = if final_mean_delta_k.abs() <= 1.0e-9 {
        0.0
    } else {
        ((max_temperature_k - min_temperature_k).abs() / final_mean_delta_k.abs()).clamp(0.0, 5.0)
    };
    let expected_final_delta = {
        let end_profile = thermo_mechanical::sample_time_profile(Some(&thermo_context), 1.0).scale;
        (0.65 * thermo_context.applied_temperature_delta_k + 0.35 * region_avg_delta)
            * end_profile
            * (1.0 - (-constitutive.response_rate).exp()).max(0.05)
    };
    let thermal_response_realization_ratio = if expected_final_delta.abs() <= 1.0e-9 {
        1.0
    } else {
        (final_mean_delta_k / expected_final_delta).clamp(-3.0, 3.0)
    };
    let monotonic_response_fraction = {
        let mut monotonic = 0usize;
        let heating = expected_final_delta >= 0.0;
        for (initial, final_value) in initial_snapshot.iter().zip(final_snapshot.iter()) {
            if (heating && *final_value >= *initial - 1.0e-9)
                || (!heating && *final_value <= *initial + 1.0e-9)
            {
                monotonic = monotonic.saturating_add(1);
            }
        }
        monotonic as f64 / node_count.max(1) as f64
    };

    let max_residual_norm = residual_norms.iter().copied().fold(0.0_f64, f64::max);
    let residual_mean = if residual_norms.is_empty() {
        0.0
    } else {
        residual_norms.iter().sum::<f64>() / residual_norms.len() as f64
    };
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_THERMAL_STABILITY".to_string(),
        severity: if max_residual_norm <= options.residual_target {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "step_count={} time_step_s={} residual_target={} max_residual_norm={} residual_mean={} min_temperature_k={} max_temperature_k={}",
            step_count,
            options.time_step_s,
            options.residual_target,
            max_residual_norm,
            residual_mean,
            min_temperature_k,
            max_temperature_k,
        ),
    }];
    diagnostics.push(FeaDiagnostic {
        code: "FEA_THERMAL_CONSTITUTIVE".to_string(),
        severity: if constitutive.conductivity_spread_ratio > 2.5
            || constitutive.heat_capacity_spread_ratio > 2.5
        {
            FeaDiagnosticSeverity::Warning
        } else {
            FeaDiagnosticSeverity::Info
        },
        message: format!(
            "conductivity_mean={} heat_capacity_mean={} density_mean={} diffusivity_proxy={} response_rate={} conductivity_spread_ratio={} heat_capacity_spread_ratio={}",
            constitutive.conductivity_mean,
            constitutive.heat_capacity_mean,
            constitutive.density_mean,
            constitutive.diffusivity_proxy,
            constitutive.response_rate,
            constitutive.conductivity_spread_ratio,
            constitutive.heat_capacity_spread_ratio,
        ),
    });
    diagnostics.push(FeaDiagnostic {
        code: "FEA_THERMAL_OUTCOME".to_string(),
        severity: if monotonic_response_fraction >= 0.75
            && (0.5..=1.6).contains(&thermal_response_realization_ratio)
        {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "final_mean_temperature_k={} final_mean_delta_k={} peak_delta_k={} spatial_gradient_index={} monotonic_response_fraction={} thermal_response_realization_ratio={}",
            final_mean_temperature_k,
            final_mean_delta_k,
            peak_delta_k,
            spatial_gradient_index,
            monotonic_response_fraction,
            thermal_response_realization_ratio,
        ),
    });
    diagnostics.push(FeaDiagnostic {
        code: "FEA_THERMAL_HEAT_BALANCE".to_string(),
        severity: if heat_balance.residual_ratio <= 0.15 {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "input_heat={} boundary_heat={} stored_energy={} numerical_loss={} heat_balance_residual_ratio={}",
            heat_balance.input_heat,
            heat_balance.boundary_heat,
            heat_balance.stored_energy,
            heat_balance.numerical_loss,
            heat_balance.residual_ratio,
        ),
    });
    diagnostics.extend(material_assignment_diagnostics(&model.material_assignments));
    if let Some(thermo_mechanical) = summary.thermo_mechanical.as_ref() {
        diagnostics.push(thermo_mechanical_diagnostic(thermo_mechanical));
    }

    let run = FeaRunResult {
        backend,
        solver_backend: if backend == ComputeBackend::Gpu {
            "runtime_tensor".to_string()
        } else {
            "cpu_reference".to_string()
        },
        solver_device_apply_k_ratio: 0.0,
        solver_method: "thermal_profile_integrator".to_string(),
        preconditioner: "none".to_string(),
        solver_host_sync_count: 0,
        diagnostics,
        fields: Vec::new(),
    };

    emit_phase(
        "fea.run_thermal",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA thermal run complete",
        Some(5),
        Some(5),
    );

    Ok(FeaThermalRunResult {
        run,
        time_points_s,
        temperature_snapshots,
        temperature_gradient_snapshots,
        heat_flux_snapshots,
        heat_source_snapshots,
        boundary_heat_flux_snapshots,
        residual_norms,
        reference_temperature_k: thermo_context.reference_temperature_k,
    })
}

fn recover_temperature_gradients(temperature_snapshots: &[Vec<f64>]) -> Vec<Vec<f64>> {
    temperature_snapshots
        .iter()
        .map(|temperatures| {
            let node_count = temperatures.len().max(1);
            let mut gradient = vec![0.0; node_count * VECTOR_COMPONENT_COUNT];
            for index in 0..temperatures.len() {
                let left = if index == 0 {
                    temperatures[index]
                } else {
                    temperatures[index - 1]
                };
                let right = if index + 1 >= temperatures.len() {
                    temperatures[index]
                } else {
                    temperatures[index + 1]
                };
                let spacing = if index == 0 || index + 1 >= temperatures.len() {
                    1.0
                } else {
                    2.0
                };
                gradient[index * VECTOR_COMPONENT_COUNT] = (right - left) / spacing;
            }
            gradient
        })
        .collect()
}

fn recover_heat_flux_snapshots(
    temperature_gradient_snapshots: &[Vec<f64>],
    conductivity_mean: f64,
) -> Vec<Vec<f64>> {
    temperature_gradient_snapshots
        .iter()
        .map(|gradient| {
            gradient
                .iter()
                .map(|component| -conductivity_mean.max(0.0) * component)
                .collect()
        })
        .collect()
}

fn recover_heat_source_snapshots(
    temperature_snapshots: &[Vec<f64>],
    dt: f64,
    density_mean: f64,
    heat_capacity_mean: f64,
) -> Vec<Vec<f64>> {
    let volumetric_capacity = density_mean.max(0.0) * heat_capacity_mean.max(0.0);
    temperature_snapshots
        .iter()
        .enumerate()
        .map(|(step, temperatures)| {
            if step == 0 {
                return vec![0.0; temperatures.len()];
            }
            let previous = &temperature_snapshots[step - 1];
            temperatures
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let previous_value = previous.get(index).copied().unwrap_or(*value);
                    volumetric_capacity * (value - previous_value) / dt
                })
                .collect()
        })
        .collect()
}

fn recover_boundary_heat_flux_snapshots(
    heat_flux_snapshots: &[Vec<f64>],
    node_count: usize,
) -> Vec<Vec<f64>> {
    heat_flux_snapshots
        .iter()
        .map(|heat_flux| {
            let left = heat_flux.first().copied().unwrap_or(0.0);
            let right_index = node_count.saturating_sub(1) * VECTOR_COMPONENT_COUNT;
            let right = heat_flux.get(right_index).copied().unwrap_or(left);
            vec![left, right]
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct ThermalHeatBalance {
    input_heat: f64,
    boundary_heat: f64,
    stored_energy: f64,
    numerical_loss: f64,
    residual_ratio: f64,
}

fn thermal_heat_balance(
    temperature_snapshots: &[Vec<f64>],
    heat_source_snapshots: &[Vec<f64>],
    boundary_heat_flux_snapshots: &[Vec<f64>],
    dt: f64,
    density_mean: f64,
    heat_capacity_mean: f64,
    reference_temperature_k: f64,
) -> ThermalHeatBalance {
    let input_heat = heat_source_snapshots
        .iter()
        .map(|snapshot| snapshot.iter().sum::<f64>() * dt.max(1.0e-12))
        .sum::<f64>();
    let boundary_heat = boundary_heat_flux_snapshots
        .iter()
        .map(|snapshot| {
            let left_outward = -snapshot.first().copied().unwrap_or(0.0);
            let right_outward = snapshot.get(1).copied().unwrap_or(0.0);
            (left_outward + right_outward) * dt.max(1.0e-12)
        })
        .sum::<f64>();
    let volumetric_capacity = density_mean.max(0.0) * heat_capacity_mean.max(0.0);
    let stored_energy = if let Some(final_snapshot) = temperature_snapshots.last() {
        let initial_snapshot = temperature_snapshots.first();
        final_snapshot
            .iter()
            .enumerate()
            .map(|(index, final_value)| {
                let initial_value = initial_snapshot
                    .and_then(|snapshot| snapshot.get(index))
                    .copied()
                    .unwrap_or(reference_temperature_k);
                volumetric_capacity * (final_value - initial_value)
            })
            .sum::<f64>()
    } else {
        0.0
    };
    let numerical_loss = input_heat - boundary_heat - stored_energy;
    let denominator = input_heat.abs() + boundary_heat.abs() + stored_energy.abs() + 1.0e-12_f64;
    let residual_ratio = (numerical_loss.abs() / denominator).clamp(0.0, 1.0e6);

    ThermalHeatBalance {
        input_heat,
        boundary_heat,
        stored_energy,
        numerical_loss,
        residual_ratio,
    }
}
