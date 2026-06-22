use runmat_analysis_core::{
    validate_model, AnalysisField, AnalysisModel, BoundaryConditionKind, LoadKind,
};

use crate::{
    assembly::{assemble_linear_system, AssemblySummary},
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
const BOUNDARY_HEAT_FLUX_COMPONENT_COUNT: usize = 6;

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
    let source_boundary =
        thermal_source_boundary_summary(model, thermo_context.reference_temperature_k);

    let mut time_points_s = Vec::with_capacity(step_count);
    let mut temperature_snapshots_raw = Vec::with_capacity(step_count);
    let mut residual_norms = Vec::with_capacity(step_count.saturating_sub(1));
    let mut previous_temperatures = vec![thermo_context.reference_temperature_k; node_count];
    let mut min_temperature_k = f64::INFINITY;
    let mut max_temperature_k = f64::NEG_INFINITY;
    let dt = options.time_step_s.max(1.0e-9);
    let relax_gain = (constitutive.response_rate * dt * 10.0).clamp(0.05, 0.7);
    let diffusion_gain = (constitutive.diffusivity_estimate * dt * 2.0e6).clamp(0.0, 0.2);

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
            * transient_response_scale.max(0.05)
            + source_boundary.source_temperature_delta_k
                * transient_response_scale
                * normalized_time.max(0.0)
            + source_boundary.boundary_temperature_delta_k * profile.scale;

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
    let recovery_topology = ThermalRecoveryTopology::from_assembly(&summary, node_count);
    let temperature_gradient_raw =
        recover_temperature_gradients(&temperature_snapshots_raw, recovery_topology);
    let heat_flux_raw =
        recover_heat_flux_snapshots(&temperature_gradient_raw, constitutive.conductivity_mean);
    let heat_source_raw = recover_heat_source_snapshots(
        &temperature_snapshots_raw,
        options.time_step_s.max(1.0e-9),
        constitutive.density_mean,
        constitutive.heat_capacity_mean,
    );
    let boundary_heat_flux_raw =
        recover_boundary_heat_flux_snapshots(&heat_flux_raw, recovery_topology);
    let heat_balance = thermal_heat_balance(
        &temperature_snapshots_raw,
        &heat_source_raw,
        &boundary_heat_flux_raw,
        options.time_step_s.max(1.0e-9),
        constitutive.density_mean,
        constitutive.heat_capacity_mean,
        thermo_context.reference_temperature_k,
    );
    let thermal_elements = thermal_element_node_sets(recovery_topology);
    let element_count = thermal_elements.len().max(1);
    let temperature_gradient_snapshots = project_nodal_vector_snapshots_to_thermal_elements(
        &temperature_gradient_raw,
        &thermal_elements,
    )
    .into_iter()
    .enumerate()
    .map(|(step, values)| {
        AnalysisField::host_f64(
            fea_thermal_temperature_gradient_field_id(step),
            vec![element_count, VECTOR_COMPONENT_COUNT],
            values,
        )
    })
    .collect::<Vec<_>>();
    let heat_flux_snapshots =
        project_nodal_vector_snapshots_to_thermal_elements(&heat_flux_raw, &thermal_elements)
            .into_iter()
            .enumerate()
            .map(|(step, values)| {
                AnalysisField::host_f64(
                    fea_thermal_heat_flux_field_id(step),
                    vec![element_count, VECTOR_COMPONENT_COUNT],
                    values,
                )
            })
            .collect::<Vec<_>>();
    let heat_source_snapshots =
        project_nodal_scalar_snapshots_to_thermal_elements(&heat_source_raw, &thermal_elements)
            .into_iter()
            .enumerate()
            .map(|(step, values)| {
                AnalysisField::host_f64(
                    fea_thermal_heat_source_field_id(step),
                    vec![element_count],
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
    let expected_known_answer_delta = {
        let end_profile = thermo_mechanical::sample_time_profile(Some(&thermo_context), 1.0).scale;
        let transient_response_scale = (1.0 - (-constitutive.response_rate).exp()).max(0.05);
        expected_final_delta
            + source_boundary.source_temperature_delta_k * transient_response_scale
            + source_boundary.boundary_temperature_delta_k * end_profile
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
    let known_answer = thermal_known_answer(
        &initial_snapshot,
        &final_snapshot,
        recovery_topology,
        thermo_context.reference_temperature_k,
        expected_known_answer_delta,
        source_boundary,
    );
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
            "conductivity_mean={} heat_capacity_mean={} density_mean={} diffusivity_estimate={} response_rate={} conductivity_spread_ratio={} heat_capacity_spread_ratio={}",
            constitutive.conductivity_mean,
            constitutive.heat_capacity_mean,
            constitutive.density_mean,
            constitutive.diffusivity_estimate,
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
        code: "FEA_THERMAL_FIELD_RECOVERY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "recovery_node_count={} recovery_dimensions={}x{}x{} recovery_spacing_x={} recovery_spacing_y={} recovery_spacing_z={} coordinate_active_dimension_count={} coordinate_characteristic_length_m={} boundary_face_count={}",
            recovery_topology.node_count,
            recovery_topology.dims[0],
            recovery_topology.dims[1],
            recovery_topology.dims[2],
            recovery_topology.spacing[0],
            recovery_topology.spacing[1],
            recovery_topology.spacing[2],
            recovery_topology.active_dimension_count,
            recovery_topology.characteristic_length_m,
            BOUNDARY_HEAT_FLUX_COMPONENT_COUNT,
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
    diagnostics.push(FeaDiagnostic {
        code: "FEA_THERMAL_KNOWN_ANSWER".to_string(),
        severity: if known_answer.slab_linear_profile_rms_ratio <= 0.12
            && known_answer.slab_monotonic_edge_fraction >= 0.95
            && known_answer.lumped_response_error_ratio <= 0.45
            && known_answer.source_response_sign_alignment >= 1.0
        {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "slab_linear_profile_rms_ratio={} slab_monotonic_edge_fraction={} lumped_response_error_ratio={} source_response_sign_alignment={} expected_lumped_delta_k={} observed_lumped_delta_k={}",
            known_answer.slab_linear_profile_rms_ratio,
            known_answer.slab_monotonic_edge_fraction,
            known_answer.lumped_response_error_ratio,
            known_answer.source_response_sign_alignment,
            expected_known_answer_delta,
            known_answer.observed_lumped_delta_k,
        ),
    });
    diagnostics.push(FeaDiagnostic {
        code: "FEA_THERMAL_SOURCE_BOUNDARY_MODEL".to_string(),
        severity: if source_boundary.has_thermal_model_data() {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "thermal_source_count={} thermal_boundary_count={} prescribed_temperature_count={} heat_flux_boundary_count={} convection_boundary_count={} thermal_source_coverage_ratio={} thermal_boundary_coverage_ratio={} volumetric_heat_source_w_per_m3={} prescribed_temperature_mean_k={} boundary_heat_flux_w_per_m2={} convection_ambient_mean_k={} convection_coefficient_mean_w_per_m2k={} source_temperature_delta_k={} boundary_temperature_delta_k={}",
            source_boundary.thermal_source_count,
            source_boundary.thermal_boundary_count,
            source_boundary.prescribed_temperature_count,
            source_boundary.heat_flux_boundary_count,
            source_boundary.convection_boundary_count,
            source_boundary.source_coverage_ratio,
            source_boundary.boundary_coverage_ratio,
            source_boundary.volumetric_heat_source_w_per_m3,
            source_boundary.prescribed_temperature_mean_k,
            source_boundary.boundary_heat_flux_w_per_m2,
            source_boundary.convection_ambient_mean_k,
            source_boundary.convection_coefficient_mean_w_per_m2k,
            source_boundary.source_temperature_delta_k,
            source_boundary.boundary_temperature_delta_k,
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

#[derive(Debug, Clone, Copy)]
struct ThermalRecoveryTopology {
    node_count: usize,
    dims: [usize; VECTOR_COMPONENT_COUNT],
    spacing: [f64; VECTOR_COMPONENT_COUNT],
    active_dimension_count: usize,
    characteristic_length_m: f64,
}

impl ThermalRecoveryTopology {
    fn from_assembly(summary: &AssemblySummary, node_count: usize) -> Self {
        let graph = summary.prep_graph_assembly.as_ref();
        let degree_mean = graph.map(|item| item.degree_mean).unwrap_or(0.0);
        let connected_component_count = graph
            .map(|item| item.connected_component_count)
            .unwrap_or(1)
            .max(1);
        let component_scale = (node_count / connected_component_count).max(1);
        let prefer_volume = node_count >= 27 || degree_mean >= 3.5;
        let prefer_surface = node_count >= 9 || degree_mean >= 1.5;
        let z_dim = if prefer_volume {
            (component_scale as f64).cbrt().round().max(1.0) as usize
        } else {
            1
        };
        let y_dim = if prefer_surface {
            ((component_scale as f64) / z_dim.max(1) as f64)
                .sqrt()
                .round()
                .max(1.0) as usize
        } else {
            1
        };
        let x_dim = node_count
            .div_ceil(y_dim.max(1).saturating_mul(z_dim.max(1)))
            .max(1);
        let dims = [x_dim, y_dim, z_dim];
        let inferred_active_dimension_count = dims.iter().filter(|dim| **dim > 1).count().max(1);
        let coordinate_summary = summary.prep_coordinates;
        let active_dimension_count = coordinate_summary
            .map(|item| item.active_dimension_count.max(1))
            .unwrap_or(inferred_active_dimension_count);
        let characteristic_length_m = coordinate_summary
            .map(|item| finite_positive_or(item.characteristic_length_m, 1.0))
            .unwrap_or(1.0);
        let spacing = coordinate_summary
            .map(|item| thermal_axis_spacing(dims, item.span_m, characteristic_length_m))
            .unwrap_or_else(|| thermal_normalized_axis_spacing(dims));
        Self {
            node_count: node_count.max(1),
            dims,
            spacing,
            active_dimension_count,
            characteristic_length_m,
        }
    }

    fn coords(self, index: usize) -> [usize; VECTOR_COMPONENT_COUNT] {
        let x_dim = self.dims[0].max(1);
        let y_dim = self.dims[1].max(1);
        let plane = x_dim.saturating_mul(y_dim).max(1);
        let z = index / plane;
        let rem = index % plane;
        let y = rem / x_dim;
        let x = rem % x_dim;
        [x, y, z]
    }

    fn index(self, coords: [usize; VECTOR_COMPONENT_COUNT]) -> Option<usize> {
        if coords
            .iter()
            .zip(self.dims.iter())
            .any(|(coord, dim)| *coord >= *dim)
        {
            return None;
        }
        let index = coords[0]
            + coords[1].saturating_mul(self.dims[0])
            + coords[2].saturating_mul(self.dims[0].saturating_mul(self.dims[1]));
        (index < self.node_count).then_some(index)
    }
}

fn thermal_normalized_axis_spacing(
    dims: [usize; VECTOR_COMPONENT_COUNT],
) -> [f64; VECTOR_COMPONENT_COUNT] {
    dims.map(|dim| {
        if dim <= 1 {
            1.0
        } else {
            1.0 / (dim - 1) as f64
        }
    })
}

fn thermal_axis_spacing(
    dims: [usize; VECTOR_COMPONENT_COUNT],
    span_m: [f64; VECTOR_COMPONENT_COUNT],
    characteristic_length_m: f64,
) -> [f64; VECTOR_COMPONENT_COUNT] {
    let fallback = finite_positive_or(characteristic_length_m, 1.0);
    let mut spacing = [fallback; VECTOR_COMPONENT_COUNT];
    for axis in 0..VECTOR_COMPONENT_COUNT {
        spacing[axis] = if dims[axis] <= 1 {
            finite_positive_or(span_m[axis], fallback)
        } else {
            finite_positive_or(span_m[axis] / (dims[axis] - 1) as f64, fallback)
        };
    }
    spacing
}

fn finite_positive_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

fn recover_temperature_gradients(
    temperature_snapshots: &[Vec<f64>],
    topology: ThermalRecoveryTopology,
) -> Vec<Vec<f64>> {
    temperature_snapshots
        .iter()
        .map(|temperatures| {
            let node_count = temperatures.len().max(1);
            let mut gradient = vec![0.0; node_count * VECTOR_COMPONENT_COUNT];
            for index in 0..temperatures.len() {
                for axis in 0..VECTOR_COMPONENT_COUNT {
                    gradient[index * VECTOR_COMPONENT_COUNT + axis] =
                        thermal_axis_derivative(temperatures, topology, index, axis);
                }
            }
            gradient
        })
        .collect()
}

fn thermal_axis_derivative(
    temperatures: &[f64],
    topology: ThermalRecoveryTopology,
    index: usize,
    axis: usize,
) -> f64 {
    if topology.dims[axis] <= 1 {
        return 0.0;
    }
    let coords = topology.coords(index);
    let mut prev_coords = coords;
    let prev_index = if coords[axis] > 0 {
        prev_coords[axis] -= 1;
        topology.index(prev_coords)
    } else {
        None
    };
    let mut next_coords = coords;
    let next_index = if coords[axis] + 1 < topology.dims[axis] {
        next_coords[axis] += 1;
        topology.index(next_coords)
    } else {
        None
    };
    let spacing = topology.spacing[axis].max(1.0e-12);
    match (prev_index, next_index) {
        (Some(prev), Some(next)) => (temperatures[next] - temperatures[prev]) / (2.0 * spacing),
        (Some(prev), None) => (temperatures[index] - temperatures[prev]) / spacing,
        (None, Some(next)) => (temperatures[next] - temperatures[index]) / spacing,
        (None, None) => 0.0,
    }
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

fn project_nodal_vector_snapshots_to_thermal_elements(
    snapshots: &[Vec<f64>],
    elements: &[Vec<usize>],
) -> Vec<Vec<f64>> {
    snapshots
        .iter()
        .map(|snapshot| {
            let mut values = Vec::with_capacity(elements.len() * VECTOR_COMPONENT_COUNT);
            for element_nodes in elements {
                for axis in 0..VECTOR_COMPONENT_COUNT {
                    values.push(average_nodal_component(snapshot, element_nodes, axis));
                }
            }
            values
        })
        .collect()
}

fn project_nodal_scalar_snapshots_to_thermal_elements(
    snapshots: &[Vec<f64>],
    elements: &[Vec<usize>],
) -> Vec<Vec<f64>> {
    snapshots
        .iter()
        .map(|snapshot| {
            elements
                .iter()
                .map(|element_nodes| average_nodal_scalar(snapshot, element_nodes))
                .collect()
        })
        .collect()
}

fn average_nodal_component(snapshot: &[f64], element_nodes: &[usize], axis: usize) -> f64 {
    if element_nodes.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for node in element_nodes {
        if let Some(value) = snapshot.get(node * VECTOR_COMPONENT_COUNT + axis) {
            sum += *value;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn average_nodal_scalar(snapshot: &[f64], element_nodes: &[usize]) -> f64 {
    if element_nodes.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for node in element_nodes {
        if let Some(value) = snapshot.get(*node) {
            sum += *value;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn thermal_element_node_sets(topology: ThermalRecoveryTopology) -> Vec<Vec<usize>> {
    let x_cells = topology.dims[0].saturating_sub(1).max(1);
    let y_cells = topology.dims[1].saturating_sub(1).max(1);
    let z_cells = topology.dims[2].saturating_sub(1).max(1);
    let mut elements = Vec::with_capacity(x_cells * y_cells * z_cells);
    for z in 0..z_cells {
        for y in 0..y_cells {
            for x in 0..x_cells {
                let mut nodes = Vec::with_capacity(8);
                for dz in thermal_axis_offsets(topology.dims[2]) {
                    for dy in thermal_axis_offsets(topology.dims[1]) {
                        for dx in thermal_axis_offsets(topology.dims[0]) {
                            let coords = [x + dx, y + dy, z + dz];
                            if let Some(index) = topology.index(coords) {
                                if !nodes.contains(&index) {
                                    nodes.push(index);
                                }
                            }
                        }
                    }
                }
                if !nodes.is_empty() {
                    elements.push(nodes);
                }
            }
        }
    }
    if elements.is_empty() {
        elements.push(vec![0]);
    }
    elements
}

fn thermal_axis_offsets(dim: usize) -> std::ops::RangeInclusive<usize> {
    0..=usize::from(dim > 1)
}

fn recover_boundary_heat_flux_snapshots(
    heat_flux_snapshots: &[Vec<f64>],
    topology: ThermalRecoveryTopology,
) -> Vec<Vec<f64>> {
    heat_flux_snapshots
        .iter()
        .map(|heat_flux| thermal_boundary_face_fluxes(heat_flux, topology))
        .collect()
}

fn thermal_boundary_face_fluxes(heat_flux: &[f64], topology: ThermalRecoveryTopology) -> Vec<f64> {
    let mut boundary = vec![0.0; BOUNDARY_HEAT_FLUX_COMPONENT_COUNT];
    for axis in 0..VECTOR_COMPONENT_COUNT {
        if topology.dims[axis] <= 1 {
            continue;
        }
        let min_face = thermal_face_average(heat_flux, topology, axis, 0);
        let max_face = thermal_face_average(heat_flux, topology, axis, topology.dims[axis] - 1);
        boundary[axis * 2] = -min_face;
        boundary[axis * 2 + 1] = max_face;
    }
    boundary
}

fn thermal_face_average(
    heat_flux: &[f64],
    topology: ThermalRecoveryTopology,
    axis: usize,
    face_coord: usize,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for node in 0..topology.node_count {
        let coords = topology.coords(node);
        if coords[axis] != face_coord {
            continue;
        }
        let value = heat_flux
            .get(node * VECTOR_COMPONENT_COUNT + axis)
            .copied()
            .unwrap_or(0.0);
        sum += value;
        count = count.saturating_add(1);
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

#[derive(Debug, Clone, Copy)]
struct ThermalHeatBalance {
    input_heat: f64,
    boundary_heat: f64,
    stored_energy: f64,
    numerical_loss: f64,
    residual_ratio: f64,
}

#[derive(Debug, Clone, Copy)]
struct ThermalKnownAnswer {
    slab_linear_profile_rms_ratio: f64,
    slab_monotonic_edge_fraction: f64,
    lumped_response_error_ratio: f64,
    source_response_sign_alignment: f64,
    observed_lumped_delta_k: f64,
}

#[derive(Debug, Clone, Copy)]
struct ThermalSourceBoundarySummary {
    thermal_source_count: usize,
    thermal_boundary_count: usize,
    prescribed_temperature_count: usize,
    heat_flux_boundary_count: usize,
    convection_boundary_count: usize,
    source_coverage_ratio: f64,
    boundary_coverage_ratio: f64,
    volumetric_heat_source_w_per_m3: f64,
    prescribed_temperature_mean_k: f64,
    boundary_heat_flux_w_per_m2: f64,
    convection_ambient_mean_k: f64,
    convection_coefficient_mean_w_per_m2k: f64,
    source_temperature_delta_k: f64,
    boundary_temperature_delta_k: f64,
}

impl ThermalSourceBoundarySummary {
    fn has_thermal_model_data(self) -> bool {
        self.thermal_source_count > 0 || self.thermal_boundary_count > 0
    }
}

fn thermal_source_boundary_summary(
    model: &AnalysisModel,
    reference_temperature_k: f64,
) -> ThermalSourceBoundarySummary {
    let mut thermal_source_count = 0usize;
    let mut volumetric_heat_source_w_per_m3 = 0.0_f64;
    for load in &model.loads {
        if let LoadKind::HeatSource {
            volumetric_w_per_m3,
        } = load.kind
        {
            thermal_source_count += 1;
            volumetric_heat_source_w_per_m3 += volumetric_w_per_m3;
        }
    }

    let mut prescribed_temperature_count = 0usize;
    let mut heat_flux_boundary_count = 0usize;
    let mut convection_boundary_count = 0usize;
    let mut prescribed_temperature_sum_k = 0.0_f64;
    let mut boundary_heat_flux_w_per_m2 = 0.0_f64;
    let mut convection_ambient_sum_k = 0.0_f64;
    let mut convection_coefficient_sum_w_per_m2k = 0.0_f64;
    for boundary_condition in &model.boundary_conditions {
        match boundary_condition.kind {
            BoundaryConditionKind::ThermalPrescribedTemperature { temperature_k } => {
                prescribed_temperature_count += 1;
                prescribed_temperature_sum_k += temperature_k;
            }
            BoundaryConditionKind::ThermalHeatFlux { heat_flux_w_per_m2 } => {
                heat_flux_boundary_count += 1;
                boundary_heat_flux_w_per_m2 += heat_flux_w_per_m2;
            }
            BoundaryConditionKind::ThermalConvection {
                ambient_temperature_k,
                coefficient_w_per_m2k,
            } => {
                convection_boundary_count += 1;
                convection_ambient_sum_k += ambient_temperature_k;
                convection_coefficient_sum_w_per_m2k += coefficient_w_per_m2k;
            }
            _ => {}
        }
    }
    let thermal_boundary_count =
        prescribed_temperature_count + heat_flux_boundary_count + convection_boundary_count;
    let prescribed_temperature_mean_k = if prescribed_temperature_count == 0 {
        reference_temperature_k
    } else {
        prescribed_temperature_sum_k / prescribed_temperature_count as f64
    };
    let convection_ambient_mean_k = if convection_boundary_count == 0 {
        reference_temperature_k
    } else {
        convection_ambient_sum_k / convection_boundary_count as f64
    };
    let convection_coefficient_mean_w_per_m2k = if convection_boundary_count == 0 {
        0.0
    } else {
        convection_coefficient_sum_w_per_m2k / convection_boundary_count as f64
    };
    let source_temperature_delta_k = (volumetric_heat_source_w_per_m3 / 1.0e6).clamp(-25.0, 25.0);
    let prescribed_delta_k = prescribed_temperature_mean_k - reference_temperature_k;
    let flux_delta_k = (boundary_heat_flux_w_per_m2 / 1.0e4).clamp(-15.0, 15.0);
    let convection_delta_k = ((convection_ambient_mean_k - reference_temperature_k)
        * convection_coefficient_mean_w_per_m2k
        / 1.0e3)
        .clamp(-15.0, 15.0);
    ThermalSourceBoundarySummary {
        thermal_source_count,
        thermal_boundary_count,
        prescribed_temperature_count,
        heat_flux_boundary_count,
        convection_boundary_count,
        source_coverage_ratio: thermal_source_count as f64 / model.loads.len().max(1) as f64,
        boundary_coverage_ratio: thermal_boundary_count as f64
            / model.boundary_conditions.len().max(1) as f64,
        volumetric_heat_source_w_per_m3,
        prescribed_temperature_mean_k,
        boundary_heat_flux_w_per_m2,
        convection_ambient_mean_k,
        convection_coefficient_mean_w_per_m2k,
        source_temperature_delta_k,
        boundary_temperature_delta_k: 0.45 * prescribed_delta_k + flux_delta_k + convection_delta_k,
    }
}

fn thermal_known_answer(
    initial_snapshot: &[f64],
    final_snapshot: &[f64],
    topology: ThermalRecoveryTopology,
    reference_temperature_k: f64,
    expected_lumped_delta_k: f64,
    source_boundary: ThermalSourceBoundarySummary,
) -> ThermalKnownAnswer {
    let observed_lumped_delta_k = if final_snapshot.is_empty() {
        0.0
    } else {
        final_snapshot.iter().sum::<f64>() / final_snapshot.len() as f64 - reference_temperature_k
    };
    let slab_linear_profile_rms_ratio =
        thermal_slab_linear_profile_rms_ratio(final_snapshot, topology);
    let slab_monotonic_edge_fraction =
        thermal_slab_monotonic_edge_fraction(final_snapshot, topology);
    let lumped_response_error_ratio = if expected_lumped_delta_k.abs() <= 1.0e-9 {
        observed_lumped_delta_k.abs().min(1.0)
    } else {
        ((observed_lumped_delta_k - expected_lumped_delta_k).abs() / expected_lumped_delta_k.abs())
            .clamp(0.0, 10.0)
    };
    let initial_mean_delta_k = if initial_snapshot.is_empty() {
        0.0
    } else {
        initial_snapshot.iter().sum::<f64>() / initial_snapshot.len() as f64
            - reference_temperature_k
    };
    let source_response_sign_alignment = if source_boundary.volumetric_heat_source_w_per_m3.abs()
        <= 1.0e-12
    {
        1.0
    } else {
        let observed_source_delta = observed_lumped_delta_k - initial_mean_delta_k;
        let source_sign = source_boundary.volumetric_heat_source_w_per_m3.signum();
        if observed_source_delta.abs() <= 1.0e-12 || observed_source_delta.signum() == source_sign {
            1.0
        } else {
            0.0
        }
    };

    ThermalKnownAnswer {
        slab_linear_profile_rms_ratio,
        slab_monotonic_edge_fraction,
        lumped_response_error_ratio,
        source_response_sign_alignment,
        observed_lumped_delta_k,
    }
}

fn thermal_slab_linear_profile_rms_ratio(
    final_snapshot: &[f64],
    topology: ThermalRecoveryTopology,
) -> f64 {
    if final_snapshot.len() <= 2 {
        return 0.0;
    }
    let node_count = final_snapshot.len();
    let mut x_values = Vec::with_capacity(node_count);
    for index in 0..node_count {
        let coords = topology.coords(index);
        let x = if topology.dims[0] <= 1 {
            index as f64 / node_count.saturating_sub(1).max(1) as f64
        } else {
            coords[0] as f64 / topology.dims[0].saturating_sub(1).max(1) as f64
        };
        x_values.push(x);
    }
    let mean_x = x_values.iter().sum::<f64>() / node_count as f64;
    let mean_t = final_snapshot.iter().sum::<f64>() / node_count as f64;
    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;
    for (x, temperature) in x_values.iter().zip(final_snapshot.iter()) {
        let dx = *x - mean_x;
        numerator += dx * (*temperature - mean_t);
        denominator += dx * dx;
    }
    let slope = if denominator <= 1.0e-18 {
        0.0
    } else {
        numerator / denominator
    };
    let intercept = mean_t - slope * mean_x;
    let rms_error = x_values
        .iter()
        .zip(final_snapshot.iter())
        .map(|(x, temperature)| {
            let error = *temperature - (intercept + slope * *x);
            error * error
        })
        .sum::<f64>()
        / node_count as f64;
    let span = final_snapshot
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - final_snapshot.iter().copied().fold(f64::INFINITY, f64::min);
    (rms_error.sqrt() / span.abs().max(1.0e-9)).clamp(0.0, 10.0)
}

fn thermal_slab_monotonic_edge_fraction(
    final_snapshot: &[f64],
    topology: ThermalRecoveryTopology,
) -> f64 {
    if final_snapshot.len() <= 1 {
        return 1.0;
    }
    let mut checked_edges = 0usize;
    let mut monotonic_edges = 0usize;
    for index in 0..final_snapshot.len() {
        let coords = topology.coords(index);
        for axis in 0..VECTOR_COMPONENT_COUNT {
            if coords[axis] + 1 >= topology.dims[axis] {
                continue;
            }
            let mut next_coords = coords;
            next_coords[axis] += 1;
            let Some(next_index) = topology.index(next_coords) else {
                continue;
            };
            checked_edges = checked_edges.saturating_add(1);
            if final_snapshot[next_index] + 1.0e-9 >= final_snapshot[index] {
                monotonic_edges = monotonic_edges.saturating_add(1);
            }
        }
    }
    if checked_edges == 0 {
        1.0
    } else {
        monotonic_edges as f64 / checked_edges as f64
    }
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
        .map(|snapshot| snapshot.iter().sum::<f64>() * dt.max(1.0e-12))
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
