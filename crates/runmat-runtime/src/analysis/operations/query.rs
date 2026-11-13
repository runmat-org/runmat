use super::super::*;
use runmat_analysis_fea::results::observation::SeriesReductionPayload;

mod acoustic_frequency;
mod structural_frequency;

macro_rules! visit_analysis_result_fields {
    ($run_result:expr, $as_ref:ident, [$($borrow:tt)*], $visit:expr) => {{
        for field in $($borrow)* $run_result.run.fields {
            $visit(field);
        }
        for observation in $($borrow)* $run_result.run.observations {
            $visit($($borrow)* observation.field);
        }
        for reduction in $($borrow)* $run_result.run.series_reductions {
            match $($borrow)* reduction.payload {
                SeriesReductionPayload::Values { field, .. } => $visit(field),
                SeriesReductionPayload::Envelope {
                    minimum, maximum, ..
                } => {
                    $visit(minimum);
                    $visit(maximum);
                }
            }
        }
        if let Some(modal) = $run_result.modal_results.$as_ref() {
            for field in $($borrow)* modal.mode_shapes {
                $visit(field);
            }
        }
        if let Some(buckling) = $run_result.buckling_results.$as_ref() {
            for fields in [
                $($borrow)* buckling.mode_shapes,
                $($borrow)* buckling.rotation_mode_shapes,
            ] {
                for field in fields {
                    $visit(field);
                }
            }
        }
        if let Some(thermal) = $run_result.thermal_results.$as_ref() {
            for fields in [
                $($borrow)* thermal.temperature_snapshots,
                $($borrow)* thermal.temperature_gradient_snapshots,
                $($borrow)* thermal.heat_flux_snapshots,
                $($borrow)* thermal.energy_storage_rate_snapshots,
                $($borrow)* thermal.interface_temperature_snapshots,
            ] {
                for field in fields {
                    $visit(field);
                }
            }
        }
        if let Some(transient) = $run_result.transient_results.$as_ref() {
            for fields in [
                $($borrow)* transient.displacement_snapshots,
                $($borrow)* transient.rotation_snapshots,
                $($borrow)* transient.velocity_snapshots,
                $($borrow)* transient.angular_velocity_snapshots,
                $($borrow)* transient.acceleration_snapshots,
                $($borrow)* transient.angular_acceleration_snapshots,
                $($borrow)* transient.strain_snapshots,
                $($borrow)* transient.stress_snapshots,
                $($borrow)* transient.strain_energy_density_snapshots,
                $($borrow)* transient.von_mises_snapshots,
                $($borrow)* transient.kinetic_energy_snapshots,
                $($borrow)* transient.strain_energy_snapshots,
                $($borrow)* transient.residual_norm_snapshots,
                $($borrow)* transient.reaction_force_snapshots,
                $($borrow)* transient.reaction_moment_snapshots,
                $($borrow)* transient.external_work_snapshots,
                $($borrow)* transient.support_work_snapshots,
                $($borrow)* transient.damping_dissipation_snapshots,
                $($borrow)* transient.energy_balance_residual_snapshots,
                $($borrow)* transient.relative_energy_balance_error_snapshots,
                $($borrow)* transient.electro_thermal_temperature_snapshots,
                $($borrow)* transient.electro_thermal_thermal_residual_snapshots,
            ] {
                for field in fields {
                    $visit(field);
                }
            }
        }
        if let Some(nonlinear) = $run_result.nonlinear_results.$as_ref() {
            for fields in [
                $($borrow)* nonlinear.displacement_snapshots,
                $($borrow)* nonlinear.rotation_snapshots,
                $($borrow)* nonlinear.von_mises_snapshots,
                $($borrow)* nonlinear.plastic_strain_snapshots,
                $($borrow)* nonlinear.equivalent_plastic_strain_snapshots,
                $($borrow)* nonlinear.contact_pressure_snapshots,
                $($borrow)* nonlinear.contact_gap_snapshots,
                $($borrow)* nonlinear.load_factor_snapshots,
                $($borrow)* nonlinear.residual_norm_snapshots,
                $($borrow)* nonlinear.thermo_mechanical_temperature_snapshots,
                $($borrow)* nonlinear.thermo_mechanical_thermal_strain_snapshots,
                $($borrow)* nonlinear.thermo_mechanical_thermal_stress_snapshots,
                $($borrow)* nonlinear.thermo_mechanical_displacement_snapshots,
                $($borrow)* nonlinear.thermo_mechanical_von_mises_snapshots,
                $($borrow)* nonlinear.thermo_mechanical_coupling_residual_snapshots,
                $($borrow)* nonlinear.electro_thermal_temperature_snapshots,
                $($borrow)* nonlinear.electro_thermal_thermal_residual_snapshots,
            ] {
                for field in fields {
                    $visit(field);
                }
            }
        }
    }};
}

pub(crate) fn visit_analysis_result_fields(
    run_result: &AnalysisRunResult,
    mut visit: impl FnMut(&AnalysisField),
) {
    visit_analysis_result_fields!(run_result, as_ref, [&], visit);
}

pub(crate) fn visit_analysis_result_fields_mut(
    run_result: &mut AnalysisRunResult,
    mut visit: impl FnMut(&mut AnalysisField),
) {
    visit_analysis_result_fields!(run_result, as_mut, [& mut], visit);
}

pub(crate) fn collect_analysis_result_fields(run_result: &AnalysisRunResult) -> Vec<AnalysisField> {
    let mut fields = Vec::new();
    let mut seen = HashSet::new();
    visit_analysis_result_fields(run_result, |field| {
        push_analysis_result_field(&mut fields, &mut seen, field);
    });
    fields
}

fn push_analysis_result_field(
    fields: &mut Vec<AnalysisField>,
    seen: &mut HashSet<String>,
    field: &AnalysisField,
) {
    if !seen.insert(field.field_id.clone()) {
        return;
    }
    fields.push(field.clone());
}

fn filter_analysis_fields_by_indices(
    fields: &[AnalysisField],
    indices: &[usize],
) -> Vec<AnalysisField> {
    if fields.is_empty() {
        return Vec::new();
    }
    indices
        .iter()
        .filter_map(|index| fields.get(*index).cloned())
        .collect()
}

pub(crate) fn analysis_run_field_ids(run_result: &AnalysisRunResult) -> Vec<String> {
    collect_analysis_result_fields(run_result)
        .into_iter()
        .map(|field| field.field_id)
        .collect()
}

pub fn analysis_results_op(
    run_result: &AnalysisRunResult,
    query: AnalysisResultsQuery,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisResultsData>, OperationErrorEnvelope> {
    analysis_results_op_with_field_inventory(run_result, query, context, None)
}

fn analysis_results_op_with_field_inventory(
    run_result: &AnalysisRunResult,
    query: AnalysisResultsQuery,
    context: OperationContext,
    field_inventory: Option<Vec<AnalysisField>>,
) -> Result<OperationEnvelope<AnalysisResultsData>, OperationErrorEnvelope> {
    let mut collected_fields =
        field_inventory.unwrap_or_else(|| collect_analysis_result_fields(run_result));

    if !query.include_fields.is_empty() {
        let mut filtered = Vec::new();
        for requested in &query.include_fields {
            let Some(field) = collected_fields
                .iter()
                .find(|field| &field.field_id == requested)
            else {
                return Err(operation_error(
                    ANALYSIS_RESULTS_OPERATION,
                    ANALYSIS_RESULTS_OP_VERSION,
                    &context,
                    OperationErrorSpec {
                        error_code: "RM.FEA.RESULTS.FIELD_NOT_FOUND",
                        error_type: OperationErrorType::Input,
                        retryable: false,
                        severity: OperationErrorSeverity::Error,
                    },
                    format!("requested FEA field '{requested}' was not produced by run"),
                    BTreeMap::from([
                        ("requested_field".to_string(), requested.clone()),
                        (
                            "available_fields".to_string(),
                            collected_fields
                                .iter()
                                .map(|field| field.field_id.clone())
                                .collect::<Vec<_>>()
                                .join(","),
                        ),
                    ]),
                ));
            };
            filtered.push(field.clone());
        }
        collected_fields = filtered;
    }
    let field_descriptors = collected_fields
        .iter()
        .map(AnalysisFieldDescriptor::from_field)
        .collect::<Vec<_>>();

    let (
        mode_count,
        available_mode_indices,
        min_frequency_hz,
        max_frequency_hz,
        max_modal_residual_norm,
        first_mode_converged,
    ) = if let Some(modal) = run_result.modal_results.as_ref() {
        let frequencies_hz = modal.frequencies_hz();
        let residual_norms = modal.residual_norms();
        let count = frequencies_hz.len();
        let max_modal_residual_norm = residual_norms.iter().copied().reduce(f64::max);
        let first_mode_converged = residual_norms.first().copied().map(|v| v <= 1.0e-6);
        let (min_frequency_hz, max_frequency_hz) = if count == 0 {
            (None, None)
        } else {
            let mut min_value = f64::INFINITY;
            let mut max_value = f64::NEG_INFINITY;
            for value in frequencies_hz.iter().copied() {
                min_value = min_value.min(value);
                max_value = max_value.max(value);
            }
            (Some(min_value), Some(max_value))
        };
        (
            count,
            (0..count).collect(),
            min_frequency_hz,
            max_frequency_hz,
            max_modal_residual_norm,
            first_mode_converged,
        )
    } else {
        (0, Vec::new(), None, None, None, None)
    };

    let (
        buckling_mode_count,
        available_buckling_mode_indices,
        minimum_buckling_load_factor,
        max_buckling_residual_norm,
    ) = if let Some(buckling) = run_result.buckling_results.as_ref() {
        (
            buckling.mode_indices.len(),
            buckling.mode_indices.clone(),
            buckling.load_factors.iter().copied().reduce(f64::min),
            buckling.residual_norms.iter().copied().reduce(f64::max),
        )
    } else {
        (0, Vec::new(), None, None)
    };

    let (
        snapshot_count,
        time_start_s,
        time_end_s,
        max_transient_residual_norm,
        final_step_converged,
    ) = if let Some(transient) = run_result.transient_results.as_ref() {
        let count = transient
            .time_points_s
            .len()
            .min(transient.displacement_snapshots.len());
        let max_residual = transient.residual_norms.iter().copied().reduce(f64::max);
        let final_step_converged = max_residual.map(|value| value <= 1.0e-6);
        if count == 0 {
            (0, None, None, max_residual, final_step_converged)
        } else {
            (
                count,
                transient.time_points_s.first().copied(),
                transient.time_points_s.get(count - 1).copied(),
                max_residual,
                final_step_converged,
            )
        }
    } else if let Some(thermal) = run_result.thermal_results.as_ref() {
        let count = thermal
            .time_points_s
            .len()
            .min(thermal.temperature_snapshots.len());
        let max_residual = thermal.residual_norms.iter().copied().reduce(f64::max);
        let final_step_converged = max_residual.map(|value| value <= 1.0e-6);
        if count == 0 {
            (0, None, None, max_residual, final_step_converged)
        } else {
            (
                count,
                thermal.time_points_s.first().copied(),
                thermal.time_points_s.get(count - 1).copied(),
                max_residual,
                final_step_converged,
            )
        }
    } else {
        (0, None, None, None, None)
    };

    let (
        increment_count,
        failed_increment_count,
        max_nonlinear_residual_norm,
        max_nonlinear_increment_norm,
        max_nonlinear_iteration_count,
        final_increment_converged,
        nonlinear_line_search_backtracks,
        nonlinear_max_backtracks_per_increment,
        nonlinear_tangent_rebuild_count,
        nonlinear_iteration_spike_count,
        nonlinear_convergence_stall_count,
        nonlinear_backtrack_burst_count,
    ) = if let Some(nonlinear) = run_result.nonlinear_results.as_ref() {
        let count = nonlinear.load_factors.len();
        let max_residual = nonlinear.residual_norms.iter().copied().reduce(f64::max);
        let max_increment_norm = nonlinear.increment_norms.iter().copied().reduce(f64::max);
        let max_iteration_count = nonlinear.iteration_counts.iter().copied().max();
        let final_converged =
            max_residual.map(|value| value <= 1.0e-6 && nonlinear.failed_increments == 0);
        (
            count,
            Some(nonlinear.failed_increments),
            max_residual,
            max_increment_norm,
            max_iteration_count,
            final_converged,
            Some(nonlinear.line_search_backtracks),
            Some(nonlinear.max_line_search_backtracks_per_increment),
            Some(nonlinear.tangent_rebuild_count),
            Some(nonlinear.iteration_spike_count),
            Some(nonlinear.convergence_stall_count),
            Some(nonlinear.backtrack_burst_count),
        )
    } else {
        (
            0, None, None, None, None, None, None, None, None, None, None, None,
        )
    };

    let thermo_coupling_enabled =
        diagnostic_metric_bool(&run_result.run.diagnostics, "FEA_TM_COUPLING", "enabled");
    let thermo_coupling_fingerprint = diagnostic_metric_u64(
        &run_result.run.diagnostics,
        "FEA_TM_COUPLING",
        "coupling_fingerprint",
    );
    let thermo_constitutive_temperature_factor = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_COUPLING",
        "constitutive_temperature_factor",
    );
    let thermo_effective_modulus_scale = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_COUPLING",
        "effective_modulus_scale",
    );
    let thermo_constitutive_material_spread_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_COUPLING",
        "constitutive_material_spread_ratio",
    );
    let thermo_assignment_heterogeneity_index = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_COUPLING",
        "assignment_heterogeneity_index",
    );
    let thermo_region_delta_count = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_COUPLING",
        "region_delta_count",
    );
    let thermo_spatial_coverage_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_COUPLING",
        "spatial_coverage_ratio",
    );
    let thermo_nonlinear_severity = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_TM_NONLINEAR",
        "severity_peak",
    )
    .or_else(|| diagnostic_metric(&run_result.run.diagnostics, "FEA_TM_NONLINEAR", "severity"));
    let electro_thermal_coupling_enabled =
        diagnostic_metric_bool(&run_result.run.diagnostics, "FEA_ET_COUPLING", "enabled");
    let electro_thermal_coupling_fingerprint = diagnostic_metric_u64(
        &run_result.run.diagnostics,
        "FEA_ET_COUPLING",
        "coupling_fingerprint",
    );
    let electro_joule_heating_scale = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_ET_COUPLING",
        "joule_heating_scale",
    );
    let electro_conductivity_spread_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_ET_COUPLING",
        "conductivity_spread_ratio",
    );
    let electro_transient_severity = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_ET_TRANSIENT",
        "severity_peak",
    )
    .or_else(|| diagnostic_metric(&run_result.run.diagnostics, "FEA_ET_TRANSIENT", "severity"));
    let electro_nonlinear_severity = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_ET_NONLINEAR",
        "severity_peak",
    )
    .or_else(|| diagnostic_metric(&run_result.run.diagnostics, "FEA_ET_NONLINEAR", "severity"));
    let plastic_nonlinear_severity = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_PLASTIC_NONLINEAR",
        "severity_peak",
    )
    .or_else(|| {
        diagnostic_metric(
            &run_result.run.diagnostics,
            "FEA_PLASTIC_NONLINEAR",
            "severity",
        )
    });
    let plastic_nonlinear_severity_mean = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_PLASTIC_NONLINEAR",
        "severity_mean",
    );
    let plastic_load_realization_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_PLASTIC_NONLINEAR",
        "load_realization_ratio",
    );
    let plastic_load_amplification_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_PLASTIC_NONLINEAR",
        "load_amplification_ratio",
    );
    let contact_nonlinear_severity = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_CONTACT_NONLINEAR",
        "severity_peak",
    )
    .or_else(|| {
        diagnostic_metric(
            &run_result.run.diagnostics,
            "FEA_CONTACT_NONLINEAR",
            "severity",
        )
    });
    let contact_nonlinear_severity_mean = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_CONTACT_NONLINEAR",
        "severity_mean",
    );
    let contact_load_realization_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_CONTACT_NONLINEAR",
        "load_realization_ratio",
    );
    let contact_load_amplification_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_CONTACT_NONLINEAR",
        "load_amplification_ratio",
    );
    let thermal_max_relative_residual_norm = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_THERMAL_STABILITY",
        "max_relative_residual_norm",
    );
    let thermal_min_temperature_k = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_THERMAL_STABILITY",
        "minimum_temperature_k",
    );
    let thermal_max_temperature_k = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_THERMAL_STABILITY",
        "maximum_temperature_k",
    );
    let thermal_conductivity_spread_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_THERMAL_CONSTITUTIVE",
        "conductivity_spread_ratio",
    );
    let thermal_heat_capacity_spread_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_THERMAL_CONSTITUTIVE",
        "heat_capacity_spread_ratio",
    );
    let electromagnetic_enabled =
        diagnostic_metric_bool(&run_result.run.diagnostics, "FEA_EM_STATIC", "enabled");
    let electromagnetic_formulation_coverage_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_FORMULATION",
        "formulation_coverage_ratio",
    );
    let electromagnetic_magnetostatic_curl_curl_coverage_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_FORMULATION",
        "magnetostatic_curl_curl_coverage_ratio",
    );
    let electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_FORMULATION",
        "magnetoquasistatic_eddy_current_coverage_ratio",
    );
    let electromagnetic_full_wave_displacement_current_coverage_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_FORMULATION",
        "full_wave_displacement_current_coverage_ratio",
    );
    let electromagnetic_displacement_to_conduction_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_FORMULATION",
        "displacement_to_conduction_ratio",
    );
    let electromagnetic_material_frequency_response_coverage_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_FORMULATION",
        "material_frequency_response_coverage_ratio",
    );
    let electromagnetic_reference_frequency_hz = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "reference_frequency_hz",
    );
    let electromagnetic_applied_current_a = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "applied_current_a",
    );
    let electromagnetic_solve_quality = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "solve_quality",
    );
    let electromagnetic_conductivity_spread_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "conductivity_spread_ratio",
    );
    let electromagnetic_relative_permittivity_spread_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "relative_permittivity_spread_ratio",
    );
    let electromagnetic_relative_permeability_spread_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "relative_permeability_spread_ratio",
    );
    let electromagnetic_material_heterogeneity_index = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "electromagnetic_material_heterogeneity_index",
    );
    let electromagnetic_assignment_coverage_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "assignment_coverage_ratio",
    );
    let electromagnetic_assigned_coefficient_coverage_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "assigned_coefficient_coverage_ratio",
    );
    let electromagnetic_region_coefficient_contrast_index = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "region_coefficient_contrast_index",
    );
    let electromagnetic_condition_number_estimate = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "condition_number_estimate",
    );
    let electromagnetic_source_realization_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "source_realization_ratio",
    );
    let electromagnetic_source_region_coverage_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "source_region_coverage_ratio",
    );
    let electromagnetic_source_material_alignment_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "source_material_alignment_ratio",
    );
    let electromagnetic_source_localization_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "source_localization_ratio",
    );
    let electromagnetic_source_overlap_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "source_overlap_ratio",
    );
    let electromagnetic_source_interference_index = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "source_interference_index",
    );
    let electromagnetic_boundary_anchor_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "boundary_anchor_ratio",
    );
    let electromagnetic_boundary_condition_localization_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "boundary_condition_localization_ratio",
    );
    let electromagnetic_ground_anchor_effectiveness_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "ground_anchor_effectiveness_ratio",
    );
    let electromagnetic_insulation_leakage_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "insulation_leakage_ratio",
    );
    let electromagnetic_flux_divergence_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "flux_divergence_ratio",
    );
    let electromagnetic_energy_imbalance_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "energy_imbalance_ratio",
    );
    let electromagnetic_boundary_energy_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "boundary_energy_ratio",
    );
    let electromagnetic_boundary_penalty_conditioning_contribution = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "boundary_penalty_conditioning_contribution",
    );
    let electromagnetic_source_region_energy_consistency_ratio = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SOURCE_ENERGY",
        "source_region_energy_consistency_ratio",
    );
    let electromagnetic_real_residual_norm = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "real_residual_norm",
    );
    let electromagnetic_imag_residual_norm = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_STATIC",
        "imag_residual_norm",
    );
    let electromagnetic_sweep_count =
        diagnostic_metric(&run_result.run.diagnostics, "FEA_EM_SWEEP", "sweep_count");
    let electromagnetic_resonance_peak_frequency_hz = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SWEEP",
        "resonance_peak_frequency_hz",
    );
    let electromagnetic_resonance_peak_flux_density = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SWEEP",
        "resonance_peak_flux_density",
    );
    let electromagnetic_resonance_bandwidth_hz = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SWEEP",
        "resonance_bandwidth_hz",
    );
    let electromagnetic_resonance_quality_factor = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SWEEP",
        "resonance_quality_factor",
    );
    let electromagnetic_resonance_flux_gain = diagnostic_metric(
        &run_result.run.diagnostics,
        "FEA_EM_SWEEP",
        "resonance_flux_gain",
    );

    let summary = AnalysisResultsSummary {
        field_count: field_descriptors.len(),
        total_elements: field_descriptors
            .iter()
            .map(|field| field.element_count)
            .sum(),
        mode_count,
        available_mode_indices,
        min_frequency_hz,
        max_frequency_hz,
        max_modal_residual_norm,
        first_mode_converged,
        buckling_mode_count,
        available_buckling_mode_indices,
        minimum_buckling_load_factor,
        max_buckling_residual_norm,
        frequency_sample_count: 0,
        available_frequencies_hz: Vec::new(),
        max_frequency_solver_residual: None,
        max_frequency_balance_error: None,
        modal_truncation_balance_error: None,
        modal_truncation_balance_frequency_hz: None,
        snapshot_count,
        time_start_s,
        time_end_s,
        max_transient_residual_norm,
        final_step_converged,
        increment_count,
        failed_increment_count,
        max_nonlinear_residual_norm,
        max_nonlinear_increment_norm,
        max_nonlinear_iteration_count,
        final_increment_converged,
        nonlinear_line_search_backtracks,
        nonlinear_max_backtracks_per_increment,
        nonlinear_tangent_rebuild_count,
        nonlinear_iteration_spike_count,
        nonlinear_convergence_stall_count,
        nonlinear_backtrack_burst_count,
        thermo_coupling_enabled,
        thermo_coupling_fingerprint,
        thermo_constitutive_temperature_factor,
        thermo_effective_modulus_scale,
        thermo_constitutive_material_spread_ratio,
        thermo_assignment_heterogeneity_index,
        thermo_region_delta_count,
        thermo_spatial_coverage_ratio,
        thermo_nonlinear_severity,
        electro_thermal_coupling_enabled,
        electro_thermal_coupling_fingerprint,
        electro_joule_heating_scale,
        electro_conductivity_spread_ratio,
        electro_transient_severity,
        electro_nonlinear_severity,
        plastic_nonlinear_severity,
        plastic_nonlinear_severity_mean,
        plastic_load_realization_ratio,
        plastic_load_amplification_ratio,
        contact_nonlinear_severity,
        contact_nonlinear_severity_mean,
        contact_load_realization_ratio,
        contact_load_amplification_ratio,
        thermal_max_relative_residual_norm,
        thermal_min_temperature_k,
        thermal_max_temperature_k,
        thermal_conductivity_spread_ratio,
        thermal_heat_capacity_spread_ratio,
        electromagnetic_enabled,
        electromagnetic_formulation_coverage_ratio,
        electromagnetic_magnetostatic_curl_curl_coverage_ratio,
        electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio,
        electromagnetic_full_wave_displacement_current_coverage_ratio,
        electromagnetic_displacement_to_conduction_ratio,
        electromagnetic_material_frequency_response_coverage_ratio,
        electromagnetic_reference_frequency_hz,
        electromagnetic_applied_current_a,
        electromagnetic_solve_quality,
        electromagnetic_conductivity_spread_ratio,
        electromagnetic_relative_permittivity_spread_ratio,
        electromagnetic_relative_permeability_spread_ratio,
        electromagnetic_material_heterogeneity_index,
        electromagnetic_assignment_coverage_ratio,
        electromagnetic_assigned_coefficient_coverage_ratio,
        electromagnetic_region_coefficient_contrast_index,
        electromagnetic_condition_number_estimate,
        electromagnetic_source_realization_ratio,
        electromagnetic_source_region_coverage_ratio,
        electromagnetic_source_material_alignment_ratio,
        electromagnetic_source_localization_ratio,
        electromagnetic_source_overlap_ratio,
        electromagnetic_source_interference_index,
        electromagnetic_boundary_anchor_ratio,
        electromagnetic_boundary_condition_localization_ratio,
        electromagnetic_ground_anchor_effectiveness_ratio,
        electromagnetic_insulation_leakage_ratio,
        electromagnetic_flux_divergence_ratio,
        electromagnetic_energy_imbalance_ratio,
        electromagnetic_boundary_energy_ratio,
        electromagnetic_boundary_penalty_conditioning_contribution,
        electromagnetic_source_region_energy_consistency_ratio,
        electromagnetic_real_residual_norm,
        electromagnetic_imag_residual_norm,
        electromagnetic_sweep_count,
        electromagnetic_resonance_peak_frequency_hz,
        electromagnetic_resonance_peak_flux_density,
        electromagnetic_resonance_bandwidth_hz,
        electromagnetic_resonance_quality_factor,
        electromagnetic_resonance_flux_gain,
    };

    let modal_results = if query.include_modal_results && query.include_field_values {
        if let Some(modal) = run_result.modal_results.as_ref() {
            if query.mode_indices.is_empty() {
                Some(modal.clone())
            } else {
                for &index in &query.mode_indices {
                    if modal.basis.modes().get(index).is_none() {
                        return Err(operation_error(
                            ANALYSIS_RESULTS_OPERATION,
                            ANALYSIS_RESULTS_OP_VERSION,
                            &context,
                            OperationErrorSpec {
                                error_code: "RM.FEA.RESULTS.MODE_NOT_FOUND",
                                error_type: OperationErrorType::Input,
                                retryable: false,
                                severity: OperationErrorSeverity::Error,
                            },
                            format!("requested modal mode index '{index}' was not produced by run"),
                            BTreeMap::from([
                                ("requested_mode_index".to_string(), index.to_string()),
                                (
                                    "available_mode_count".to_string(),
                                    modal.basis.modes().len().to_string(),
                                ),
                            ]),
                        ));
                    }
                }
                Some(ModalResultsData {
                    basis: modal.basis.clone(),
                    mode_indices: query.mode_indices.clone(),
                    mode_shapes: query
                        .mode_indices
                        .iter()
                        .filter_map(|index| modal.mode_shapes.get(*index).cloned())
                        .collect(),
                    mode_units: modal.mode_units,
                    frequency_basis: modal.frequency_basis,
                })
            }
        } else {
            None
        }
    } else {
        None
    };

    let transient_results = if query.include_transient_results && query.include_field_values {
        if let Some(transient) = run_result.transient_results.as_ref() {
            transient.validate().map_err(|detail| {
                operation_error(
                    ANALYSIS_RESULTS_OPERATION,
                    ANALYSIS_RESULTS_OP_VERSION,
                    &context,
                    OperationErrorSpec {
                        error_code: "RM.FEA.RESULTS.INVALID_TRANSIENT_INVENTORY",
                        error_type: OperationErrorType::Contract,
                        retryable: false,
                        severity: OperationErrorSeverity::Error,
                    },
                    detail,
                    BTreeMap::new(),
                )
            })?;
            if query.transient_snapshot_indices.is_empty() {
                Some(transient.clone())
            } else {
                let mut time_points_s = Vec::with_capacity(query.transient_snapshot_indices.len());
                let mut displacement_snapshots =
                    Vec::with_capacity(query.transient_snapshot_indices.len());
                let rotation_snapshots = filter_analysis_fields_by_indices(
                    &transient.rotation_snapshots,
                    &query.transient_snapshot_indices,
                );
                let mut velocity_snapshots =
                    Vec::with_capacity(query.transient_snapshot_indices.len());
                let angular_velocity_snapshots = filter_analysis_fields_by_indices(
                    &transient.angular_velocity_snapshots,
                    &query.transient_snapshot_indices,
                );
                let mut acceleration_snapshots =
                    Vec::with_capacity(query.transient_snapshot_indices.len());
                let angular_acceleration_snapshots = filter_analysis_fields_by_indices(
                    &transient.angular_acceleration_snapshots,
                    &query.transient_snapshot_indices,
                );
                let strain_snapshots = filter_analysis_fields_by_indices(
                    &transient.strain_snapshots,
                    &query.transient_snapshot_indices,
                );
                let stress_snapshots = filter_analysis_fields_by_indices(
                    &transient.stress_snapshots,
                    &query.transient_snapshot_indices,
                );
                let strain_energy_density_snapshots = filter_analysis_fields_by_indices(
                    &transient.strain_energy_density_snapshots,
                    &query.transient_snapshot_indices,
                );
                let von_mises_snapshots = filter_analysis_fields_by_indices(
                    &transient.von_mises_snapshots,
                    &query.transient_snapshot_indices,
                );
                let mut kinetic_energy_snapshots =
                    Vec::with_capacity(query.transient_snapshot_indices.len());
                let mut strain_energy_snapshots =
                    Vec::with_capacity(query.transient_snapshot_indices.len());
                let mut residual_norm_snapshots =
                    Vec::with_capacity(query.transient_snapshot_indices.len());
                let reaction_force_snapshots = filter_analysis_fields_by_indices(
                    &transient.reaction_force_snapshots,
                    &query.transient_snapshot_indices,
                );
                let reaction_moment_snapshots = filter_analysis_fields_by_indices(
                    &transient.reaction_moment_snapshots,
                    &query.transient_snapshot_indices,
                );
                let external_work_snapshots = filter_analysis_fields_by_indices(
                    &transient.external_work_snapshots,
                    &query.transient_snapshot_indices,
                );
                let support_work_snapshots = filter_analysis_fields_by_indices(
                    &transient.support_work_snapshots,
                    &query.transient_snapshot_indices,
                );
                let damping_dissipation_snapshots = filter_analysis_fields_by_indices(
                    &transient.damping_dissipation_snapshots,
                    &query.transient_snapshot_indices,
                );
                let energy_balance_residual_snapshots = filter_analysis_fields_by_indices(
                    &transient.energy_balance_residual_snapshots,
                    &query.transient_snapshot_indices,
                );
                let relative_energy_balance_error_snapshots = filter_analysis_fields_by_indices(
                    &transient.relative_energy_balance_error_snapshots,
                    &query.transient_snapshot_indices,
                );
                let dynamic_balance_snapshots = query
                    .transient_snapshot_indices
                    .iter()
                    .filter_map(|index| transient.dynamic_balance_snapshots.get(*index).cloned())
                    .collect();
                let mut residual_norms = Vec::with_capacity(query.transient_snapshot_indices.len());
                let electro_thermal_temperature_snapshots = filter_analysis_fields_by_indices(
                    &transient.electro_thermal_temperature_snapshots,
                    &query.transient_snapshot_indices,
                );
                let electro_thermal_thermal_residual_snapshots = filter_analysis_fields_by_indices(
                    &transient.electro_thermal_thermal_residual_snapshots,
                    &query.transient_snapshot_indices,
                );

                for &index in &query.transient_snapshot_indices {
                    let time_point = transient.time_points_s.get(index).copied().ok_or_else(|| {
                        operation_error(
                            ANALYSIS_RESULTS_OPERATION,
                            ANALYSIS_RESULTS_OP_VERSION,
                            &context,
                            OperationErrorSpec {
                                error_code: "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
                                error_type: OperationErrorType::Input,
                                retryable: false,
                                severity: OperationErrorSeverity::Error,
                            },
                            format!(
                                "requested transient snapshot index '{index}' was not produced by run"
                            ),
                            BTreeMap::from([
                                ("requested_snapshot_index".to_string(), index.to_string()),
                                (
                                    "available_snapshot_count".to_string(),
                                    transient.time_points_s.len().to_string(),
                                ),
                            ]),
                        )
                    })?;
                    let snapshot = transient
                        .displacement_snapshots
                        .get(index)
                        .cloned()
                        .ok_or_else(|| {
                            operation_error(
                                ANALYSIS_RESULTS_OPERATION,
                                ANALYSIS_RESULTS_OP_VERSION,
                                &context,
                                OperationErrorSpec {
                                    error_code: "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
                                    error_type: OperationErrorType::Input,
                                    retryable: false,
                                    severity: OperationErrorSeverity::Error,
                                },
                                format!(
                                    "requested transient snapshot index '{index}' is missing displacement data"
                                ),
                                BTreeMap::from([
                                    ("requested_snapshot_index".to_string(), index.to_string()),
                                    (
                                        "available_displacement_snapshot_count".to_string(),
                                        transient.displacement_snapshots.len().to_string(),
                                    ),
                                ]),
                            )
                        })?;
                    let velocity = transient.velocity_snapshots.get(index).cloned().ok_or_else(|| {
                        operation_error(
                            ANALYSIS_RESULTS_OPERATION,
                            ANALYSIS_RESULTS_OP_VERSION,
                            &context,
                            OperationErrorSpec {
                                error_code: "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
                                error_type: OperationErrorType::Input,
                                retryable: false,
                                severity: OperationErrorSeverity::Error,
                            },
                            format!(
                                "requested transient snapshot index '{index}' is missing velocity data"
                            ),
                            BTreeMap::from([
                                ("requested_snapshot_index".to_string(), index.to_string()),
                                (
                                    "available_velocity_snapshot_count".to_string(),
                                    transient.velocity_snapshots.len().to_string(),
                                ),
                            ]),
                        )
                    })?;
                    let acceleration =
                        transient
                            .acceleration_snapshots
                            .get(index)
                            .cloned()
                            .ok_or_else(|| {
                                operation_error(
                                    ANALYSIS_RESULTS_OPERATION,
                                    ANALYSIS_RESULTS_OP_VERSION,
                                    &context,
                                    OperationErrorSpec {
                                        error_code:
                                            "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
                                        error_type: OperationErrorType::Input,
                                        retryable: false,
                                        severity: OperationErrorSeverity::Error,
                                    },
                                    format!(
                                        "requested transient snapshot index '{index}' is missing acceleration data"
                                    ),
                                    BTreeMap::from([
                                        ("requested_snapshot_index".to_string(), index.to_string()),
                                        (
                                            "available_acceleration_snapshot_count".to_string(),
                                            transient.acceleration_snapshots.len().to_string(),
                                        ),
                                    ]),
                                )
                            })?;
                    let kinetic_energy =
                        transient
                            .kinetic_energy_snapshots
                            .get(index)
                            .cloned()
                            .ok_or_else(|| {
                                operation_error(
                                    ANALYSIS_RESULTS_OPERATION,
                                    ANALYSIS_RESULTS_OP_VERSION,
                                    &context,
                                    OperationErrorSpec {
                                        error_code:
                                            "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
                                        error_type: OperationErrorType::Input,
                                        retryable: false,
                                        severity: OperationErrorSeverity::Error,
                                    },
                                    format!(
                                        "requested transient snapshot index '{index}' is missing kinetic energy data"
                                    ),
                                    BTreeMap::from([
                                        ("requested_snapshot_index".to_string(), index.to_string()),
                                        (
                                            "available_kinetic_energy_snapshot_count".to_string(),
                                            transient.kinetic_energy_snapshots.len().to_string(),
                                        ),
                                    ]),
                                )
                            })?;
                    let strain_energy =
                        transient
                            .strain_energy_snapshots
                            .get(index)
                            .cloned()
                            .ok_or_else(|| {
                                operation_error(
                                    ANALYSIS_RESULTS_OPERATION,
                                    ANALYSIS_RESULTS_OP_VERSION,
                                    &context,
                                    OperationErrorSpec {
                                        error_code:
                                            "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
                                        error_type: OperationErrorType::Input,
                                        retryable: false,
                                        severity: OperationErrorSeverity::Error,
                                    },
                                    format!(
                                        "requested transient snapshot index '{index}' is missing strain energy data"
                                    ),
                                    BTreeMap::from([
                                        ("requested_snapshot_index".to_string(), index.to_string()),
                                        (
                                            "available_strain_energy_snapshot_count".to_string(),
                                            transient.strain_energy_snapshots.len().to_string(),
                                        ),
                                    ]),
                                )
                            })?;
                    let residual_norm_snapshot = transient
                        .residual_norm_snapshots
                        .get(index)
                        .cloned()
                        .ok_or_else(|| {
                            operation_error(
                                ANALYSIS_RESULTS_OPERATION,
                                ANALYSIS_RESULTS_OP_VERSION,
                                &context,
                                OperationErrorSpec {
                                    error_code: "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
                                    error_type: OperationErrorType::Input,
                                    retryable: false,
                                    severity: OperationErrorSeverity::Error,
                                },
                                format!(
                                    "requested transient snapshot index '{index}' is missing residual field data"
                                ),
                                BTreeMap::from([
                                    ("requested_snapshot_index".to_string(), index.to_string()),
                                    (
                                        "available_residual_snapshot_count".to_string(),
                                        transient.residual_norm_snapshots.len().to_string(),
                                    ),
                                ]),
                            )
                        })?;

                    {
                        let residual = transient.residual_norms.get(index).copied().ok_or_else(|| {
                            operation_error(
                                ANALYSIS_RESULTS_OPERATION,
                                ANALYSIS_RESULTS_OP_VERSION,
                                &context,
                                OperationErrorSpec {
                                    error_code: "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND",
                                    error_type: OperationErrorType::Input,
                                    retryable: false,
                                    severity: OperationErrorSeverity::Error,
                                },
                                format!(
                                    "requested transient snapshot index '{index}' is missing residual data"
                                ),
                                BTreeMap::from([
                                    ("requested_snapshot_index".to_string(), index.to_string()),
                                    (
                                        "available_residual_count".to_string(),
                                        transient.residual_norms.len().to_string(),
                                    ),
                                ]),
                            )
                        })?;
                        residual_norms.push(residual);
                    }

                    time_points_s.push(time_point);
                    displacement_snapshots.push(snapshot);
                    velocity_snapshots.push(velocity);
                    acceleration_snapshots.push(acceleration);
                    kinetic_energy_snapshots.push(kinetic_energy);
                    strain_energy_snapshots.push(strain_energy);
                    residual_norm_snapshots.push(residual_norm_snapshot);
                }

                Some(TransientResultsData {
                    transient_payload_version: transient.transient_payload_version.clone(),
                    initial_state: transient.initial_state.clone(),
                    time_points_s,
                    displacement_snapshots,
                    rotation_snapshots,
                    velocity_snapshots,
                    angular_velocity_snapshots,
                    acceleration_snapshots,
                    angular_acceleration_snapshots,
                    strain_snapshots,
                    stress_snapshots,
                    strain_energy_density_snapshots,
                    von_mises_snapshots,
                    kinetic_energy_snapshots,
                    strain_energy_snapshots,
                    residual_norm_snapshots,
                    reaction_force_snapshots,
                    reaction_moment_snapshots,
                    external_work_snapshots,
                    support_work_snapshots,
                    damping_dissipation_snapshots,
                    energy_balance_residual_snapshots,
                    relative_energy_balance_error_snapshots,
                    dynamic_balance_snapshots,
                    electro_thermal_temperature_snapshots,
                    electro_thermal_thermal_residual_snapshots,
                    residual_norms,
                    integration_method: transient.integration_method,
                })
            }
        } else {
            None
        }
    } else {
        None
    };

    let thermal_results = if query.include_field_values {
        run_result.thermal_results.clone()
    } else {
        None
    };

    let buckling_results = if query.include_buckling_results && query.include_field_values {
        run_result.buckling_results.clone()
    } else {
        None
    };

    let nonlinear_results = if query.include_nonlinear_results && query.include_field_values {
        run_result.nonlinear_results.clone()
    } else {
        None
    };
    let fields = if query.include_field_values {
        collected_fields
    } else {
        Vec::new()
    };

    let data = AnalysisResultsData {
        identity: run_result.run.identity.clone(),
        field_descriptors,
        fields,
        modal_results,
        buckling_results,
        thermal_results,
        transient_results,
        nonlinear_results,
        structural_frequency_results: None,
        acoustic_frequency_results: None,
        diagnostics: if query.include_diagnostics {
            if query.diagnostic_codes.is_empty() {
                Some(run_result.run.diagnostics.clone())
            } else {
                Some(
                    run_result
                        .run
                        .diagnostics
                        .iter()
                        .filter(|diag| query.diagnostic_codes.iter().any(|code| code == &diag.code))
                        .cloned()
                        .collect(),
                )
            }
        } else {
            None
        },
        run_status: run_result.run_status,
        numerically_acceptable: run_result.numerically_acceptable,
        quality_reasons: run_result.quality_reasons.clone(),
        provenance: run_result.provenance.clone(),
        summary,
    };

    Ok(OperationEnvelope::new(
        ANALYSIS_RESULTS_OPERATION,
        ANALYSIS_RESULTS_OP_VERSION,
        &context,
        data,
    ))
}

/// Field payload a default results object holds inline; larger runs page values through
/// `fea.field`.
pub const RESULTS_INLINE_FIELD_BYTES: u64 = 16 * 1024 * 1024;

/// The default results view of a stored run: values of the selected fields inline when they fit
/// [`RESULTS_INLINE_FIELD_BYTES`], descriptors only otherwise. Only a caller that chose no
/// field-value option applies this; an explicit field-value request is never narrowed.
pub fn bounded_default_results_query(
    run_id: &str,
    mut query: AnalysisResultsQuery,
) -> Result<AnalysisResultsQuery, String> {
    if !query.include_field_values {
        return Ok(query);
    }
    let run_id = storage::resolve_run_attempt(run_id)?;
    let Some(inventory) = storage::load_field_inventory(&run_id)? else {
        return Ok(query);
    };
    let payload = inventory
        .iter()
        .filter(|field| {
            query.include_fields.is_empty() || query.include_fields.contains(&field.field_id)
        })
        .map(|field| {
            field
                .shape
                .iter()
                .try_fold(8u64, |bytes, extent| bytes.checked_mul(*extent as u64))
                .unwrap_or(u64::MAX)
        })
        .fold(0u64, u64::saturating_add);
    if payload > RESULTS_INLINE_FIELD_BYTES {
        query.include_field_values = false;
    }
    Ok(query)
}

pub fn analysis_results_by_run_id_op(
    run_id: &str,
    query: AnalysisResultsQuery,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisResultsData>, OperationErrorEnvelope> {
    // A definition run id names every attempt of a study step; results are those of the latest
    // committed attempt, and the returned run id says which attempt that is.
    let resolved_run_id = storage::resolve_run_attempt(run_id).map_err(|err| {
        operation_error(
            ANALYSIS_RESULTS_OPERATION,
            ANALYSIS_RESULTS_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RESULTS.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to resolve the latest FEA run attempt: {err}"),
            BTreeMap::from([("run_id".to_string(), run_id.to_string())]),
        )
    })?;
    let run_id = resolved_run_id.as_str();
    // The query is answered exactly: requested field values are returned whatever their size.
    // Bounded default views are chosen by the caller (`bounded_default_results_query`).
    let field_inventory = if query.include_field_values {
        None
    } else {
        storage::load_field_inventory(run_id).map_err(|err| {
            operation_error(
                ANALYSIS_RESULTS_OPERATION,
                ANALYSIS_RESULTS_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RESULTS.ARTIFACT_STORE_FAILED",
                    error_type: OperationErrorType::Internal,
                    retryable: true,
                    severity: OperationErrorSeverity::Error,
                },
                format!("failed to load FEA field inventory: {err}"),
                BTreeMap::from([("run_id".to_string(), run_id.to_string())]),
            )
        })?
    };
    let run_result = if query.include_field_values {
        storage::load_run_result(run_id)
    } else {
        storage::load_run_metadata(run_id)
    }
    .map_err(|err| {
        operation_error(
            ANALYSIS_RESULTS_OPERATION,
            ANALYSIS_RESULTS_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RESULTS.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to load FEA run artifact: {err}"),
            BTreeMap::from([("run_id".to_string(), run_id.to_string())]),
        )
    })?;

    let Some(run_result) = run_result else {
        let frequency = storage::load_structural_frequency_result(run_id).map_err(|err| {
            operation_error(
                ANALYSIS_RESULTS_OPERATION,
                ANALYSIS_RESULTS_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RESULTS.ARTIFACT_STORE_FAILED",
                    error_type: OperationErrorType::Internal,
                    retryable: true,
                    severity: OperationErrorSeverity::Error,
                },
                format!("failed to load structural frequency artifact: {err}"),
                BTreeMap::from([("run_id".to_string(), run_id.to_string())]),
            )
        })?;
        if let Some(frequency) = frequency {
            return structural_frequency::results_op(&frequency, query, context);
        }
        let frequency = storage::load_acoustic_frequency_result(run_id).map_err(|err| {
            operation_error(
                ANALYSIS_RESULTS_OPERATION,
                ANALYSIS_RESULTS_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RESULTS.ARTIFACT_STORE_FAILED",
                    error_type: OperationErrorType::Internal,
                    retryable: true,
                    severity: OperationErrorSeverity::Error,
                },
                format!("failed to load acoustic frequency artifact: {err}"),
                BTreeMap::from([("run_id".to_string(), run_id.to_string())]),
            )
        })?;
        if let Some(frequency) = frequency {
            return acoustic_frequency::results_op(&frequency, query, context);
        }

        return Err(operation_error(
            ANALYSIS_RESULTS_OPERATION,
            ANALYSIS_RESULTS_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RESULTS.RUN_NOT_FOUND",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!("FEA run_id '{run_id}' was not found"),
            BTreeMap::from([("run_id".to_string(), run_id.to_string())]),
        ));
    };

    let mut envelope =
        analysis_results_op_with_field_inventory(&run_result, query, context, field_inventory)?;
    for descriptor in &mut envelope.data.field_descriptors {
        if descriptor.storage == AnalysisFieldStorage::HostF64 {
            descriptor.storage_ref.kind = "chunked_field".to_string();
        }
    }
    Ok(envelope)
}

fn frequency_field_descriptor(
    entry: &runmat_analysis_fea::results::field::RecoveredFieldArtifactEntry,
) -> Result<AnalysisFieldDescriptor, String> {
    AnalysisFieldDescriptor::from_recovered_entry(entry)
}

pub fn analysis_field_page_by_run_id_op(
    run_id: &str,
    selection: AnalysisFieldSelection,
    options: AnalysisFieldRequestOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisFieldPageResult>, OperationErrorEnvelope> {
    let field_id = selection.field_id.clone();
    let lookup = storage::load_field_page(run_id, &selection, options).map_err(|err| {
        operation_error(
            ANALYSIS_RESULTS_OPERATION,
            ANALYSIS_RESULTS_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RESULTS.FIELD_ARTIFACT_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to load FEA field artifact: {err}"),
            BTreeMap::from([
                ("run_id".to_string(), run_id.to_string()),
                ("field_id".to_string(), field_id.to_string()),
            ]),
        )
    })?;
    let page = match lookup {
        storage::AnalysisFieldPageLookup::Page(page) => *page,
        storage::AnalysisFieldPageLookup::RunNotFound => {
            return Err(operation_error(
                ANALYSIS_RESULTS_OPERATION,
                ANALYSIS_RESULTS_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RESULTS.RUN_NOT_FOUND",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                format!("FEA run_id '{run_id}' was not found"),
                BTreeMap::from([("run_id".to_string(), run_id.to_string())]),
            ));
        }
        storage::AnalysisFieldPageLookup::FieldNotFound => {
            return Err(operation_error(
                ANALYSIS_RESULTS_OPERATION,
                ANALYSIS_RESULTS_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RESULTS.FIELD_NOT_FOUND",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                format!("requested FEA field '{field_id}' was not produced by run"),
                BTreeMap::from([
                    ("run_id".to_string(), run_id.to_string()),
                    ("requested_field".to_string(), field_id.to_string()),
                ]),
            ));
        }
        storage::AnalysisFieldPageLookup::FieldSelectionAmbiguous(available_fields) => {
            return Err(operation_error(
                ANALYSIS_RESULTS_OPERATION,
                ANALYSIS_RESULTS_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RESULTS.FIELD_AMBIGUOUS",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                format!("requested FEA field '{field_id}' matches more than one produced field"),
                BTreeMap::from([
                    ("run_id".to_string(), run_id.to_string()),
                    ("requested_field".to_string(), field_id.clone()),
                    ("available_fields".to_string(), available_fields.join(",")),
                ]),
            ));
        }
        storage::AnalysisFieldPageLookup::SampleSelectionRequired(samples) => {
            return Err(operation_error(
                ANALYSIS_RESULTS_OPERATION,
                ANALYSIS_RESULTS_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RESULTS.SAMPLE_REQUIRED",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                format!("requested FEA field '{field_id}' has more than one published sample"),
                BTreeMap::from([
                    ("run_id".to_string(), run_id.to_string()),
                    ("requested_field".to_string(), field_id.clone()),
                    (
                        "available_samples".to_string(),
                        serde_json::to_string(&samples).unwrap_or_default(),
                    ),
                ]),
            ));
        }
        storage::AnalysisFieldPageLookup::SampleNotFound => {
            return Err(operation_error(
                ANALYSIS_RESULTS_OPERATION,
                ANALYSIS_RESULTS_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RESULTS.SAMPLE_NOT_FOUND",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                format!("requested sample was not produced for FEA field '{field_id}'"),
                BTreeMap::from([
                    ("run_id".to_string(), run_id.to_string()),
                    ("requested_field".to_string(), field_id),
                ]),
            ));
        }
    };
    Ok(OperationEnvelope::new(
        ANALYSIS_RESULTS_OPERATION,
        ANALYSIS_RESULTS_OP_VERSION,
        &context,
        page,
    ))
}
