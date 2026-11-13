use super::super::*;
use super::run_solve_ms;
use statistics::{diagnostic_warning_rate, mean, percentile};

mod statistics;

pub fn analysis_trends_op(
    query: AnalysisTrendsQuery,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisTrendsData>, OperationErrorEnvelope> {
    let runs = storage::list_run_metadata().map_err(|err| {
        operation_error(
            ANALYSIS_TRENDS_OPERATION,
            ANALYSIS_TRENDS_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.TRENDS.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to list FEA run artifacts: {err}"),
            BTreeMap::new(),
        )
    })?;

    let mut grouped: HashMap<AnalysisRunIdentity, Vec<AnalysisRunResult>> = HashMap::new();
    for run in runs {
        grouped
            .entry(run.run.identity.clone())
            .or_default()
            .push(run);
    }

    let window = query.window_size.max(1);
    let mut summaries = Vec::new();
    let mut grouped = grouped.into_iter().collect::<Vec<_>>();
    grouped.sort_by_key(|(identity, _)| format!("{identity:?}"));
    for (identity, mut entries) in grouped {
        let is_electromagnetic = identity.family() == PhysicsFamily::Electromagnetic;
        entries.sort_by(|a, b| b.run_id.cmp(&a.run_id));
        if entries.len() > window {
            entries.truncate(window);
        }
        let sample_count = entries.len();
        if sample_count == 0 {
            continue;
        }

        let mut solve_samples = entries
            .iter()
            .filter_map(run_solve_ms)
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        solve_samples.sort_by(|a, b| a.total_cmp(b));
        let median_solve_ms = percentile(&solve_samples, 0.5);
        let p95_solve_ms = percentile(&solve_samples, 0.95);
        let numerically_acceptable_rate = entries
            .iter()
            .filter(|run| run.numerically_acceptable)
            .count() as f64
            / sample_count as f64;

        let failed_increment_rate = if identity.procedure() == AnalysisProcedure::Nonlinear {
            let failed = entries
                .iter()
                .filter_map(|run| run.nonlinear_results.as_ref())
                .filter(|nonlinear| nonlinear.failed_increments > 0)
                .count();
            Some(failed as f64 / sample_count as f64)
        } else {
            None
        };
        let mean_spike_count = if identity.procedure() == AnalysisProcedure::Nonlinear {
            let values = entries
                .iter()
                .filter_map(|run| run.nonlinear_results.as_ref())
                .map(|nonlinear| nonlinear.iteration_spike_count as f64)
                .collect::<Vec<_>>();
            Some(mean(&values))
        } else {
            None
        };
        let mean_stall_count = if identity.procedure() == AnalysisProcedure::Nonlinear {
            let values = entries
                .iter()
                .filter_map(|run| run.nonlinear_results.as_ref())
                .map(|nonlinear| nonlinear.convergence_stall_count as f64)
                .collect::<Vec<_>>();
            Some(mean(&values))
        } else {
            None
        };
        let thermo_coupling_enabled_rate = {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric_bool(&run.run.diagnostics, "FEA_TM_COUPLING", "enabled")
                })
                .collect::<Vec<_>>();
            if values.is_empty() {
                None
            } else {
                Some(values.iter().filter(|value| **value).count() as f64 / values.len() as f64)
            }
        };
        let thermo_nonlinear_warn_rate = if identity.procedure() == AnalysisProcedure::Nonlinear {
            diagnostic_warning_rate(&entries, "FEA_TM_NONLINEAR")
        } else {
            None
        };
        let thermo_spread_breach_rate = {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_TM_COUPLING",
                        "constitutive_material_spread_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, THERMO_SPREAD_THRESHOLD_BALANCED)
        };
        let thermo_heterogeneity_breach_rate = {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_TM_COUPLING",
                        "assignment_heterogeneity_index",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, THERMO_HETEROGENEITY_THRESHOLD_BALANCED)
        };
        let electro_thermal_coupling_enabled_rate = {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric_bool(&run.run.diagnostics, "FEA_ET_COUPLING", "enabled")
                })
                .collect::<Vec<_>>();
            if values.is_empty() {
                None
            } else {
                Some(values.iter().filter(|value| **value).count() as f64 / values.len() as f64)
            }
        };
        let electro_transient_warn_rate =
            if identity.procedure() == AnalysisProcedure::TransientResponse {
                diagnostic_warning_rate(&entries, "FEA_ET_TRANSIENT")
            } else {
                None
            };
        let electro_nonlinear_warn_rate = if identity.procedure() == AnalysisProcedure::Nonlinear {
            diagnostic_warning_rate(&entries, "FEA_ET_NONLINEAR")
        } else {
            None
        };
        let plastic_nonlinear_warn_rate = if identity.procedure() == AnalysisProcedure::Nonlinear {
            diagnostic_warning_rate(&entries, "FEA_PLASTIC_NONLINEAR")
        } else {
            None
        };
        let contact_nonlinear_warn_rate = if identity.procedure() == AnalysisProcedure::Nonlinear {
            diagnostic_warning_rate(&entries, "FEA_CONTACT_NONLINEAR")
        } else {
            None
        };
        let thermal_stability_warn_rate = if identity.family() == PhysicsFamily::Thermal {
            diagnostic_warning_rate(&entries, "FEA_THERMAL_STABILITY")
        } else {
            None
        };
        let thermal_constitutive_warn_rate = if identity.family() == PhysicsFamily::Thermal {
            diagnostic_warning_rate(&entries, "FEA_THERMAL_CONSTITUTIVE")
        } else {
            None
        };
        let thermal_spread_breach_rate = if identity.family() == PhysicsFamily::Thermal {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_THERMAL_CONSTITUTIVE",
                        "conductivity_spread_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, 2.5)
        } else {
            None
        };
        let electromagnetic_solve_warn_rate = if is_electromagnetic {
            Some(diagnostic_warning_rate(&entries, "FEA_EM_STATIC").unwrap_or(0.0))
        } else {
            None
        };
        let electromagnetic_spread_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_STATIC",
                        "conductivity_spread_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_CONDUCTIVITY_SPREAD_THRESHOLD_BALANCED)
        } else {
            None
        };
        let electromagnetic_heterogeneity_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_STATIC",
                        "electromagnetic_material_heterogeneity_index",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_HETEROGENEITY_THRESHOLD_BALANCED)
        } else {
            None
        };
        let electromagnetic_coverage_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_STATIC",
                        "assignment_coverage_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_less_than(&values, EM_ASSIGNMENT_COVERAGE_MIN_BALANCED)
        } else {
            None
        };
        let electromagnetic_contrast_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_STATIC",
                        "region_coefficient_contrast_index",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_REGION_CONTRAST_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_conditioning_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_STATIC",
                        "condition_number_estimate",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_CONDITIONING_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_source_realization_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_SOURCE_ENERGY",
                        "source_realization_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_less_than(&values, EM_SOURCE_REALIZATION_MIN_BALANCED)
        } else {
            None
        };
        let electromagnetic_source_region_coverage_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_SOURCE_ENERGY",
                        "source_region_coverage_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_less_than(&values, EM_SOURCE_REGION_COVERAGE_MIN_BALANCED)
        } else {
            None
        };
        let electromagnetic_source_material_alignment_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_SOURCE_ENERGY",
                        "source_material_alignment_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_less_than(&values, EM_SOURCE_MATERIAL_ALIGNMENT_MIN_BALANCED)
        } else {
            None
        };
        let electromagnetic_source_overlap_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_SOURCE_ENERGY",
                        "source_overlap_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_SOURCE_OVERLAP_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_source_interference_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_SOURCE_ENERGY",
                        "source_interference_index",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_SOURCE_INTERFERENCE_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_boundary_anchor_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_SOURCE_ENERGY",
                        "boundary_anchor_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_less_than(&values, EM_BOUNDARY_ANCHOR_MIN_BALANCED)
        } else {
            None
        };
        let electromagnetic_boundary_localization_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_SOURCE_ENERGY",
                        "boundary_condition_localization_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_less_than(&values, EM_BOUNDARY_LOCALIZATION_MIN_BALANCED)
        } else {
            None
        };
        let electromagnetic_ground_effectiveness_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_SOURCE_ENERGY",
                        "ground_anchor_effectiveness_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_less_than(&values, EM_GROUND_EFFECTIVENESS_MIN_BALANCED)
        } else {
            None
        };
        let electromagnetic_insulation_leakage_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_SOURCE_ENERGY",
                        "insulation_leakage_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_INSULATION_LEAKAGE_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_divergence_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_STATIC",
                        "flux_divergence_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_FLUX_DIVERGENCE_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_energy_imbalance_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_SOURCE_ENERGY",
                        "energy_imbalance_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_ENERGY_IMBALANCE_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_boundary_energy_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_SOURCE_ENERGY",
                        "boundary_energy_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_less_than(&values, EM_BOUNDARY_ENERGY_MIN_BALANCED)
        } else {
            None
        };
        let electromagnetic_boundary_penalty_contribution_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_SOURCE_ENERGY",
                        "boundary_penalty_conditioning_contribution",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_BOUNDARY_PENALTY_CONTRIBUTION_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_source_region_energy_consistency_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_SOURCE_ENERGY",
                        "source_region_energy_consistency_ratio",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_less_than(&values, EM_SOURCE_REGION_ENERGY_CONSISTENCY_MIN_BALANCED)
        } else {
            None
        };
        let electromagnetic_real_residual_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(&run.run.diagnostics, "FEA_EM_STATIC", "real_residual_norm")
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_REAL_RESIDUAL_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_imag_residual_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(&run.run.diagnostics, "FEA_EM_STATIC", "imag_residual_norm")
                })
                .collect::<Vec<_>>();
            breach_rate_greater_than(&values, EM_IMAG_RESIDUAL_MAX_BALANCED)
        } else {
            None
        };
        let electromagnetic_sweep_coverage_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(&run.run.diagnostics, "FEA_EM_SWEEP", "sweep_count")
                })
                .collect::<Vec<_>>();
            breach_rate_less_than(&values, EM_SWEEP_COUNT_MIN_BALANCED)
        } else {
            None
        };
        let electromagnetic_resonance_sharpness_breach_rate = if is_electromagnetic {
            let values = entries
                .iter()
                .filter_map(|run| {
                    diagnostic_metric(
                        &run.run.diagnostics,
                        "FEA_EM_SWEEP",
                        "resonance_quality_factor",
                    )
                })
                .collect::<Vec<_>>();
            breach_rate_less_than(&values, EM_RESONANCE_Q_MIN_BALANCED)
        } else {
            None
        };

        summaries.push(AnalysisTrendKindSummary {
            identity,
            sample_count,
            median_solve_ms,
            p95_solve_ms,
            numerically_acceptable_rate,
            failed_increment_rate,
            mean_spike_count,
            mean_stall_count,
            thermo_coupling_enabled_rate,
            thermo_nonlinear_warn_rate,
            thermo_spread_breach_rate,
            thermo_heterogeneity_breach_rate,
            electro_thermal_coupling_enabled_rate,
            electro_transient_warn_rate,
            electro_nonlinear_warn_rate,
            plastic_nonlinear_warn_rate,
            contact_nonlinear_warn_rate,
            thermal_stability_warn_rate,
            thermal_constitutive_warn_rate,
            thermal_spread_breach_rate,
            electromagnetic_solve_warn_rate,
            electromagnetic_spread_breach_rate,
            electromagnetic_heterogeneity_breach_rate,
            electromagnetic_coverage_breach_rate,
            electromagnetic_contrast_breach_rate,
            electromagnetic_conditioning_breach_rate,
            electromagnetic_source_realization_breach_rate,
            electromagnetic_source_region_coverage_breach_rate,
            electromagnetic_source_material_alignment_breach_rate,
            electromagnetic_source_overlap_breach_rate,
            electromagnetic_source_interference_breach_rate,
            electromagnetic_boundary_anchor_breach_rate,
            electromagnetic_boundary_localization_breach_rate,
            electromagnetic_ground_effectiveness_breach_rate,
            electromagnetic_insulation_leakage_breach_rate,
            electromagnetic_divergence_breach_rate,
            electromagnetic_energy_imbalance_breach_rate,
            electromagnetic_boundary_energy_breach_rate,
            electromagnetic_boundary_penalty_contribution_breach_rate,
            electromagnetic_source_region_energy_consistency_breach_rate,
            electromagnetic_real_residual_breach_rate,
            electromagnetic_imag_residual_breach_rate,
            electromagnetic_sweep_coverage_breach_rate,
            electromagnetic_resonance_sharpness_breach_rate,
        });
    }

    Ok(OperationEnvelope::new(
        ANALYSIS_TRENDS_OPERATION,
        ANALYSIS_TRENDS_OP_VERSION,
        &context,
        AnalysisTrendsData {
            window_size: window,
            summaries,
        },
    ))
}
