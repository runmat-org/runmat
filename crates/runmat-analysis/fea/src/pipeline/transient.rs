use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel, AnalysisProcedure};

use crate::assembly::AssemblySummary;
use crate::{
    algebra::apply_m,
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    contracts::{
        fea_transient_acceleration_field_id, fea_transient_angular_acceleration_field_id,
        fea_transient_angular_velocity_field_id, fea_transient_displacement_field_id,
        fea_transient_kinetic_energy_field_id, fea_transient_residual_norm_field_id,
        fea_transient_rotation_field_id, fea_transient_strain_energy_field_id,
        fea_transient_velocity_field_id, fea_transient_von_mises_field_id, ComputeBackend,
        FeaRunError, FeaRunResult, FeaTransientRunResult, FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        FEA_FIELD_STRUCTURAL_VON_MISES,
    },
    diagnostics::{
        builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
        FeaDiagnostic, FeaDiagnosticSeverity,
    },
    excitation::{realize_step_excitations, time_discontinuities, ExcitationSamplePoint},
    pipeline::electro_thermal::recover_realized_electro_thermal_fields,
    pipeline::observation_history::{
        evaluate_field_histories, required_time_history_samples, HistoryFieldSeries,
        HistoryTopology,
    },
    pipeline::structural_nodal::{
        nodal_vector_shape, project_structural_nodal_snapshot, project_structural_nodal_snapshots,
    },
    pipeline::thermo_mechanical::recover_realized_thermo_mechanical_snapshots,
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::transient::{
        canonical_transient_step_boundaries, solve_transient_system_with_assembly,
        TransientSolveOptions,
    },
};

const VECTOR_COMPONENT_COUNT: usize = 3;
const TRANSIENT_DISPLACEMENT_HISTORY_SOURCE: &str = "structural.transient.displacement";
const TRANSIENT_VELOCITY_HISTORY_SOURCE: &str = "structural.transient.velocity";
const TRANSIENT_ACCELERATION_HISTORY_SOURCE: &str = "structural.transient.acceleration";
const TRANSIENT_ROTATION_HISTORY_SOURCE: &str = "structural.transient.rotation";

pub fn run_transient(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaTransientRunResult, FeaRunError> {
    run_transient_with_options(model, backend, TransientSolveOptions::default())
}

pub fn run_transient_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: TransientSolveOptions,
) -> Result<FeaTransientRunResult, FeaRunError> {
    let execution_step = super::admission::require_structural(
        model,
        AnalysisProcedure::TransientResponse,
        options.formulation_requirement,
    )?;
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating transient FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    let required_history_samples = required_time_history_samples(execution_step)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "transient model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;
    let time_grid = crate::excitation::ProcedureTimeGrid::from_interval_count(
        execution_step,
        options.interval_count,
    )
    .map_err(FeaRunError::InvalidModel)?;
    let start_time_s = time_grid.start_s();
    let active_model = super::activation::model_with_active_constraints(model, execution_step);
    let excitations = realize_step_excitations(
        &active_model,
        execution_step,
        ExcitationSamplePoint::time(start_time_s),
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling transient system",
        Some(1),
        Some(5),
    );
    let summary = assemble_structural_system(
        &active_model,
        &excitations,
        options.solver_mesh.clone(),
        StructuralFormulationRequirement::ExperimentalSyntheticOperator,
    )
    .map_err(|err| FeaRunError::Assembly(err.to_string()))?;
    let identity = super::admission::admit_structural(
        &summary,
        execution_step,
        options.formulation_requirement,
    )?;
    super::validate_rotational_dof_targets(model, &summary)?;
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "transient system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;

    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "solving transient steps",
        Some(2),
        Some(5),
    );
    let end_time_s = time_grid.end_s();
    let discontinuities =
        time_discontinuities(&active_model, execution_step, start_time_s, end_time_s)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let step_boundaries = canonical_transient_step_boundaries(
        &discontinuities,
        &required_history_samples,
        start_time_s,
        end_time_s,
    )
    .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let transient = solve_transient_system_with_assembly(
        &summary,
        options.clone(),
        backend,
        start_time_s,
        end_time_s,
        &step_boundaries,
        |time_s, approach| {
            let excitations = realize_step_excitations(
                &active_model,
                execution_step,
                ExcitationSamplePoint::TimeSeconds {
                    value: time_s,
                    approach,
                },
            )
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
            let assembly = assemble_structural_system(
                &active_model,
                &excitations,
                options.solver_mesh.clone(),
                StructuralFormulationRequirement::ExperimentalSyntheticOperator,
            )
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
            if assembly.dof_count != summary.dof_count {
                return Err(FeaRunError::Assembly(format!(
                    "transient reassembly changed structural DOF count from {} to {}",
                    summary.dof_count, assembly.dof_count
                )));
            }
            Ok::<AssemblySummary, FeaRunError>(assembly)
        },
    )?;
    let reporting_summary = transient
        .assembly_snapshots
        .last()
        .cloned()
        .unwrap_or_else(|| summary.clone());
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "transient solve complete",
        Some(3),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;

    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Started,
        "recovering transient fields",
        Some(3),
        Some(5),
    );
    let mut diagnostics = transient.diagnostics.clone();
    extend_common_run_diagnostics(
        &mut diagnostics,
        CommonRunDiagnosticInputs {
            model,
            summary: &reporting_summary,
        },
    );

    let displacement = transient
        .displacement_snapshots
        .last()
        .cloned()
        .unwrap_or_else(|| vec![0.0; reporting_summary.dof_count.max(3)]);
    let projected_displacement =
        project_structural_nodal_snapshot(&reporting_summary.structural_dof_layout, &displacement)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let von_mises = displacement
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        * 1.0e11;

    let mut run = FeaRunResult {
        identity,
        requested_backend: backend,
        execution_backend: crate::contracts::ExecutionBackend::from_solver_backend(
            transient.solver_backend,
        ),
        solver_backend: transient.solver_backend,
        solver_device_apply_k_ratio: if transient.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            transient.device_apply_k_count as f64 / transient.device_apply_k_attempt_count as f64
        },
        solver_method: transient.solver_method,
        preconditioner: transient.preconditioner,
        solver_host_sync_count: transient.solver_host_sync_count,
        accelerator_evidence: transient.accelerator_evidence.clone(),
        diagnostics,
        spatial_projection_evidence: transient
            .assembly_snapshots
            .iter()
            .flat_map(|summary| summary.spatial_projection_evidence.iter().cloned())
            .collect(),
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields: vec![
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_DISPLACEMENT,
                nodal_vector_shape(&reporting_summary.structural_dof_layout),
                projected_displacement.translations,
            ),
            AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_VON_MISES, vec![1], vec![von_mises]),
        ],
    };

    let velocity_values = recover_velocity_snapshots(
        &transient.displacement_snapshots,
        &transient.time_points_s,
        reporting_summary.dof_count,
    );
    let acceleration_values = recover_acceleration_snapshots(
        &velocity_values,
        &transient.time_points_s,
        reporting_summary.dof_count,
    );
    let displacement_values = project_structural_nodal_snapshots(
        &reporting_summary.structural_dof_layout,
        &transient.displacement_snapshots,
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let projected_velocity_values = project_structural_nodal_snapshots(
        &reporting_summary.structural_dof_layout,
        &velocity_values,
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let projected_acceleration_values = project_structural_nodal_snapshots(
        &reporting_summary.structural_dof_layout,
        &acceleration_values,
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let von_mises_values = recover_von_mises_snapshots(
        &transient.displacement_snapshots,
        reporting_summary.dof_count,
    );
    let kinetic_energy_values =
        recover_kinetic_energy_snapshots(&transient.assembly_snapshots, &velocity_values);
    let strain_energy_values = transient.strain_energy_snapshots.clone();
    run.diagnostics.push(transient_energy_balance_diagnostic(
        &kinetic_energy_values,
        &strain_energy_values,
    ));
    let residual_norm_values = recover_residual_norm_snapshots(
        transient.displacement_snapshots.len(),
        &transient.residual_norms,
    );

    let displacement_snapshots = displacement_values
        .iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_transient_displacement_field_id(index),
                nodal_vector_shape(&reporting_summary.structural_dof_layout),
                snapshot.translations.clone(),
            )
        })
        .collect::<Vec<_>>();
    let rotation_snapshots = displacement_values
        .iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_transient_rotation_field_id(index),
                nodal_vector_shape(&reporting_summary.structural_dof_layout),
                snapshot.rotations.clone(),
            )
        })
        .collect::<Vec<_>>();
    let velocity_snapshots = projected_velocity_values
        .iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_transient_velocity_field_id(index),
                nodal_vector_shape(&reporting_summary.structural_dof_layout),
                snapshot.translations.clone(),
            )
        })
        .collect::<Vec<_>>();
    let angular_velocity_snapshots = projected_velocity_values
        .iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_transient_angular_velocity_field_id(index),
                nodal_vector_shape(&reporting_summary.structural_dof_layout),
                snapshot.rotations.clone(),
            )
        })
        .collect::<Vec<_>>();
    let acceleration_snapshots = projected_acceleration_values
        .iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_transient_acceleration_field_id(index),
                nodal_vector_shape(&reporting_summary.structural_dof_layout),
                snapshot.translations.clone(),
            )
        })
        .collect::<Vec<_>>();
    let angular_acceleration_snapshots = projected_acceleration_values
        .iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_transient_angular_acceleration_field_id(index),
                nodal_vector_shape(&reporting_summary.structural_dof_layout),
                snapshot.rotations.clone(),
            )
        })
        .collect::<Vec<_>>();
    let von_mises_snapshots = von_mises_values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_transient_von_mises_field_id(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
    let kinetic_energy_snapshots = kinetic_energy_values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_transient_kinetic_energy_field_id(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
    let strain_energy_snapshots = strain_energy_values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_transient_strain_energy_field_id(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
    let residual_norm_snapshots = residual_norm_values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_transient_residual_norm_field_id(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
    let thermo_mechanical_summaries = transient
        .assembly_snapshots
        .iter()
        .map(|summary| summary.thermo_mechanical.clone())
        .collect::<Vec<_>>();
    let electro_thermal_summaries = transient
        .assembly_snapshots
        .iter()
        .map(|summary| summary.electro_thermal.clone())
        .collect::<Vec<_>>();
    let thermo_mechanical_fields = recover_realized_thermo_mechanical_snapshots(
        &thermo_mechanical_summaries,
        &displacement_snapshots,
        &von_mises_snapshots,
        &residual_norm_values,
        element_count_for_dofs(reporting_summary.dof_count),
    );
    let electro_thermal_fields = recover_realized_electro_thermal_fields(
        &electro_thermal_summaries,
        &residual_norm_values,
        reporting_summary.dof_count,
        options.solver_mesh.as_ref(),
    );
    run.diagnostics.extend(thermo_mechanical_fields.diagnostics);
    run.diagnostics.extend(electro_thermal_fields.diagnostics);
    run.fields.extend(electro_thermal_fields.static_fields);
    let evaluated_histories = if execution_step.observations.iter().any(|request| {
        matches!(
            request,
            runmat_analysis_core::ObservationRequest::History(_)
        )
    }) {
        let coordinates = transient
            .time_points_s
            .iter()
            .copied()
            .map(crate::results::field::RecoveredFieldSample::TimeSeconds)
            .collect::<Vec<_>>();
        evaluate_field_histories(
            execution_step,
            HistoryTopology::Structural {
                model: &active_model,
                mesh: options.solver_mesh.as_ref(),
            },
            &coordinates,
            &[
                HistoryFieldSeries {
                    quantity: runmat_analysis_core::ObservationQuantity::Displacement,
                    source_field_id: TRANSIENT_DISPLACEMENT_HISTORY_SOURCE,
                    location: runmat_analysis_core::FieldLocation::Node,
                    entity_domain:
                        crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                    value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                    numeric_representation:
                        crate::results::field::RecoveredFieldNumericRepresentation::Real,
                    snapshots: &displacement_snapshots,
                },
                HistoryFieldSeries {
                    quantity: runmat_analysis_core::ObservationQuantity::Velocity,
                    source_field_id: TRANSIENT_VELOCITY_HISTORY_SOURCE,
                    location: runmat_analysis_core::FieldLocation::Node,
                    entity_domain:
                        crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                    value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                    numeric_representation:
                        crate::results::field::RecoveredFieldNumericRepresentation::Real,
                    snapshots: &velocity_snapshots,
                },
                HistoryFieldSeries {
                    quantity: runmat_analysis_core::ObservationQuantity::Acceleration,
                    source_field_id: TRANSIENT_ACCELERATION_HISTORY_SOURCE,
                    location: runmat_analysis_core::FieldLocation::Node,
                    entity_domain:
                        crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                    value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                    numeric_representation:
                        crate::results::field::RecoveredFieldNumericRepresentation::Real,
                    snapshots: &acceleration_snapshots,
                },
                HistoryFieldSeries {
                    quantity: runmat_analysis_core::ObservationQuantity::Rotation,
                    source_field_id: TRANSIENT_ROTATION_HISTORY_SOURCE,
                    location: runmat_analysis_core::FieldLocation::Node,
                    entity_domain:
                        crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                    value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                    numeric_representation:
                        crate::results::field::RecoveredFieldNumericRepresentation::Real,
                    snapshots: &rotation_snapshots,
                },
            ],
        )
        .map_err(|error| FeaRunError::Observation(error.to_string()))?
    } else {
        Default::default()
    };
    run.observation_histories = evaluated_histories.histories;
    run.series_reductions = evaluated_histories.series_reductions;

    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Completed,
        "transient result field recovery complete",
        Some(4),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA transient run complete",
        Some(5),
        Some(5),
    );

    Ok(FeaTransientRunResult {
        run,
        time_points_s: transient.time_points_s,
        displacement_snapshots,
        rotation_snapshots,
        velocity_snapshots,
        angular_velocity_snapshots,
        acceleration_snapshots,
        angular_acceleration_snapshots,
        von_mises_snapshots,
        kinetic_energy_snapshots,
        strain_energy_snapshots,
        residual_norm_snapshots,
        thermo_mechanical_temperature_snapshots: thermo_mechanical_fields.temperature_snapshots,
        thermo_mechanical_thermal_strain_snapshots: thermo_mechanical_fields
            .thermal_strain_snapshots,
        thermo_mechanical_thermal_stress_snapshots: thermo_mechanical_fields
            .thermal_stress_snapshots,
        thermo_mechanical_displacement_snapshots: thermo_mechanical_fields.displacement_snapshots,
        thermo_mechanical_von_mises_snapshots: thermo_mechanical_fields.von_mises_snapshots,
        thermo_mechanical_coupling_residual_snapshots: thermo_mechanical_fields
            .coupling_residual_snapshots,
        electro_thermal_temperature_snapshots: electro_thermal_fields.temperature_snapshots,
        electro_thermal_thermal_residual_snapshots: electro_thermal_fields
            .thermal_residual_snapshots,
        residual_norms: transient.residual_norms,
    })
}

fn recover_velocity_snapshots(
    displacement_snapshots: &[Vec<f64>],
    time_points_s: &[f64],
    dof_count: usize,
) -> Vec<Vec<f64>> {
    let mut velocities = Vec::with_capacity(displacement_snapshots.len());
    for (index, snapshot) in displacement_snapshots.iter().enumerate() {
        if index == 0 {
            velocities.push(vec![0.0; dof_count.max(snapshot.len())]);
            continue;
        }
        let previous = &displacement_snapshots[index - 1];
        let dt = (time_points_s.get(index).copied().unwrap_or(index as f64)
            - time_points_s
                .get(index - 1)
                .copied()
                .unwrap_or(index.saturating_sub(1) as f64))
        .abs()
        .max(1.0e-12);
        velocities.push(difference_quotient(snapshot, previous, dt, dof_count));
    }
    velocities
}

fn recover_acceleration_snapshots(
    velocity_snapshots: &[Vec<f64>],
    time_points_s: &[f64],
    dof_count: usize,
) -> Vec<Vec<f64>> {
    let mut accelerations = Vec::with_capacity(velocity_snapshots.len());
    for (index, snapshot) in velocity_snapshots.iter().enumerate() {
        if index == 0 {
            accelerations.push(vec![0.0; dof_count.max(snapshot.len())]);
            continue;
        }
        let previous = &velocity_snapshots[index - 1];
        let dt = (time_points_s.get(index).copied().unwrap_or(index as f64)
            - time_points_s
                .get(index - 1)
                .copied()
                .unwrap_or(index.saturating_sub(1) as f64))
        .abs()
        .max(1.0e-12);
        accelerations.push(difference_quotient(snapshot, previous, dt, dof_count));
    }
    accelerations
}

fn difference_quotient(current: &[f64], previous: &[f64], dt: f64, dof_count: usize) -> Vec<f64> {
    let count = dof_count.max(current.len()).max(previous.len());
    (0..count)
        .map(|index| {
            (current.get(index).copied().unwrap_or(0.0)
                - previous.get(index).copied().unwrap_or(0.0))
                / dt
        })
        .collect()
}

fn recover_von_mises_snapshots(displacement_snapshots: &[Vec<f64>], dof_count: usize) -> Vec<f64> {
    displacement_snapshots
        .iter()
        .map(|snapshot| {
            snapshot
                .iter()
                .take(dof_count.max(snapshot.len()))
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max)
                * 1.0e11
        })
        .collect()
}

fn recover_kinetic_energy_snapshots(
    summaries: &[crate::assembly::AssemblySummary],
    velocity_snapshots: &[Vec<f64>],
) -> Vec<f64> {
    velocity_snapshots
        .iter()
        .enumerate()
        .map(|(index, velocity)| {
            let Some(summary) = summaries.get(index).or_else(|| summaries.last()) else {
                return 0.0;
            };
            let momentum = apply_m(&summary.operator, velocity);
            0.5 * velocity
                .iter()
                .zip(momentum.iter())
                .map(|(v, p)| v * p)
                .sum::<f64>()
        })
        .collect()
}

fn recover_residual_norm_snapshots(snapshot_count: usize, residual_norms: &[f64]) -> Vec<f64> {
    (0..snapshot_count)
        .map(|index| {
            if index == 0 {
                0.0
            } else {
                residual_norms
                    .get(index - 1)
                    .copied()
                    .unwrap_or(f64::INFINITY)
            }
        })
        .collect()
}

fn transient_energy_balance_diagnostic(
    kinetic_energy_values: &[f64],
    strain_energy_values: &[f64],
) -> FeaDiagnostic {
    let snapshot_count = kinetic_energy_values.len().max(strain_energy_values.len());
    let total_energy = (0..snapshot_count)
        .map(|index| {
            kinetic_energy_values.get(index).copied().unwrap_or(0.0)
                + strain_energy_values.get(index).copied().unwrap_or(0.0)
        })
        .collect::<Vec<_>>();
    let initial_total_energy = total_energy.first().copied().unwrap_or(0.0);
    let final_total_energy = total_energy.last().copied().unwrap_or(0.0);
    let max_total_energy = total_energy.iter().copied().fold(0.0_f64, f64::max);
    let reference_energy = total_energy
        .iter()
        .copied()
        .find(|value| value.is_finite() && *value > 1.0e-12)
        .unwrap_or(max_total_energy.max(1.0));
    let energy_growth_ratio = if reference_energy > 0.0 {
        max_total_energy / reference_energy
    } else {
        0.0
    };
    let max_step_energy_jump_ratio = total_energy
        .windows(2)
        .map(|window| {
            let previous = window[0].abs().max(reference_energy).max(1.0e-12);
            (window[1] - window[0]).abs() / previous
        })
        .fold(0.0_f64, f64::max);
    let metrics_are_finite = [
        initial_total_energy,
        final_total_energy,
        max_total_energy,
        energy_growth_ratio,
        max_step_energy_jump_ratio,
    ]
    .iter()
    .all(|value| value.is_finite());
    FeaDiagnostic {
        code: "FEA_TRANSIENT_ENERGY_BALANCE".to_string(),
        severity: if metrics_are_finite && energy_growth_ratio <= 10.0 {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "initial_total_energy={} final_total_energy={} max_total_energy={} energy_growth_ratio={} max_step_energy_jump_ratio={}",
            initial_total_energy,
            final_total_energy,
            max_total_energy,
            energy_growth_ratio,
            max_step_energy_jump_ratio,
        ),
    }
}

fn element_count_for_dofs(dof_count: usize) -> usize {
    dof_count.div_ceil(VECTOR_COMPONENT_COUNT).max(1)
}
