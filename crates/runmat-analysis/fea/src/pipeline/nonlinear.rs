use runmat_analysis_core::{
    validate_model, AnalysisField, AnalysisModel, AnalysisProcedure, IndependentDomain,
    ObservationEvent, ObservationRequest, SamplingPolicy,
};

use crate::{
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    contracts::{
        fea_nonlinear_contact_gap_field_id, fea_nonlinear_contact_pressure_field_id,
        fea_nonlinear_displacement_field_id, fea_nonlinear_equivalent_plastic_strain_field_id,
        fea_nonlinear_load_factor_field_id, fea_nonlinear_plastic_strain_field_id,
        fea_nonlinear_residual_norm_field_id, fea_nonlinear_rotation_field_id,
        fea_nonlinear_von_mises_field_id, ComputeBackend, FeaContactInterfaceContext,
        FeaNonlinearRunResult, FeaRunError, FeaRunResult, FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        FEA_FIELD_STRUCTURAL_VON_MISES,
    },
    diagnostics::builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    execution::{SequenceIdentity, SolverResultIdentity, StepIdentity},
    pipeline::electro_thermal::recover_electro_thermal_fields,
    pipeline::observation_history::{
        evaluate_field_histories, required_load_factor_history_samples, HistoryFieldSeries,
        HistoryTopology,
    },
    pipeline::structural_nodal::{
        nodal_vector_shape, project_structural_nodal_snapshot, project_structural_nodal_snapshots,
    },
    pipeline::thermo_mechanical::recover_thermo_mechanical_snapshots,
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::{
        nonlinear::{
            detect_completed_nonlinear_events, nonlinear_solver_content_digest,
            solve_nonlinear_system_at_boundaries, NonlinearSolveOptions, NonlinearSolveResult,
        },
        transient::canonical_transient_step_boundaries,
    },
};

const VECTOR_COMPONENT_COUNT: usize = 3;
const TENSOR_COMPONENT_COUNT: usize = 6;
const NONLINEAR_DISPLACEMENT_HISTORY_SOURCE: &str = "structural.nonlinear.displacement";
const NONLINEAR_ROTATION_HISTORY_SOURCE: &str = "structural.nonlinear.rotation";

pub fn run_nonlinear(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaNonlinearRunResult, FeaRunError> {
    run_nonlinear_with_options(model, backend, NonlinearSolveOptions::default())
}

pub fn run_nonlinear_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: NonlinearSolveOptions,
) -> Result<FeaNonlinearRunResult, FeaRunError> {
    let execution_step = super::admission::require_structural(
        model,
        AnalysisProcedure::Nonlinear,
        options.formulation_requirement,
    )?;
    validate_nonlinear_event_requests(execution_step)?;
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating nonlinear FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    let required_history_samples = required_load_factor_history_samples(execution_step)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let IndependentDomain::LoadFactor {
        start: load_factor_start,
        end: load_factor_end,
    } = execution_step.independent_domain
    else {
        return Err(FeaRunError::InvalidModel(
            "nonlinear structural execution requires a load-factor domain".into(),
        ));
    };
    let step_boundaries = canonical_transient_step_boundaries(
        &[],
        &required_history_samples,
        load_factor_start,
        load_factor_end,
    )
    .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "nonlinear model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;
    let excitations =
        realize_step_excitations(model, execution_step, ExcitationSamplePoint::Static)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling nonlinear system",
        Some(1),
        Some(5),
    );
    let summary = assemble_structural_system(
        model,
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
        "fea.run_nonlinear",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "nonlinear system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;

    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "solving nonlinear increments",
        Some(2),
        Some(5),
    );
    let nonlinear = solve_nonlinear_system_at_boundaries(
        &summary,
        options.clone(),
        backend,
        load_factor_start,
        load_factor_end,
        &step_boundaries,
    );
    let event_attributions = nonlinear_event_attributions(model, execution_step, &nonlinear)?;
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "nonlinear solve complete",
        Some(3),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;

    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Started,
        "recovering nonlinear fields",
        Some(3),
        Some(5),
    );
    let mut diagnostics = nonlinear.diagnostics.clone();
    extend_common_run_diagnostics(
        &mut diagnostics,
        CommonRunDiagnosticInputs {
            model,
            summary: &summary,
        },
    );

    let displacement = nonlinear
        .displacement_snapshots
        .last()
        .cloned()
        .unwrap_or_else(|| vec![0.0; summary.dof_count.max(3)]);
    let projected_displacement =
        project_structural_nodal_snapshot(&summary.structural_dof_layout, &displacement)
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
            nonlinear.solver_backend,
        ),
        solver_backend: nonlinear.solver_backend,
        solver_device_apply_k_ratio: if nonlinear.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            nonlinear.device_apply_k_count as f64 / nonlinear.device_apply_k_attempt_count as f64
        },
        solver_method: nonlinear.solver_method,
        preconditioner: nonlinear.preconditioner,
        solver_host_sync_count: nonlinear.solver_host_sync_count,
        accelerator_evidence: nonlinear.accelerator_evidence.clone(),
        diagnostics,
        spatial_projection_evidence: summary.spatial_projection_evidence.clone(),
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields: vec![
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_DISPLACEMENT,
                nodal_vector_shape(&summary.structural_dof_layout),
                projected_displacement.translations,
            ),
            AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_VON_MISES, vec![1], vec![von_mises]),
        ],
    };

    let element_count = element_count_for_dofs(summary.dof_count);
    let projected_displacement_values = project_structural_nodal_snapshots(
        &summary.structural_dof_layout,
        &nonlinear.displacement_snapshots,
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;

    let displacement_snapshots = projected_displacement_values
        .iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_nonlinear_displacement_field_id(index),
                nodal_vector_shape(&summary.structural_dof_layout),
                snapshot.translations.clone(),
            )
        })
        .collect::<Vec<_>>();
    let rotation_snapshots = projected_displacement_values
        .iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_nonlinear_rotation_field_id(index),
                nodal_vector_shape(&summary.structural_dof_layout),
                snapshot.rotations.clone(),
            )
        })
        .collect::<Vec<_>>();
    let von_mises_snapshots = nonlinear
        .von_mises_snapshots
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, values)| {
            AnalysisField::host_f64(
                fea_nonlinear_von_mises_field_id(index),
                vec![element_count],
                values,
            )
        })
        .collect::<Vec<_>>();
    let plastic_strain_snapshots = nonlinear
        .plastic_strain_snapshots
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, values)| {
            AnalysisField::host_f64(
                fea_nonlinear_plastic_strain_field_id(index),
                vec![element_count, TENSOR_COMPONENT_COUNT],
                values,
            )
        })
        .collect::<Vec<_>>();
    let equivalent_plastic_strain_snapshots = nonlinear
        .equivalent_plastic_strain_snapshots
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, values)| {
            AnalysisField::host_f64(
                fea_nonlinear_equivalent_plastic_strain_field_id(index),
                vec![element_count],
                values,
            )
        })
        .collect::<Vec<_>>();
    let contact_count = contact_entity_count(options.contact_context.as_ref());
    let contact_pressure_snapshots = nonlinear
        .contact_pressure_snapshots
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, values)| {
            AnalysisField::host_f64(
                fea_nonlinear_contact_pressure_field_id(index),
                vec![contact_count],
                values,
            )
        })
        .collect::<Vec<_>>();
    let contact_gap_snapshots = nonlinear
        .contact_gap_snapshots
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, values)| {
            AnalysisField::host_f64(
                fea_nonlinear_contact_gap_field_id(index),
                vec![contact_count],
                values,
            )
        })
        .collect::<Vec<_>>();
    let load_factor_snapshots = nonlinear
        .load_factors
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_nonlinear_load_factor_field_id(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
    let residual_norm_snapshots = nonlinear
        .residual_norms
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_nonlinear_residual_norm_field_id(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
    let thermo_mechanical_fields = recover_thermo_mechanical_snapshots(
        summary.thermo_mechanical.as_ref(),
        &nonlinear.load_factors,
        &displacement_snapshots,
        &von_mises_snapshots,
        &nonlinear.residual_norms,
        element_count,
    );
    let electro_thermal_fields = recover_electro_thermal_fields(
        summary.electro_thermal.as_ref(),
        &nonlinear.load_factors,
        &nonlinear.residual_norms,
        summary.dof_count,
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
        let mut coordinates = nonlinear
            .load_factors
            .iter()
            .copied()
            .map(crate::results::field::RecoveredFieldSample::LoadFactor)
            .collect::<Vec<_>>();
        let mut history_displacement_snapshots = displacement_snapshots.clone();
        coordinates.insert(
            0,
            crate::results::field::RecoveredFieldSample::LoadFactor(load_factor_start),
        );
        let initial = project_structural_nodal_snapshot(
            &summary.structural_dof_layout,
            &nonlinear.initial_displacement,
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        history_displacement_snapshots.insert(
            0,
            AnalysisField::host_f64(
                "structural.nonlinear.initial_displacement",
                nodal_vector_shape(&summary.structural_dof_layout),
                initial.translations.clone(),
            ),
        );
        let mut history_rotation_snapshots = rotation_snapshots.clone();
        history_rotation_snapshots.insert(
            0,
            AnalysisField::host_f64(
                "structural.nonlinear.initial_rotation",
                nodal_vector_shape(&summary.structural_dof_layout),
                initial.rotations,
            ),
        );
        evaluate_field_histories(
            execution_step,
            HistoryTopology::Structural {
                model,
                mesh: options.solver_mesh.as_ref(),
            },
            &coordinates,
            Some(&event_attributions),
            &[
                HistoryFieldSeries {
                    quantity: runmat_analysis_core::ObservationQuantity::Displacement,
                    source_field_id: NONLINEAR_DISPLACEMENT_HISTORY_SOURCE,
                    location: runmat_analysis_core::FieldLocation::Node,
                    entity_domain:
                        crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                    value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                    numeric_representation:
                        crate::results::field::RecoveredFieldNumericRepresentation::Real,
                    snapshots: &history_displacement_snapshots,
                },
                HistoryFieldSeries {
                    quantity: runmat_analysis_core::ObservationQuantity::Rotation,
                    source_field_id: NONLINEAR_ROTATION_HISTORY_SOURCE,
                    location: runmat_analysis_core::FieldLocation::Node,
                    entity_domain:
                        crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                    value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                    numeric_representation:
                        crate::results::field::RecoveredFieldNumericRepresentation::Real,
                    snapshots: &history_rotation_snapshots,
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
        "fea.run_nonlinear",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Completed,
        "nonlinear result field recovery complete",
        Some(4),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA nonlinear run complete",
        Some(5),
        Some(5),
    );

    Ok(FeaNonlinearRunResult {
        run,
        load_factors: nonlinear.load_factors,
        displacement_snapshots,
        rotation_snapshots,
        von_mises_snapshots,
        plastic_strain_snapshots,
        equivalent_plastic_strain_snapshots,
        contact_pressure_snapshots,
        contact_gap_snapshots,
        load_factor_snapshots,
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
        residual_norms: nonlinear.residual_norms,
        increment_norms: nonlinear.increment_norms,
        iteration_counts: nonlinear.iteration_counts,
        failed_increments: nonlinear.failed_increments,
        line_search_backtracks: nonlinear.line_search_backtracks,
        max_line_search_backtracks_per_increment: nonlinear
            .max_line_search_backtracks_per_increment,
        tangent_rebuild_count: nonlinear.tangent_rebuild_count,
        iteration_spike_count: nonlinear.iteration_spike_count,
        convergence_stall_count: nonlinear.convergence_stall_count,
        backtrack_burst_count: nonlinear.backtrack_burst_count,
    })
}

fn validate_nonlinear_event_requests(
    step: &runmat_analysis_core::ProcedureStep,
) -> Result<(), FeaRunError> {
    for request in &step.observations {
        let ObservationRequest::History(spec) = request else {
            continue;
        };
        let SamplingPolicy::Event(event) = &spec.sampling else {
            continue;
        };
        if !matches!(*event, ObservationEvent::Solver(_)) {
            return Err(FeaRunError::Observation(
                "nonlinear execution currently owns solver events; derived-series and accuracy events require their source result or report"
                    .into(),
            ));
        }
    }
    Ok(())
}

fn nonlinear_event_attributions(
    model: &AnalysisModel,
    step: &runmat_analysis_core::ProcedureStep,
    result: &NonlinearSolveResult,
) -> Result<Vec<crate::results::observation::ObservationEventAttribution>, FeaRunError> {
    if !step.observations.iter().any(|request| {
        matches!(
            request,
            ObservationRequest::History(runmat_analysis_core::ObservationSpec {
                sampling: SamplingPolicy::Event(ObservationEvent::Solver(_)),
                ..
            })
        )
    }) {
        return Ok(Vec::new());
    }
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let producer = StepIdentity::from_solver_contract(sequence, step)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let content = nonlinear_solver_content_digest(result)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let source = SolverResultIdentity::from_step_result(producer, content)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    Ok(detect_completed_nonlinear_events(result)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?
        .into_iter()
        .map(|event| {
            crate::results::observation::ObservationEventAttribution::Solver(
                crate::results::observation::SolverEventAttribution {
                    event: event.event,
                    sample: crate::results::field::RecoveredFieldSample::LoadFactor(
                        event.load_factor,
                    ),
                    occurrence_index: event.occurrence_index,
                    source: crate::results::observation::SolverEventSource::CompletedResult {
                        result: source,
                    },
                },
            )
        })
        .collect())
}

fn element_count_for_dofs(dof_count: usize) -> usize {
    dof_count
        .div_ceil(VECTOR_COMPONENT_COUNT)
        .saturating_sub(1)
        .max(1)
}

fn contact_entity_count(contact: Option<&FeaContactInterfaceContext>) -> usize {
    if contact.is_some_and(|context| context.enabled) {
        1
    } else {
        0
    }
}
