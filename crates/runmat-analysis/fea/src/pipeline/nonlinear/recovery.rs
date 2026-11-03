//! Shared original numerical-result projection, with no equilibrium re-solve.
use runmat_analysis_core::{AnalysisFieldLayout, ObservationQuantity, FieldLocation};
use super::*;

pub(super) fn recover(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: &NonlinearSolveOptions,
    prepared: preparation::PreparedNonlinear<'_>,
    nonlinear: NonlinearSolveResult,
    first_output_index: usize,
) -> Result<FeaNonlinearRunResult, FeaRunError> {
    let load_factor_start = nonlinear.initial_load_factor;
    let event_attributions = nonlinear_event_attributions(
        model,
        prepared.execution_step,
        &nonlinear,
        &prepared.resource_plan,
    )?;
    let preparation::PreparedNonlinear {
        execution_step,
        summary,
        identity,
        ..
    } = prepared;
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
        .ok_or_else(|| FeaRunError::Solve("nonlinear solve returned no accepted state".into()))?;
    let projected_displacement =
        project_structural_nodal_snapshot(&summary.structural_dof_layout, &displacement)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let von_mises = nonlinear
        .von_mises_snapshots
        .last()
        .cloned()
        .ok_or_else(|| {
            FeaRunError::Solve("nonlinear solve returned no accepted stress recovery".into())
        })?;
    let element_count = summary.structural_solid_element_count;
    let stress = nonlinear.stress_snapshots.last().cloned().ok_or_else(|| {
        FeaRunError::Solve("nonlinear solve returned no accepted physical stress".into())
    })?;
    let strain = nonlinear
        .total_strain_snapshots
        .last()
        .cloned()
        .ok_or_else(|| {
            FeaRunError::Solve("nonlinear solve returned no accepted total strain".into())
        })?;
    if von_mises.len() != element_count
        || stress.len() != element_count * TENSOR_COMPONENT_COUNT
        || strain.len() != element_count * TENSOR_COMPONENT_COUNT
    {
        return Err(FeaRunError::Solve(
            "accepted solid stress inventory differs from mesh".into(),
        ));
    }
    let mesh = options.solver_mesh.as_ref().ok_or_else(|| {
        FeaRunError::Assembly(
            "canonical nonlinear solids require an authoritative solver mesh".into(),
        )
    })?;
    let internal_force = nonlinear.internal_force_snapshots.last().ok_or_else(|| {
        FeaRunError::Solve("nonlinear solve returned no accepted internal force".into())
    })?;
    let factor = nonlinear.load_factors.last().copied().ok_or_else(|| {
        FeaRunError::Solve("nonlinear solve returned no accepted load factor".into())
    })?;
    let reaction = internal_force
        .iter()
        .zip(&summary.physical_load)
        .map(|(internal, external)| internal - factor * external)
        .collect::<Vec<_>>();
    let projected_reaction =
        project_structural_nodal_snapshot(&summary.structural_dof_layout, &reaction)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let constraints =
        crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(model, Some(mesh))
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let constraint_result =
        crate::constraints::CanonicalStructuralConstraintResult::recover_from_solver_state(
            &constraints,
            mesh,
            &displacement,
            &reaction,
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;

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
        structural_static_convergence: None,
        structural_static_balance: None,
        structural_constraint_result: Some(Box::new(constraint_result)),
        structural_connector_result: None,
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields: vec![
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_DISPLACEMENT, AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node),
                nodal_vector_shape(&summary.structural_dof_layout),
                projected_displacement.translations,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_VON_MISES, AnalysisFieldLayout::scalar(ObservationQuantity::VonMisesStress, FieldLocation::Element),
                vec![element_count],
                von_mises,
            ),
            AnalysisField::host_f64(
                crate::contracts::FEA_FIELD_STRUCTURAL_STRESS, AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::Element),
                vec![element_count, TENSOR_COMPONENT_COUNT],
                stress,
            ),
            AnalysisField::host_f64(
                crate::contracts::FEA_FIELD_STRUCTURAL_STRAIN, AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Strain, FieldLocation::Element),
                vec![element_count, TENSOR_COMPONENT_COUNT],
                strain,
            ),
            AnalysisField::host_f64(
                crate::contracts::FEA_FIELD_STRUCTURAL_REACTION_FORCE, AnalysisFieldLayout::vector3(ObservationQuantity::ReactionForce, FieldLocation::Node),
                nodal_vector_shape(&summary.structural_dof_layout),
                projected_reaction.translations,
            ),
        ],
    };

    if nonlinear.load_factors.last().copied() == Some(prepared.load_factor_end) {
        observations::evaluate(model, mesh, execution_step, &constraints, &mut run)?;
    }

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
                fea_nonlinear_displacement_field_id(first_output_index + index), AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node).step(first_output_index + index),
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
                fea_nonlinear_rotation_field_id(first_output_index + index), AnalysisFieldLayout::vector3(ObservationQuantity::Rotation, FieldLocation::Node).step(first_output_index + index),
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
                fea_nonlinear_von_mises_field_id(first_output_index + index), AnalysisFieldLayout::scalar(ObservationQuantity::VonMisesStress, FieldLocation::Element).step(first_output_index + index),
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
                fea_nonlinear_plastic_strain_field_id(first_output_index + index), AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::PlasticStrain, FieldLocation::Element).step(first_output_index + index),
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
                fea_nonlinear_equivalent_plastic_strain_field_id(first_output_index + index), AnalysisFieldLayout::scalar(ObservationQuantity::EquivalentPlasticStrain, FieldLocation::Element).step(first_output_index + index),
                vec![element_count],
                values,
            )
        })
        .collect::<Vec<_>>();
    // One value per contact point of the prepared contact operator, in operator order.
    let contact_point_fields = |snapshots: &[Vec<f64>],
                                id: fn(usize) -> String,
                                quantity: runmat_analysis_core::ObservationQuantity| {
        snapshots
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, values)| {
                AnalysisField::host_f64(
                    id(first_output_index + index),
                    runmat_analysis_core::AnalysisFieldLayout::scalar(
                        quantity,
                        runmat_analysis_core::FieldLocation::Interface,
                    )
                    .step(first_output_index + index),
                    vec![values.len()],
                    values,
                )
            })
            .collect::<Vec<_>>()
    };
    let contact_pressure_snapshots = contact_point_fields(
        &nonlinear.contact_pressure_snapshots,
        crate::contracts::fea_nonlinear_contact_pressure_field_id,
        runmat_analysis_core::ObservationQuantity::ContactPressure,
    );
    let contact_gap_snapshots = contact_point_fields(
        &nonlinear.contact_gap_snapshots,
        crate::contracts::fea_nonlinear_contact_gap_field_id,
        runmat_analysis_core::ObservationQuantity::ContactGap,
    );
    let load_factor_snapshots = nonlinear
        .load_factors
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_nonlinear_load_factor_field_id(first_output_index + index), AnalysisFieldLayout::scalar(ObservationQuantity::LoadFactor, FieldLocation::Global).step(first_output_index + index),
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
                fea_nonlinear_residual_norm_field_id(first_output_index + index), AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global).step(first_output_index + index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
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
                "structural.nonlinear.initial_displacement", AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node),
                nodal_vector_shape(&summary.structural_dof_layout),
                initial.translations.clone(),
            ),
        );
        let mut history_rotation_snapshots = rotation_snapshots.clone();
        history_rotation_snapshots.insert(
            0,
            AnalysisField::host_f64(
                "structural.nonlinear.initial_rotation", AnalysisFieldLayout::vector3(ObservationQuantity::Rotation, FieldLocation::Node),
                nodal_vector_shape(&summary.structural_dof_layout),
                initial.rotations,
            ),
        );
        evaluate_field_histories(
            execution_step,
            HistoryTopology::Structural {
                model,
                mesh: options.solver_mesh.as_ref(),
                connector_plan: None,
            },
            &coordinates,
            Some(&event_attributions),
            &[
                HistoryFieldSeries {
                    quantity: runmat_analysis_core::ObservationQuantity::Displacement,
                    source_field_id: NONLINEAR_DISPLACEMENT_HISTORY_SOURCE,
                    location: runmat_analysis_core::FieldLocation::Node,
                    entity_domain: crate::results::field::RecoveredFieldEntityDomain::Nodes,
                    value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                    numeric_representation:
                        crate::results::field::RecoveredFieldNumericRepresentation::Real,
                    snapshots: &history_displacement_snapshots,
                },
                HistoryFieldSeries {
                    quantity: runmat_analysis_core::ObservationQuantity::Rotation,
                    source_field_id: NONLINEAR_ROTATION_HISTORY_SOURCE,
                    location: runmat_analysis_core::FieldLocation::Node,
                    entity_domain: crate::results::field::RecoveredFieldEntityDomain::Nodes,
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

    run.validate_solver_formulation()?;
    Ok(FeaNonlinearRunResult {
        run,
        accepted_state: nonlinear.accepted_state,
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
        thermo_mechanical_temperature_snapshots: Vec::new(),
        thermo_mechanical_thermal_strain_snapshots: Vec::new(),
        thermo_mechanical_thermal_stress_snapshots: Vec::new(),
        thermo_mechanical_displacement_snapshots: Vec::new(),
        thermo_mechanical_von_mises_snapshots: Vec::new(),
        thermo_mechanical_coupling_residual_snapshots: Vec::new(),
        electro_thermal_temperature_snapshots: Vec::new(),
        electro_thermal_thermal_residual_snapshots: Vec::new(),
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

/// Project an authenticated original raw block under shared physical/resource preparation.
/// No equilibrium solve is run. Integrity and Newton closure do not establish continuum accuracy.
pub fn recover_original_nonlinear_block(
    workload: &crate::execution::NonlinearSolidWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    block: &crate::execution::NonlinearSolidBlockArtifact,
    additional_retained_bytes: u64,
) -> Result<FeaNonlinearRunResult, FeaRunError> {
    workload
        .validate()
        .map_err(|e| FeaRunError::InvalidModel(e.to_string()))?;
    block
        .validate()
        .map_err(|e| FeaRunError::Solve(e.to_string()))?;
    if block.request.workload
        != workload
            .identity()
            .map_err(|e| FeaRunError::InvalidModel(e.to_string()))?
        || block.request.topology
            != runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
    {
        return Err(FeaRunError::InvalidModel(
            "original recovery source differs from workload/topology".into(),
        ));
    }
    require_nonlinear_terminal_field_outputs(&workload.model.sequence.steps[0])?;
    let mut options = workload.controls.clone();
    options.solver_mesh = Some(mesh.clone());
    let prepared = preparation::prepare_recovery(
        &workload.model,
        &options,
        block.result.load_factors.len(),
        additional_retained_bytes,
    )?;
    crate::solve::nonlinear::validate_original_terminal_result(
        &workload.model,
        &prepared.summary,
        &options,
        &block.result,
    )?;
    let first = usize::try_from(block.request.first_output_ordinal - 1)
        .map_err(|_| FeaRunError::Solve("original output ordinal overflow".into()))?;
    first
        .checked_add(block.result.load_factors.len())
        .ok_or_else(|| FeaRunError::Solve("original output ordinal overflow".into()))?;
    recover(
        &workload.model,
        ComputeBackend::Cpu,
        &options,
        prepared,
        block.result.clone(),
        first,
    )
}
