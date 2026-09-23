use super::*;

pub(super) struct TransientRunFinalization<'a> {
    pub(super) model: &'a AnalysisModel,
    pub(super) execution_step: &'a ProcedureStep,
    pub(super) active_model: &'a AnalysisModel,
    pub(super) backend: ComputeBackend,
    pub(super) options: &'a TransientSolveOptions,
    pub(super) identity: crate::admission::AnalysisRunIdentity,
    pub(super) initial_state: StructuralInitialStateEvidence,
}

pub(super) fn finish_transient_run(
    input: TransientRunFinalization<'_>,
    transient: crate::solve::transient::TransientSolveResult,
) -> Result<FeaTransientRunResult, FeaRunError> {
    let TransientRunFinalization {
        model,
        execution_step,
        active_model,
        backend,
        options,
        identity,
        initial_state,
    } = input;
    let reporting_summary = transient
        .assembly_snapshots
        .last()
        .cloned()
        .ok_or_else(|| {
            FeaRunError::Assembly("transient solve produced no reporting assembly snapshot".into())
        })?;
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

    let recovered = recovery::recover_transient_fields(&transient)?;
    let kinetic_energy_values = transient.kinetic_energy_snapshots.clone();
    let strain_energy_values = transient.strain_energy_snapshots.clone();
    recovery::require_snapshot_count(
        transient.displacement_snapshots.len(),
        "residual norm",
        transient.residual_norms.len(),
    )?;
    let residual_norm_values = transient.residual_norms.clone();
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
        structural_static_balance: None,
        structural_constraint_result: None,
        structural_connector_result: None,
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields: recovered.final_deformation_fields.clone(),
    };
    run.diagnostics.push(mechanical_energy_history_diagnostic(
        &kinetic_energy_values,
        &strain_energy_values,
        &transient.external_work_j,
        &transient.support_work_j,
        &transient.damping_dissipation_j,
        &transient.energy_balance_residual_j,
        &transient.relative_energy_balance_error,
    ));

    let displacement_snapshots = recovered.displacement_snapshots;
    let rotation_snapshots = recovered.rotation_snapshots;
    let velocity_snapshots = recovered.velocity_snapshots;
    let angular_velocity_snapshots = recovered.angular_velocity_snapshots;
    let acceleration_snapshots = recovered.acceleration_snapshots;
    let angular_acceleration_snapshots = recovered.angular_acceleration_snapshots;
    let reaction_force_snapshots = recovered.reaction_force_snapshots;
    let reaction_moment_snapshots = recovered.reaction_moment_snapshots;
    let strain_snapshots = recovered.strain_snapshots;
    let stress_snapshots = recovered.stress_snapshots;
    let von_mises_snapshots = recovered.von_mises_snapshots;
    let strain_energy_density_snapshots = recovered.strain_energy_density_snapshots;
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
    let external_work_snapshots = scalar_snapshots(
        &transient.external_work_j,
        fea_transient_external_work_field_id,
    );
    let support_work_snapshots = scalar_snapshots(
        &transient.support_work_j,
        fea_transient_support_work_field_id,
    );
    let damping_dissipation_snapshots = scalar_snapshots(
        &transient.damping_dissipation_j,
        fea_transient_damping_dissipation_field_id,
    );
    let energy_balance_residual_snapshots = scalar_snapshots(
        &transient.energy_balance_residual_j,
        fea_transient_energy_balance_residual_field_id,
    );
    let relative_energy_balance_error_snapshots = scalar_snapshots(
        &transient.relative_energy_balance_error,
        fea_transient_relative_energy_balance_error_field_id,
    );
    if let Some(field) = reaction_force_snapshots.last() {
        let mut field = field.clone();
        field.field_id = FEA_FIELD_STRUCTURAL_REACTION_FORCE.into();
        run.fields.push(field);
    }
    if let Some(field) = reaction_moment_snapshots.last() {
        let mut field = field.clone();
        field.field_id = FEA_FIELD_STRUCTURAL_REACTION_MOMENT.into();
        run.fields.push(field);
    }
    let electro_thermal_summaries = transient
        .assembly_snapshots
        .iter()
        .map(|summary| summary.electro_thermal.clone())
        .collect::<Vec<_>>();
    let electro_thermal_fields = recover_realized_electro_thermal_fields(
        &electro_thermal_summaries,
        &residual_norm_values,
        reporting_summary.dof_count,
        options.solver_mesh.as_ref(),
    );
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
        let mut series = vec![
            HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Displacement,
                source_field_id: TRANSIENT_DISPLACEMENT_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Node,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &displacement_snapshots,
            },
            HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Velocity,
                source_field_id: TRANSIENT_VELOCITY_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Node,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &velocity_snapshots,
            },
            HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Acceleration,
                source_field_id: TRANSIENT_ACCELERATION_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Node,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &acceleration_snapshots,
            },
        ];
        if rotation_snapshots.len() == coordinates.len() {
            series.push(HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Rotation,
                source_field_id: TRANSIENT_ROTATION_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Node,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &rotation_snapshots,
            });
        }
        if strain_snapshots.len() == coordinates.len() {
            series.push(HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Strain,
                source_field_id: TRANSIENT_STRAIN_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Element,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::SolidElements,
                value_shape: runmat_analysis_core::ObservationValueShape::SymmetricTensor3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &strain_snapshots,
            });
        }
        if stress_snapshots.len() == coordinates.len() {
            series.push(HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Stress,
                source_field_id: TRANSIENT_STRESS_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Element,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::SolidElements,
                value_shape: runmat_analysis_core::ObservationValueShape::SymmetricTensor3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &stress_snapshots,
            });
        }
        if strain_energy_density_snapshots.len() == coordinates.len() {
            series.push(HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::StrainEnergy,
                source_field_id: TRANSIENT_STRAIN_ENERGY_DENSITY_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Element,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::SolidElements,
                value_shape: runmat_analysis_core::ObservationValueShape::Scalar,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &strain_energy_density_snapshots,
            });
        }
        evaluate_field_histories(
            execution_step,
            HistoryTopology::Structural {
                model: active_model,
                mesh: options.solver_mesh.as_ref(),
            },
            &coordinates,
            None,
            &series,
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

    run.validate_solver_formulation()?;
    Ok(FeaTransientRunResult {
        run,
        initial_state,
        time_points_s: transient.time_points_s,
        displacement_snapshots,
        rotation_snapshots,
        velocity_snapshots,
        angular_velocity_snapshots,
        acceleration_snapshots,
        angular_acceleration_snapshots,
        strain_snapshots,
        stress_snapshots,
        von_mises_snapshots,
        strain_energy_density_snapshots,
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
        dynamic_balance_snapshots: transient.dynamic_balance_snapshots,
        electro_thermal_temperature_snapshots: electro_thermal_fields.temperature_snapshots,
        electro_thermal_thermal_residual_snapshots: electro_thermal_fields
            .thermal_residual_snapshots,
        residual_norms: transient.residual_norms,
    })
}

fn scalar_snapshots(values: &[f64], field_id: impl Fn(usize) -> String) -> Vec<AnalysisField> {
    values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| AnalysisField::host_f64(field_id(index), vec![1], vec![value]))
        .collect()
}
