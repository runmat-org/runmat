use runmat_analysis_core::{AnalysisFieldLayout, ObservationQuantity, FieldLocation};
use super::*;

pub(super) struct TransientRunFinalization<'a> {
    pub(super) model: &'a AnalysisModel,
    pub(super) execution_step: &'a ProcedureStep,
    pub(super) active_model: &'a AnalysisModel,
    pub(super) backend: ComputeBackend,
    pub(super) options: &'a TransientSolveOptions,
    pub(super) initial_assembly: &'a AssemblySummary,
    pub(super) identity: crate::admission::AnalysisRunIdentity,
    pub(super) initial_state: StructuralInitialStateEvidence,
}

pub(super) fn finish_transient_run(
    input: TransientRunFinalization<'_>,
    mut transient: crate::solve::transient::TransientSolveResult,
    mut reassemble: Option<&mut dyn FnMut(f64) -> Result<AssemblySummary, FeaRunError>>,
) -> Result<FeaTransientRunResult, FeaRunError> {
    let TransientRunFinalization {
        model,
        execution_step,
        active_model,
        backend,
        options,
        initial_assembly,
        identity,
        initial_state,
    } = input;
    // Motion realization has already consumed the solver integration intervals.
    // Their duplicated endpoint assemblies and states are not part of the public
    // history, so release them before constructing recovered field histories.
    transient.integration_intervals.clear();
    let terminal_time = *transient.time_points_s.last().ok_or_else(|| {
        FeaRunError::Assembly("transient solve produced no reporting time point".into())
    })?;
    let mut terminal_assembly = reassemble
        .as_mut()
        .map(|assemble| assemble(terminal_time))
        .transpose()?;
    let reporting_summary = terminal_assembly
        .as_ref()
        .or_else(|| transient.assembly_snapshots.last())
        .ok_or_else(|| {
            FeaRunError::Assembly("transient solve produced no reporting assembly snapshot".into())
        })?;
    let reporting_dof_count = reporting_summary.dof_count;
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
    let mut diagnostics = std::mem::take(&mut transient.diagnostics);
    extend_common_run_diagnostics(
        &mut diagnostics,
        CommonRunDiagnosticInputs {
            model,
            summary: reporting_summary,
        },
    );

    let empty_connector_plan = if reporting_summary.structural_connector_plan.is_none()
        && active_model.connectors.is_empty()
    {
        let default_structural = runmat_analysis_core::StructuralModel::default();
        let structural = active_model
            .structural
            .as_ref()
            .unwrap_or(&default_structural);
        let topology = if let Some(mesh) = options.solver_mesh.as_ref() {
            if active_model.structural.is_some() {
                crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace::derive(
                    mesh,
                    structural,
                )
                .map_err(|error| FeaRunError::Assembly(error.to_string()))?
                .identity
            } else {
                crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace::derive_solid(
                    mesh,
                )
                .map_err(|error| FeaRunError::Assembly(error.to_string()))?
                .identity
            }
        } else {
            crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace::derive_explicit(
                structural,
            )
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?
            .identity
        };
        Some(
            crate::connectors::CanonicalStructuralConnectorPlan::empty(
                topology,
                reporting_summary.structural_node_count,
            )
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?,
        )
    } else {
        None
    };
    recovery::require_snapshot_count(
        transient.time_points_s.len(),
        "residual norm",
        transient.residual_norms.len(),
    )?;
    let recovered = if let Some(assemble) = reassemble {
        let terminal_index = transient.time_points_s.len() - 1;
        let times = transient.time_points_s.clone();
        let summaries = times.into_iter().enumerate().map(|(index, time)| {
            if index == terminal_index {
                Ok(terminal_assembly
                    .take()
                    .expect("terminal assembly was prepared"))
            } else {
                assemble(time)
            }
        });
        recovery::recover_transient_fields(
            &mut transient,
            empty_connector_plan.as_ref(),
            initial_assembly,
            summaries,
        )?
    } else {
        let summaries = std::mem::take(&mut transient.assembly_snapshots)
            .into_iter()
            .map(Ok);
        recovery::recover_transient_fields(
            &mut transient,
            empty_connector_plan.as_ref(),
            initial_assembly,
            summaries,
        )?
    };
    // Recovery consumed the solver-owned state vectors as it built the typed
    // histories; no duplicate raw history survives into observation work.
    for connector_result in &recovered.connector_results {
        connector_result
            .plan
            .validate_against_authored(&active_model.connectors)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    }
    let final_connector_result = recovered.connector_results.last().cloned().ok_or_else(|| {
        FeaRunError::Assembly(
            "transient solve produced no canonical connector reporting state".into(),
        )
    })?;
    let recovery::TransientFieldRecovery {
        final_deformation_fields,
        displacement_snapshots,
        rotation_snapshots,
        velocity_snapshots,
        angular_velocity_snapshots,
        acceleration_snapshots,
        angular_acceleration_snapshots,
        reaction_force_snapshots,
        reaction_moment_snapshots,
        connector_results,
        connector_force_snapshots,
        connector_stored_energy_snapshots,
        strain_snapshots,
        stress_snapshots,
        von_mises_snapshots,
        strain_energy_density_snapshots,
        spatial_projection_evidence,
        electro_thermal_summaries,
    } = recovered;
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
        accelerator_evidence: transient.accelerator_evidence,
        diagnostics,
        spatial_projection_evidence,
        structural_static_convergence: None,
        structural_static_balance: None,
        structural_constraint_result: None,
        structural_connector_result: Some(Box::new(final_connector_result)),
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields: final_deformation_fields,
    };
    run.diagnostics.push(mechanical_energy_history_diagnostic(
        &transient.kinetic_energy_snapshots,
        &transient.strain_energy_snapshots,
        transient.external_work_j.canonical_values(),
        transient.support_work_j.canonical_values(),
        transient.damping_dissipation_j.canonical_values(),
        transient.energy_balance_residual_j.canonical_values(),
        &transient.relative_energy_balance_error,
    ));

    let kinetic_energy_snapshots = transient
        .kinetic_energy_snapshots
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_transient_kinetic_energy_field_id(index), AnalysisFieldLayout::scalar(ObservationQuantity::KineticEnergy, FieldLocation::Global).step(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
    let strain_energy_snapshots = transient
        .strain_energy_snapshots
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_transient_strain_energy_field_id(index), AnalysisFieldLayout::scalar(ObservationQuantity::StrainEnergy, FieldLocation::Global).step(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
    let residual_norm_snapshots = transient
        .residual_norms
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_transient_residual_norm_field_id(index), AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global).step(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
    let external_work_snapshots = scalar_snapshots(
        transient.external_work_j.canonical_values(),
        fea_transient_external_work_field_id,
        runmat_analysis_core::ObservationQuantity::ExternalWork,
    );
    let support_work_snapshots = scalar_snapshots(
        transient.support_work_j.canonical_values(),
        fea_transient_support_work_field_id,
        runmat_analysis_core::ObservationQuantity::SupportWork,
    );
    let damping_dissipation_snapshots = scalar_snapshots(
        transient.damping_dissipation_j.canonical_values(),
        fea_transient_damping_dissipation_field_id,
        runmat_analysis_core::ObservationQuantity::DampingDissipation,
    );
    let energy_balance_residual_snapshots = scalar_snapshots(
        transient.energy_balance_residual_j.canonical_values(),
        fea_transient_energy_balance_residual_field_id,
        runmat_analysis_core::ObservationQuantity::Balance,
    );
    let relative_energy_balance_error_snapshots = scalar_snapshots(
        &transient.relative_energy_balance_error,
        fea_transient_relative_energy_balance_error_field_id,
        runmat_analysis_core::ObservationQuantity::Balance,
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
    if let Some(field) = connector_force_snapshots.last() {
        let mut field = field.clone();
        field.field_id = crate::contracts::FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE.into();
        run.fields.push(field);
    }
    if let Some(field) = connector_stored_energy_snapshots.last() {
        let mut field = field.clone();
        field.field_id = crate::contracts::FEA_FIELD_STRUCTURAL_CONNECTOR_STORED_ENERGY.into();
        run.fields.push(field);
    }
    let electro_thermal_fields = recover_realized_electro_thermal_fields(
        &electro_thermal_summaries,
        &transient.residual_norms,
        reporting_dof_count,
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
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::Nodes,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &displacement_snapshots,
            },
            HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Velocity,
                source_field_id: TRANSIENT_VELOCITY_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Node,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::Nodes,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &velocity_snapshots,
            },
            HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Acceleration,
                source_field_id: TRANSIENT_ACCELERATION_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Node,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::Nodes,
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
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::Nodes,
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
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::VolumeElements,
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
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::VolumeElements,
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
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::VolumeElements,
                value_shape: runmat_analysis_core::ObservationValueShape::Scalar,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &strain_energy_density_snapshots,
            });
        }
        if connector_force_snapshots.len() == coordinates.len() {
            series.push(HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::ReactionForce,
                source_field_id: TRANSIENT_CONNECTOR_FORCE_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Connector,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::Connectors,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &connector_force_snapshots,
            });
        }
        if connector_stored_energy_snapshots.len() == coordinates.len() {
            series.push(HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::StrainEnergy,
                source_field_id: TRANSIENT_CONNECTOR_STORED_ENERGY_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Connector,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::Connectors,
                value_shape: runmat_analysis_core::ObservationValueShape::Scalar,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &connector_stored_energy_snapshots,
            });
        }
        evaluate_field_histories(
            execution_step,
            HistoryTopology::Structural {
                model: active_model,
                mesh: options.solver_mesh.as_ref(),
                connector_plan: connector_results.last().map(|result| &result.plan),
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
        connector_results,
        connector_force_snapshots,
        connector_stored_energy_snapshots,
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

fn scalar_snapshots(
    values: &[f64],
    field_id: impl Fn(usize) -> String,
    quantity: runmat_analysis_core::ObservationQuantity,
) -> Vec<AnalysisField> {
    let layout = runmat_analysis_core::AnalysisFieldLayout::scalar(
        quantity,
        runmat_analysis_core::FieldLocation::Global,
    );
    values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(field_id(index), layout.clone().step(index), vec![1], vec![value])
        })
        .collect()
}
