use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel, AnalysisProcedure};

use crate::{
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    contracts::{
        fea_modal_mode_shape_field_id, ComputeBackend, FeaModalRunResult, FeaRunError,
        FeaRunResult, ModalRecoveredMode, ModalSolveOptions,
        FEA_FIELD_MODAL_CUMULATIVE_EFFECTIVE_MASS, FEA_FIELD_MODAL_CUMULATIVE_MASS_RATIO,
        FEA_FIELD_MODAL_EFFECTIVE_MASS, FEA_FIELD_MODAL_EIGENVALUE, FEA_FIELD_MODAL_FREQUENCY_HZ,
        FEA_FIELD_MODAL_MODAL_MASS, FEA_FIELD_MODAL_MODAL_STIFFNESS,
        FEA_FIELD_MODAL_M_ORTHOGONALITY, FEA_FIELD_MODAL_PARTICIPATION_FACTOR,
        FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION, FEA_FIELD_MODAL_RESIDUAL_NORM,
        FEA_FIELD_MODAL_TOTAL_FREE_MASS, FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        FEA_FIELD_STRUCTURAL_STRAIN, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
        FEA_FIELD_STRUCTURAL_STRESS,
    },
    diagnostics::builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    pipeline::modal_basis::ModalBasisArtifact,
    pipeline::observation_history::{
        evaluate_field_histories, required_mode_history_indices, HistoryFieldSeries,
        HistoryTopology,
    },
    pipeline::structural_nodal::{nodal_vector_shape, project_structural_nodal_snapshots},
    post::fields::recover_deformation_result_fields,
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::modal::solve_modal_system,
};

const MODAL_DISPLACEMENT_HISTORY_SOURCE: &str = "structural.modal.displacement";
const MODAL_ROTATION_HISTORY_SOURCE: &str = "structural.modal.rotation";
const MODAL_STRAIN_HISTORY_SOURCE: &str = "structural.modal.strain";
const MODAL_STRESS_HISTORY_SOURCE: &str = "structural.modal.stress";
const MODAL_STRAIN_ENERGY_HISTORY_SOURCE: &str = "structural.modal.strain_energy_density";

pub fn run_modal(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaModalRunResult, FeaRunError> {
    run_modal_with_options(model, backend, ModalSolveOptions::default())
}

pub fn run_modal_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ModalSolveOptions,
) -> Result<FeaModalRunResult, FeaRunError> {
    let execution_step = super::admission::require_structural(
        model,
        AnalysisProcedure::ModalExtraction,
        options.formulation_requirement,
    )?;
    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating modal FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_modal")?;
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    let required_history_modes = required_mode_history_indices(execution_step)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    if required_history_modes
        .last()
        .is_some_and(|mode| *mode > options.mode_count)
    {
        return Err(FeaRunError::Observation(
            "modal history requests a mode beyond the admitted solver mode count".into(),
        ));
    }
    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "modal model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_modal")?;

    let excitations =
        realize_step_excitations(model, execution_step, ExcitationSamplePoint::Static)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;

    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling modal system",
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
        "fea.run_modal",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "modal system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_modal")?;

    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "solving modal basis",
        Some(2),
        Some(5),
    );
    let modal = solve_modal_system(&summary, options.mode_count, backend);
    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "modal solve complete",
        Some(3),
        Some(5),
    );
    check_cancelled("fea.run_modal")?;

    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Started,
        "recovering modal fields",
        Some(3),
        Some(5),
    );
    let mut diagnostics = modal.diagnostics.clone();
    extend_common_run_diagnostics(
        &mut diagnostics,
        CommonRunDiagnosticInputs {
            model,
            summary: &summary,
        },
    );

    let projected_mode_shapes =
        project_structural_nodal_snapshots(&summary.structural_dof_layout, &modal.mode_shapes)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let basis = ModalBasisArtifact::new(
        identity.clone(),
        &summary,
        &modal.frequencies_hz,
        &modal.mode_shapes,
        &modal.residual_norms,
    )?;
    let recovered_modes = modal
        .mode_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            recover_deformation_result_fields(&summary, shape)
                .map(|fields| ModalRecoveredMode {
                    mode_index: index + 1,
                    fields,
                })
                .map_err(|error| FeaRunError::Assembly(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mode_shape = nodal_vector_shape(&summary.structural_dof_layout);
    let first_translation = basis
        .modes()
        .first()
        .and_then(|_| projected_mode_shapes.first())
        .map(|shape| shape.translations.clone())
        .unwrap_or_else(|| vec![0.0; mode_shape.iter().product()]);
    let mut run_fields = recovered_modes
        .first()
        .map(|mode| mode.fields.clone())
        .unwrap_or_else(|| {
            vec![AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_DISPLACEMENT,
                mode_shape.clone(),
                first_translation,
            )]
        });
    run_fields.extend(modal_result_fields(&basis));

    let mut run = FeaRunResult {
        identity,
        requested_backend: backend,
        execution_backend: crate::contracts::ExecutionBackend::from_solver_backend(
            modal.solver_backend,
        ),
        solver_backend: modal.solver_backend,
        solver_device_apply_k_ratio: if modal.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            modal.device_apply_k_count as f64 / modal.device_apply_k_attempt_count as f64
        },
        solver_method: modal.solver_method,
        preconditioner: if backend == ComputeBackend::Gpu {
            crate::contracts::SolverPreconditioner::Jacobi
        } else {
            crate::contracts::SolverPreconditioner::None
        },
        solver_host_sync_count: modal.solver_host_sync_count,
        accelerator_evidence: modal.accelerator_evidence.clone(),
        diagnostics,
        spatial_projection_evidence: summary.spatial_projection_evidence.clone(),
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields: run_fields,
    };

    let mode_shapes = projected_mode_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            AnalysisField::host_f64(
                fea_modal_mode_shape_field_id(index + 1),
                mode_shape.clone(),
                shape.translations.clone(),
            )
        })
        .collect::<Vec<_>>();
    let rotation_mode_shapes = projected_mode_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            AnalysisField::host_f64(
                format!("structural.modal.rotation.{}", index + 1),
                mode_shape.clone(),
                shape.rotations.clone(),
            )
        })
        .collect::<Vec<_>>();
    let strain_mode_fields =
        collect_recovered_mode_fields(&recovered_modes, FEA_FIELD_STRUCTURAL_STRAIN);
    let stress_mode_fields =
        collect_recovered_mode_fields(&recovered_modes, FEA_FIELD_STRUCTURAL_STRESS);
    let strain_energy_mode_fields =
        collect_recovered_mode_fields(&recovered_modes, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY);
    let evaluated_histories = if execution_step.observations.iter().any(|request| {
        matches!(
            request,
            runmat_analysis_core::ObservationRequest::History(_)
        )
    }) {
        let coordinates = (1..=mode_shapes.len())
            .map(crate::results::field::RecoveredFieldSample::ModeIndex)
            .collect::<Vec<_>>();
        let mut series = vec![
            HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Displacement,
                source_field_id: MODAL_DISPLACEMENT_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Node,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &mode_shapes,
            },
            HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Rotation,
                source_field_id: MODAL_ROTATION_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Node,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &rotation_mode_shapes,
            },
        ];
        if strain_mode_fields.len() == coordinates.len() {
            series.push(HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Strain,
                source_field_id: MODAL_STRAIN_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Element,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::SolidElements,
                value_shape: runmat_analysis_core::ObservationValueShape::SymmetricTensor3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &strain_mode_fields,
            });
        }
        if stress_mode_fields.len() == coordinates.len() {
            series.push(HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Stress,
                source_field_id: MODAL_STRESS_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Element,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::SolidElements,
                value_shape: runmat_analysis_core::ObservationValueShape::SymmetricTensor3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &stress_mode_fields,
            });
        }
        if strain_energy_mode_fields.len() == coordinates.len() {
            series.push(HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::StrainEnergy,
                source_field_id: MODAL_STRAIN_ENERGY_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Element,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::SolidElements,
                value_shape: runmat_analysis_core::ObservationValueShape::Scalar,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &strain_energy_mode_fields,
            });
        }
        evaluate_field_histories(
            execution_step,
            HistoryTopology::Structural {
                model,
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
        "fea.run_modal",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Completed,
        "modal result field recovery complete",
        Some(4),
        Some(5),
    );
    check_cancelled("fea.run_modal")?;
    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA modal run complete",
        Some(5),
        Some(5),
    );

    Ok(FeaModalRunResult {
        run,
        basis,
        mode_shapes,
        rotation_mode_shapes,
        recovered_modes,
    })
}

fn collect_recovered_mode_fields(
    modes: &[ModalRecoveredMode],
    field_id: &str,
) -> Vec<AnalysisField> {
    modes
        .iter()
        .filter_map(|mode| {
            mode.fields
                .iter()
                .find(|field| field.field_id == field_id)
                .cloned()
        })
        .collect()
}

fn modal_result_fields(basis: &ModalBasisArtifact) -> Vec<AnalysisField> {
    let mode_count = basis.modes().len();
    let directional = |select: fn(crate::pipeline::modal_basis::ModalDirectionalMetric) -> f64| {
        basis
            .modes()
            .iter()
            .flat_map(|mode| mode.directional().iter().copied().map(select))
            .collect::<Vec<_>>()
    };
    vec![
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_FREQUENCY_HZ,
            vec![mode_count],
            basis
                .modes()
                .iter()
                .map(|mode| mode.frequency_hz())
                .collect(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_EIGENVALUE,
            vec![mode_count],
            basis
                .modes()
                .iter()
                .map(|mode| mode.eigenvalue_rad2_per_s2())
                .collect(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_MODAL_MASS,
            vec![mode_count],
            basis.modes().iter().map(|mode| mode.modal_mass()).collect(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_MODAL_STIFFNESS,
            vec![mode_count],
            basis
                .modes()
                .iter()
                .map(|mode| mode.modal_stiffness())
                .collect(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_PARTICIPATION_FACTOR,
            vec![mode_count, 3],
            directional(|metric| metric.participation_factor()),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_EFFECTIVE_MASS,
            vec![mode_count, 3],
            directional(|metric| metric.effective_mass()),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_CUMULATIVE_EFFECTIVE_MASS,
            vec![mode_count, 3],
            directional(|metric| metric.cumulative_effective_mass()),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_CUMULATIVE_MASS_RATIO,
            vec![mode_count, 3],
            directional(|metric| metric.cumulative_mass_ratio()),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_TOTAL_FREE_MASS,
            vec![3],
            basis.total_free_mass().to_vec(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_RESIDUAL_NORM,
            vec![mode_count],
            basis
                .modes()
                .iter()
                .map(|mode| mode.residual_norm())
                .collect(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION,
            vec![mode_count],
            basis
                .modes()
                .iter()
                .map(|mode| mode.relative_frequency_separation())
                .collect(),
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_M_ORTHOGONALITY,
            vec![mode_count, mode_count],
            basis.mass_orthogonality().to_vec(),
        ),
    ]
}
