use runmat_analysis_core::{AnalysisFieldLayout, ObservationQuantity, FieldLocation};
use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use super::CompletedModalCore;
use crate::{
    assembly::{
        assemble_structural_perturbation_system, assemble_structural_system,
        StructuralFormulationRequirement,
    },
    contracts::{
        fea_modal_mode_shape_field_id, FeaModalRunResult, FeaRunError, ModalRecoveredMode,
        FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_STRAIN,
        FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY, FEA_FIELD_STRUCTURAL_STRESS,
    },
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    execution::{
        CanonicalSemanticState, StructuralModalNumericalReceipt, StructuralModalSolverResult,
        StructuralModalWorkload,
    },
    pipeline::observation_history::{
        evaluate_field_histories, HistoryFieldSeries, HistoryTopology,
    },
    pipeline::structural_nodal::{nodal_vector_shape, project_structural_nodal_snapshots},
    post::fields::recover_deformation_result_fields,
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
};

const MODAL_DISPLACEMENT_HISTORY_SOURCE: &str = "structural.modal.displacement";
const MODAL_ROTATION_HISTORY_SOURCE: &str = "structural.modal.rotation";
const MODAL_STRAIN_HISTORY_SOURCE: &str = "structural.modal.strain";
const MODAL_STRESS_HISTORY_SOURCE: &str = "structural.modal.stress";
const MODAL_STRAIN_ENERGY_HISTORY_SOURCE: &str = "structural.modal.strain_energy_density";

/// Reconstructs result-facing modal fields from a portable solver result without repeating the
/// eigensolve.
///
/// The immutable basis remains the numerical authority. Recovery reassembles the exact authored
/// operator so the basis, topology, DOF layout, and run identity are all revalidated before its
/// DOF vectors are projected into nodal and formulation-owned fields.
pub fn recover_completed_structural_modal_run(
    workload: &StructuralModalWorkload,
    solver_mesh: &SolverMeshArtifact,
    result: &StructuralModalSolverResult,
    numerical_receipt: &StructuralModalNumericalReceipt,
) -> Result<FeaModalRunResult, FeaRunError> {
    if workload.prestress_transition().is_some() {
        return Err(FeaRunError::InvalidModel(
            "prestressed modal recovery requires its exact geometric-stiffness state".into(),
        ));
    }
    recover_completed_structural_modal_run_with_state(
        workload,
        solver_mesh,
        None,
        result,
        numerical_receipt,
    )
}

pub fn recover_completed_structural_prestressed_modal_run(
    workload: &StructuralModalWorkload,
    solver_mesh: &SolverMeshArtifact,
    prestress_state: &CanonicalSemanticState,
    result: &StructuralModalSolverResult,
    numerical_receipt: &StructuralModalNumericalReceipt,
) -> Result<FeaModalRunResult, FeaRunError> {
    recover_completed_structural_modal_run_with_state(
        workload,
        solver_mesh,
        Some(prestress_state),
        result,
        numerical_receipt,
    )
}

fn recover_completed_structural_modal_run_with_state(
    workload: &StructuralModalWorkload,
    solver_mesh: &SolverMeshArtifact,
    prestress_state: Option<&CanonicalSemanticState>,
    result: &StructuralModalSolverResult,
    numerical_receipt: &StructuralModalNumericalReceipt,
) -> Result<FeaModalRunResult, FeaRunError> {
    workload
        .validate()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    result
        .validate_for(workload)
        .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    numerical_receipt
        .validate_against(numerical_receipt.source_result(), result)
        .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    solver_mesh
        .validate_canonical()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let topology_digest = Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    if result.topology_digest() != topology_digest
        || solver_mesh.geometry.geometry_revision != u64::from(workload.model().geometry_revision)
    {
        return Err(FeaRunError::InvalidModel(
            "portable modal result does not belong to the supplied solver mesh".into(),
        ));
    }
    let linearization = match (workload.prestress_transition(), prestress_state) {
        (None, None) => None,
        (Some(transition), Some(state)) => Some(
            super::super::prestress::structural_linearization_from_state(
                transition,
                state,
                topology_digest,
            )
            .map_err(FeaRunError::InvalidModel)?,
        ),
        _ => {
            return Err(FeaRunError::InvalidModel(
                "modal workload and prestress-state presence differ during recovery".into(),
            ))
        }
    };

    let execution_step = super::super::admission::require_structural(
        workload.model(),
        workload
            .procedure_step()
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?
            .procedure,
    )?;
    validate_model(workload.model())
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let excitations = realize_step_excitations(
        workload.model(),
        execution_step,
        ExcitationSamplePoint::Static,
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let assemble = if linearization.is_some() {
        assemble_structural_perturbation_system
    } else {
        assemble_structural_system
    };
    let summary = assemble(
        workload.model(),
        &excitations,
        Some(solver_mesh.clone()),
        StructuralFormulationRequirement::SyntheticOperator,
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let summary = if let Some(linearization) = linearization {
        super::super::prestress::prestressed_structural_summary(&summary, linearization)
            .map_err(FeaRunError::Assembly)?
    } else {
        summary
    };
    let expected_run = super::super::admission::admit_structural(&summary, execution_step)?;
    if result.run().identity != expected_run {
        return Err(FeaRunError::Assembly(
            "portable modal result run identity differs from the recovered operator".into(),
        ));
    }
    result.basis().validate_for(&summary)?;

    recover_modal_core(
        workload.model(),
        execution_step,
        Some(solver_mesh),
        CompletedModalCore {
            run: result.run().clone(),
            basis: result.basis().clone(),
            summary,
        },
    )
}

pub(super) fn recover_modal_core(
    model: &AnalysisModel,
    execution_step: &runmat_analysis_core::ProcedureStep,
    solver_mesh: Option<&SolverMeshArtifact>,
    core: CompletedModalCore,
) -> Result<FeaModalRunResult, FeaRunError> {
    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Started,
        "recovering modal fields",
        Some(3),
        Some(5),
    );
    let CompletedModalCore {
        mut run,
        basis,
        summary,
    } = core;
    let dof_shapes = basis
        .modes()
        .iter()
        .map(|mode| mode.dof_shape().to_vec())
        .collect::<Vec<_>>();
    let projected_mode_shapes =
        project_structural_nodal_snapshots(&summary.structural_dof_layout, &dof_shapes)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let recovered_modes = dof_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            recover_deformation_result_fields(&summary, shape)
                .map(|fields| ModalRecoveredMode {
                    mode_index: index + 1,
                    // Fields recovered from an eigenvector are that mode's normalized shape.
                    fields: fields
                        .into_iter()
                        .map(|mut field| {
                            field.layout.ordinal = runmat_analysis_core::FieldOrdinal::Mode(index);
                            field
                        })
                        .collect(),
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
    let mut recovered_run_fields = recovered_modes
        .first()
        .map(|mode| mode.fields.clone())
        .unwrap_or_else(|| {
            vec![AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_DISPLACEMENT, AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node).mode(0),
                mode_shape.clone(),
                first_translation,
            )]
        });
    recovered_run_fields.retain(|field| !field.is_empty());
    recovered_run_fields.extend(run.fields);
    run.fields = recovered_run_fields;

    let mode_shapes = projected_mode_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            AnalysisField::host_f64(
                fea_modal_mode_shape_field_id(index + 1), AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node).mode(index),
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
                format!("structural.modal.rotation.{}", index + 1), AnalysisFieldLayout::vector3(ObservationQuantity::Rotation, FieldLocation::Node).mode(index),
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
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::Nodes,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &mode_shapes,
            },
            HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Rotation,
                source_field_id: MODAL_ROTATION_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Node,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::Nodes,
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
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::VolumeElements,
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
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::VolumeElements,
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
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::VolumeElements,
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
                mesh: solver_mesh,
                connector_plan: None,
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
