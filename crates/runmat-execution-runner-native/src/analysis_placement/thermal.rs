//! Typed local thermal-prefix placement and terminal result translation.
use super::*;
use crate::ThermalPrestressStructuralCompletion;
use runmat_runtime::analysis::{
    StructuralStudyPrestressedFrequencyReductionResult,
    StructuralStudyPrestressedModalFrequencyReductionResult,
};
mod source;
pub(crate) use source::convert as convert_completed_thermal_study;
pub use source::thermal_study_completion_from_source;

pub(super) async fn execute(
    config: NativeExecutionConfig,
    preparation: crate::ThermalPrestressStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> crate::NativeExecutionResult<StructuralStudyNumericalResult> {
    let prefix = preparation
        .structural
        .workloads
        .prestress_prefix()
        .map_err(crate::analysis_task_pool::protocol)?;
    let preload_step_id = prefix.preload.step_id.clone();
    let consumers = prefix.consumer_step_ids;
    let frozen_project_digest = preparation.structural.frozen_project_digest;
    let access = preparation.structural.artifact_access.clone();
    let (completed, source) =
        crate::execute_thermal_prestress_study_locally_with_source(config, preparation, policy)
            .await?;
    let (thermal, thermal_source) = source::convert(completed.thermal, source.clone())
        .map_err(crate::analysis_task_pool::protocol)?;
    let structural = match completed.structural {
        ThermalPrestressStructuralCompletion::Buckling(result) => {
            StructuralStudyNumericalResult::StructuralBuckling(Box::new(
                StructuralStudyBucklingNumericalResult {
                    frozen_project_digest,
                    preload_step_id,
                    buckling_step_id: consumers[0].clone(),
                    preload_result: result.preload.result,
                    preload_solver_result_identity: result.preload.identity,
                    preload_numerical_receipt: result.preload.numerical_receipt,
                    prestress_state: result.prestress_state.state,
                    buckling_result: result.buckling.result,
                    buckling_numerical_receipt: result.buckling.numerical_receipt,
                },
            ))
        }
        ThermalPrestressStructuralCompletion::Modal(result) => {
            StructuralStudyNumericalResult::StructuralPrestressedModal(Box::new(
                StructuralStudyPrestressedModalNumericalResult {
                    frozen_project_digest,
                    preload_step_id,
                    modal_step_id: consumers[0].clone(),
                    preload_result: result.preload.result,
                    preload_solver_result_identity: result.preload.identity,
                    preload_numerical_receipt: result.preload.numerical_receipt,
                    prestress_state: result.prestress_state.state,
                    modal_result: result.modal.result,
                    modal_numerical_receipt: result.modal.numerical_receipt,
                },
            ))
        }
        ThermalPrestressStructuralCompletion::Frequency(result) => {
            let (catalog, roots) = result.frequency.published_artifacts(&access)?;
            let frequency = StructuralStudyFrequencyNumericalResult::new(
                None,
                result.resolved_workload,
                result.frequency.result.result,
                result.frequency.publication,
                catalog,
                roots,
                ArtifactPublishedObjectSource::new(source),
            )
            .map_err(crate::analysis_task_pool::protocol)?;
            StructuralStudyNumericalResult::StructuralPrestressedFrequency(Box::new(
                StructuralStudyPrestressedFrequencyNumericalResult {
                    frozen_project_digest,
                    preload_step_id,
                    frequency_step_id: consumers[0].clone(),
                    preload_result: result.preload.result,
                    preload_solver_result_identity: result.preload.identity,
                    preload_numerical_receipt: result.preload.numerical_receipt,
                    prestress_state: result.prestress_state.state,
                    reduction: StructuralStudyPrestressedFrequencyReductionResult::Direct,
                    frequency,
                },
            ))
        }
        ThermalPrestressStructuralCompletion::ModalFrequency(result) => {
            let (catalog, roots) = result.frequency.published_artifacts(&access)?;
            let frequency = StructuralStudyFrequencyNumericalResult::new(
                None,
                result.resolved_workload,
                result.frequency.result.result,
                result.frequency.publication,
                catalog,
                roots,
                ArtifactPublishedObjectSource::new(source),
            )
            .map_err(crate::analysis_task_pool::protocol)?;
            StructuralStudyNumericalResult::StructuralPrestressedFrequency(Box::new(
                StructuralStudyPrestressedFrequencyNumericalResult {
                    frozen_project_digest,
                    preload_step_id,
                    frequency_step_id: consumers[1].clone(),
                    preload_result: result.preload.result,
                    preload_solver_result_identity: result.preload.identity,
                    preload_numerical_receipt: result.preload.numerical_receipt,
                    prestress_state: result.prestress_state.state,
                    reduction: StructuralStudyPrestressedFrequencyReductionResult::ModalReduced(
                        Box::new(StructuralStudyPrestressedModalFrequencyReductionResult {
                            modal_step_id: consumers[0].clone(),
                            modal_result: result.modal.result,
                            modal_numerical_receipt: result.modal.numerical_receipt,
                            basis_state: result.basis_state.state,
                        }),
                    ),
                    frequency,
                },
            ))
        }
    };
    Ok(StructuralStudyNumericalResult::ThermalPrestress {
        thermal: Box::new(thermal),
        thermal_source,
        structural: Box::new(structural),
    })
}
