use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use base64::Engine as _;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactPublishedObjectSource, ProgramExecutionObject, ProgramExecutionResponse,
};
use runmat_execution_runner_native::{
    import_exact_structural_study_result, import_exact_structural_sweep_result,
    ExactStructuralStudyResultExpectation, ExactStructuralSweepResultExpectation,
    ExactStructuralSweepTerminal,
};
use runmat_runtime::analysis::{
    analysis_complete_study_sweep_op, structural_study_execution_failure, AnalysisStudyRunData,
    AnalysisStudySweepData, AnalysisStudySweepRunOutcome, AnalysisStudySweepSpec,
    PreparedStructuralStudyIntent, StructuralStudyBucklingNumericalResult,
    StructuralStudyExecutionFailure, StructuralStudyFrequencyNumericalResult,
    StructuralStudyModalNumericalResult, StructuralStudyModalTransientNumericalResult,
    StructuralStudyNumericalResult, StructuralStudyPrestressedFrequencyNumericalResult,
    StructuralStudyPrestressedModalNumericalResult, StructuralStudySequenceNumericalResult,
    StructuralStudyStepNumericalResult, StructuralStudyTransientNumericalResult,
};
use runmat_runtime::operations::OperationEnvelope;

use super::preparation::{artifact_access, authored_meshing_request};

pub(crate) fn materialize_remote_outputs(
    run_id: &str,
    response: ProgramExecutionResponse,
    objects: Vec<ProgramExecutionObject>,
    continuation: &str,
    signal_materializations: &[runmat_analysis_execution::SignalMaterializationExpectation],
    revision: ProgramRevision,
) -> Result<OperationEnvelope<AnalysisStudyRunData>> {
    let continuation = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(continuation)
        .context("saved structural study continuation is malformed")?;
    let study = PreparedStructuralStudyIntent::from_continuation_bytes(&continuation)
        .map_err(anyhow::Error::msg)?;
    let expectation = result_expectation(&study, signal_materializations, revision, run_id)?;
    let cache = DownloadedResultCache::from_objects(objects);
    let result = import_exact_structural_study_result(
        &cache,
        response,
        &expectation,
        ObjectInventoryLimits::default(),
    )?;
    let numerical = study_result(&result, cache)?;
    runmat_runtime::analysis::analysis_complete_structural_study_intent_op(
        study,
        runmat_runtime::meshing::MeshingExecutionOutput {
            artifact: result.mesh.artifact,
            evidence: result.mesh.evidence,
            boundary_region_ids: result.boundary_resolution.aliases,
        },
        numerical,
        runmat_runtime::analysis::StructuralStudyCompletionIdentity::new(run_id)
            .map_err(anyhow::Error::msg)?,
    )
    .map_err(super::super::report_operation_error)
}

pub(crate) fn materialize_remote_sweep_outputs(
    run_id: &str,
    response: ProgramExecutionResponse,
    objects: Vec<ProgramExecutionObject>,
    spec: &AnalysisStudySweepSpec,
    continuations: &[String],
    signal_materializations: &[Vec<runmat_analysis_execution::SignalMaterializationExpectation>],
    revision: ProgramRevision,
) -> Result<OperationEnvelope<AnalysisStudySweepData>> {
    let intents = continuations
        .iter()
        .map(|continuation| {
            let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(continuation)
                .context("saved structural sweep continuation is malformed")?;
            PreparedStructuralStudyIntent::from_continuation_bytes(&bytes)
                .map_err(anyhow::Error::msg)
        })
        .collect::<Result<Vec<_>>>()?;
    validate_continuations(spec, &intents)?;
    if signal_materializations.len() != intents.len() {
        anyhow::bail!("saved signal expectation count differs from the structural sweep");
    }
    let context = intents
        .first()
        .map(|intent| intent.operation_context().clone())
        .context("structural sweep contains no study intent")?;
    let definition = runmat_runtime::analysis::analysis_study_sweep_definition(spec, &context)
        .map_err(super::super::report_operation_error)?;
    let expectation = ExactStructuralSweepResultExpectation {
        program_revision: revision.clone(),
        artifact_access: artifact_access(run_id),
        definition,
        studies: intents
            .iter()
            .zip(signal_materializations)
            .map(|(intent, signals)| result_expectation(intent, signals, revision.clone(), run_id))
            .collect::<Result<Vec<_>>>()?,
    };
    let cache = DownloadedResultCache::from_objects(objects);
    let result = import_exact_structural_sweep_result(
        &cache,
        response,
        &expectation,
        ObjectInventoryLimits::default(),
    )?;
    let outcomes = result
        .entries
        .into_iter()
        .zip(intents)
        .map(|(entry, intent)| match entry.terminal {
            ExactStructuralSweepTerminal::Succeeded(result) => {
                let completion = runmat_runtime::analysis::StructuralStudyCompletionIdentity::new(
                    format!("{run_id}_study_{}", entry.study_index),
                )
                .map_err(anyhow::Error::msg)?;
                let numerical = study_result(&result, cache.clone())?;
                let envelope =
                    runmat_runtime::analysis::analysis_complete_structural_study_intent_op(
                        intent,
                        runmat_runtime::meshing::MeshingExecutionOutput {
                            artifact: result.mesh.artifact,
                            evidence: result.mesh.evidence,
                            boundary_region_ids: result.boundary_resolution.aliases,
                        },
                        numerical,
                        completion,
                    )
                    .map_err(super::super::report_operation_error)?;
                Ok(AnalysisStudySweepRunOutcome::Succeeded(Box::new(envelope)))
            }
            ExactStructuralSweepTerminal::Failed(message) => Ok(
                AnalysisStudySweepRunOutcome::Failed(structural_study_execution_failure(
                    &intent,
                    StructuralStudyExecutionFailure::Failed(message),
                )),
            ),
            ExactStructuralSweepTerminal::Cancelled => Ok(AnalysisStudySweepRunOutcome::Cancelled(
                structural_study_execution_failure(
                    &intent,
                    StructuralStudyExecutionFailure::Cancelled,
                ),
            )),
            ExactStructuralSweepTerminal::Indeterminate(message) => Ok(
                AnalysisStudySweepRunOutcome::Indeterminate(structural_study_execution_failure(
                    &intent,
                    StructuralStudyExecutionFailure::Indeterminate(message),
                )),
            ),
        })
        .collect::<Result<Vec<_>>>()?;
    analysis_complete_study_sweep_op(spec, context, outcomes)
        .map_err(super::super::report_operation_error)
}

fn validate_continuations(
    spec: &AnalysisStudySweepSpec,
    intents: &[PreparedStructuralStudyIntent],
) -> Result<()> {
    if intents.len() != spec.studies.len() {
        anyhow::bail!("structural sweep continuation count differs from its authored studies");
    }
    for (intent, authored) in intents.iter().zip(&spec.studies) {
        if intent.study_id() != authored.study_id
            || intent.authored_study_digest()
                != runmat_runtime::analysis::analysis_study_content_digest(authored)
        {
            anyhow::bail!("structural sweep continuation differs from its authored study");
        }
    }
    Ok(())
}

fn result_expectation(
    study: &PreparedStructuralStudyIntent,
    signal_materializations: &[runmat_analysis_execution::SignalMaterializationExpectation],
    revision: ProgramRevision,
    authority: &str,
) -> Result<ExactStructuralStudyResultExpectation> {
    let expectation = ExactStructuralStudyResultExpectation {
        program_revision: revision.clone(),
        artifact_access: artifact_access(authority),
        authored_study_digest: study.authored_study_digest(),
        frozen_project_digest: Some(*revision.graph_digest()),
        geometry_source_digest: study.geometry_source_digest(),
        workloads: study.workloads(),
        analysis_accelerator_requests: Vec::new(),
        signal_materializations: signal_materializations.to_vec(),
        expected_meshing_request: authored_meshing_request(study)?,
    };
    expectation.validate()?;
    Ok(expectation)
}

fn study_result(
    result: &runmat_execution_runner_native::ExactStructuralStudyResult,
    source: DownloadedResultCache,
) -> Result<StructuralStudyNumericalResult> {
    Ok(match &result.analysis {
        runmat_execution_runner_native::ExactStructuralStudyAnalysisResult::LinearSequence {
            sequence,
            ..
        } => {
            StructuralStudyNumericalResult::LinearSequence(StructuralStudySequenceNumericalResult {
                frozen_project_digest: result.frozen_project_digest,
                steps: sequence
                    .steps
                    .iter()
                    .map(|step| StructuralStudyStepNumericalResult {
                        step_id: step.step_id.clone(),
                        solver_result_identity: step.result_identity,
                        result: step.result.run.clone(),
                    })
                    .collect(),
                states: sequence
                    .states
                    .iter()
                    .map(|state| state.state.clone())
                    .collect(),
            })
        }
        runmat_execution_runner_native::ExactStructuralStudyAnalysisResult::StructuralTransient {
            resolved_workload,
            result: transient,
            numerical_receipt,
        } => StructuralStudyNumericalResult::StructuralTransient(Box::new(
            StructuralStudyTransientNumericalResult {
                frozen_project_digest: result.frozen_project_digest,
                step_id: resolved_workload.step_id.clone(),
                result: (**transient).clone(),
                numerical_receipt: numerical_receipt.clone(),
            },
        )),
        runmat_execution_runner_native::ExactStructuralStudyAnalysisResult::StructuralModal {
            resolved_workload,
            result: modal,
            numerical_receipt,
        } => StructuralStudyNumericalResult::StructuralModal(Box::new(
            StructuralStudyModalNumericalResult {
                frozen_project_digest: result.frozen_project_digest,
                step_id: resolved_workload.step_id().clone(),
                result: (**modal).clone(),
                numerical_receipt: numerical_receipt.clone(),
            },
        )),
        runmat_execution_runner_native::ExactStructuralStudyAnalysisResult::StructuralModalTransient(
            modal_transient,
        ) => StructuralStudyNumericalResult::StructuralModalTransient(Box::new(
            StructuralStudyModalTransientNumericalResult {
                frozen_project_digest: result.frozen_project_digest,
                modal_step_id: modal_transient.modal_workload.step_id().clone(),
                transient_step_id: modal_transient.transient_workload.step_id.clone(),
                modal_result: (*modal_transient.modal_result).clone(),
                modal_numerical_receipt: modal_transient.modal_numerical_receipt.clone(),
                basis_state: modal_transient.basis_state.clone(),
                transient_result: (*modal_transient.transient_result).clone(),
                transient_numerical_receipt: modal_transient
                    .transient_numerical_receipt
                    .clone(),
            },
        )),
        runmat_execution_runner_native::ExactStructuralStudyAnalysisResult::StructuralBuckling(
            buckling,
        ) => StructuralStudyNumericalResult::StructuralBuckling(Box::new(
            StructuralStudyBucklingNumericalResult {
                frozen_project_digest: result.frozen_project_digest,
                preload_step_id: buckling.preload_workload.step_id.clone(),
                buckling_step_id: buckling.buckling_workload.step_id().clone(),
                preload_result: buckling.preload_result.run.clone(),
                preload_solver_result_identity: buckling.preload_result_identity,
                prestress_state: buckling.prestress_state.clone(),
                buckling_result: (*buckling.buckling_result).clone(),
                buckling_numerical_receipt: buckling.buckling_numerical_receipt.clone(),
            },
        )),
        runmat_execution_runner_native::ExactStructuralStudyAnalysisResult::StructuralPrestressedModal(
            modal,
        ) => StructuralStudyNumericalResult::StructuralPrestressedModal(Box::new(
            StructuralStudyPrestressedModalNumericalResult {
                frozen_project_digest: result.frozen_project_digest,
                preload_step_id: modal.preload_workload.step_id.clone(),
                modal_step_id: modal.modal_workload.step_id().clone(),
                preload_result: modal.preload_result.run.clone(),
                preload_solver_result_identity: modal.preload_result_identity,
                prestress_state: modal.prestress_state.clone(),
                modal_result: (*modal.modal_result).clone(),
                modal_numerical_receipt: modal.modal_numerical_receipt.clone(),
            },
        )),
        runmat_execution_runner_native::ExactStructuralStudyAnalysisResult::StructuralPrestressedFrequency(
            frequency,
        ) => {
            let nested = StructuralStudyFrequencyNumericalResult::new(
                result.frozen_project_digest,
                (*frequency.frequency.resolved_workload).clone(),
                (*frequency.frequency.result).clone(),
                (*frequency.frequency.publication).clone(),
                frequency.frequency.object_catalog.clone(),
                frequency.frequency.artifact_roots.clone(),
                ArtifactPublishedObjectSource::new(source),
            )
            .map_err(anyhow::Error::msg)?;
            StructuralStudyNumericalResult::StructuralPrestressedFrequency(Box::new(
                StructuralStudyPrestressedFrequencyNumericalResult {
                    frozen_project_digest: result.frozen_project_digest,
                    preload_step_id: frequency.preload_workload.step_id.clone(),
                    frequency_step_id: frequency.frequency.resolved_workload.step_id().clone(),
                    preload_result: frequency.preload_result.run.clone(),
                    preload_solver_result_identity: frequency.preload_result_identity,
                    prestress_state: frequency.prestress_state.clone(),
                    frequency: nested,
                },
            ))
        }
        runmat_execution_runner_native::ExactStructuralStudyAnalysisResult::StructuralFrequency(
            frequency,
        ) => StructuralStudyNumericalResult::StructuralFrequency(Box::new(
            StructuralStudyFrequencyNumericalResult::new(
                result.frozen_project_digest,
                (*frequency.resolved_workload).clone(),
                (*frequency.result).clone(),
                (*frequency.publication).clone(),
                frequency.object_catalog.clone(),
                frequency.artifact_roots.clone(),
                ArtifactPublishedObjectSource::new(source),
            )
            .map_err(anyhow::Error::msg)?,
        )),
    })
}

#[derive(Clone)]
struct DownloadedResultCache(Arc<BTreeMap<Digest, Vec<u8>>>);

impl DownloadedResultCache {
    fn from_objects(objects: Vec<ProgramExecutionObject>) -> Self {
        Self(Arc::new(
            objects
                .into_iter()
                .map(|object| (object.reference.logical_digest, object.bytes))
                .collect(),
        ))
    }
}

impl CacheImport for DownloadedResultCache {
    fn read_verified(
        &self,
        digest: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        let Some(bytes) = self.0.get(&digest) else {
            return Ok(None);
        };
        if Digest::sha256(bytes) != digest {
            return Err(runmat_execution_artifact::ArtifactError::Identity(
                "downloaded structural result differs from its digest".into(),
            ));
        }
        Ok(Some(bytes.clone()))
    }
}
