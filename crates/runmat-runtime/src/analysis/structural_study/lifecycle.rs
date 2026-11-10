use std::collections::BTreeMap;

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

use super::publication::{
    complete_study_run_prepared, CompletedStudyStep, CompletedStudyStepRun, DeferredStudyRun,
    PreparedStudyCompletion,
};
use super::{
    AnalysisStudyCompletionIdentity, PreparedStructuralStudyIntent,
    PreparedStructuralStudyIntentStep,
};
use crate::analysis::{
    analysis_admit_buckling_run_with_id, analysis_admit_linear_static_run_with_id,
    analysis_admit_modal_run_with_id, analysis_admit_transient_run_with_id,
    analysis_study_content_digest, prepare_linear_static_workload_template,
    prepare_study_semantics, study_fingerprint_from_digest, AnalysisSolverOptions,
    AnalysisStudyRunData, AnalysisStudySpec, PreparedStructuralLinearRun,
    StructuralStudyModalNumericalResult, StructuralStudyModalTransientNumericalResult,
    StructuralStudyNumericalResult, StructuralStudySequenceNumericalResult,
    StructuralStudyTransientNumericalResult, ANALYSIS_RUN_STUDY_OPERATION,
    ANALYSIS_RUN_STUDY_OP_VERSION,
};

pub(in crate::analysis) mod accuracy_completion;
mod intent_completion;
pub(in crate::analysis) use intent_completion::prepare_structural_study_intent_completion;
mod buckling_completion;
mod errors;
mod frequency_completion;
mod modal_completion;
mod modal_frequency_completion;
mod modal_intent_completion;
pub(in crate::analysis) use frequency_completion::complete_frequency_study;
use frequency_completion::complete_structural_frequency;
mod preparation;
mod prestressed_frequency_completion;
mod prestressed_frequency_modal_reduction;
mod prestressed_frequency_states;
mod prestressed_modal_completion;
mod random_response_completion;
mod response_spectrum_completion;
mod statistical_persistence;
mod thermal_completion;
pub use errors::structural_study_execution_failure;
use errors::{completion_error, intent_completion_error};

pub(in crate::analysis) use buckling_completion::complete_buckling_study;
use buckling_completion::complete_structural_buckling;
pub(in crate::analysis) use modal_completion::{
    complete_modal_study, complete_modal_transient_study,
};
use modal_frequency_completion::{complete_structural_modal_frequency, ModalFrequencyMeshes};
use modal_intent_completion::{complete_structural_modal, complete_structural_modal_transient};
pub use preparation::analysis_prepare_structural_study_intent_op;
pub(in crate::analysis) use preparation::{
    complete_response_spectrum_modal_options, prepare_structural_buckling_template,
    prepare_structural_frequency_workload, prepare_structural_modal_transient_template,
    prepare_structural_modal_workload, prepare_structural_prestressed_frequency_template,
    prepare_structural_prestressed_modal_template, prepare_structural_random_response_template,
    prepare_structural_response_spectrum_template,
};
pub(in crate::analysis) use prestressed_frequency_completion::complete_prestressed_frequency_study;
use prestressed_frequency_completion::complete_structural_prestressed_frequency;
pub(in crate::analysis) use prestressed_modal_completion::complete_prestressed_modal_study;
use prestressed_modal_completion::complete_structural_prestressed_modal;
pub(in crate::analysis) use random_response_completion::complete_random_response_study;
use random_response_completion::complete_structural_random_response;
pub(in crate::analysis) use response_spectrum_completion::complete_response_spectrum_study;
use response_spectrum_completion::complete_structural_response_spectrum;

/// Admits the remotely meshed and solved result of a path-free structural study intent.
///
/// Canonical mesh and evidence validation, boundary binding, numerical result admission, quality
/// classification, and persistence remain client-runtime responsibilities. The server allocation
/// identity deterministically fences local run publication. Retrying or concurrently attaching the
/// same completed server run therefore converges on one public run ID; collision checks in the
/// artifact store reject different content under that identity.
///
/// Mesh and immutable provenance objects may be prepared independently. Numerical run records
/// remain private until the exact closed study inventory and its manifest commit together.
pub fn analysis_complete_structural_study_intent_op(
    intent: PreparedStructuralStudyIntent,
    meshing_output: crate::meshing::MeshingExecutionOutput,
    result: StructuralStudyNumericalResult,
    completion_identity: AnalysisStudyCompletionIdentity,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let mut preparation =
        crate::analysis::study_publication::preparation::PublicationPreparation::capture(
            intent.authored_study_digest,
            &(
                &intent,
                &meshing_output.artifact,
                &meshing_output.evidence,
                &result,
            ),
        )
        .map_err(|error| {
            super::publication::completion_storage_error(
                &intent.context,
                "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED",
                error,
            )
        })?;
    preparation
        .admit_structural_sources(&result)
        .map_err(|error| {
            super::publication::completion_storage_error(
                &intent.context,
                "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED",
                error,
            )
        })?;
    prepare_structural_study_intent_completion(
        intent,
        meshing_output,
        result,
        completion_identity,
        preparation,
    )?
    .publish_legacy()
}

fn complete_linear_sequence(
    intent: PreparedStructuralStudyIntent,
    study_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: super::super::meshing::ResolvedStudyMesh,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    mut authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralLinearWorkload,
    >,
    result: StructuralStudySequenceNumericalResult,
    completion_identity: AnalysisStudyCompletionIdentity,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let effective_memory_ceiling_bytes = result.effective_memory_ceiling_bytes;
    let mut numerical_by_step = BTreeMap::new();
    for step in result.steps {
        if numerical_by_step
            .insert(step.step_id.clone(), step)
            .is_some()
        {
            return Err(intent_completion_error(
                &intent,
                "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                OperationErrorType::Contract,
                OperationErrorSeverity::Fatal,
                "remote sequence contains a duplicate procedure-step result",
            ));
        }
    }
    let mut completion_steps = Vec::with_capacity(intent.steps.len());
    let mut authored_steps = Vec::with_capacity(intent.steps.len());
    for step in intent.steps.clone() {
        let numerical = numerical_by_step
            .remove(&step.procedure_step.step_id)
            .ok_or_else(|| {
                intent_completion_error(
                    &intent,
                    "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                    OperationErrorType::Contract,
                    OperationErrorSeverity::Fatal,
                    "remote sequence omitted an authored procedure-step result",
                )
            })?;
        let mut workload = authored_workloads
            .remove(&step.procedure_step.step_id)
            .ok_or_else(|| {
                intent_completion_error(
                    &intent,
                    "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                    OperationErrorType::Contract,
                    OperationErrorSeverity::Fatal,
                    "prepared linear workload is absent for an authored procedure step",
                )
            })?;
        workload.model = study_model.clone();
        let AnalysisSolverOptions::StructuralLinear(mut options) =
            step.authored_execution.solver_options.clone()
        else {
            return Err(intent_completion_error(
                &intent,
                "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                OperationErrorType::Contract,
                OperationErrorSeverity::Fatal,
                "linear sequence workload has non-linear authored solver options",
            ));
        };
        options.solver_mesh_artifact_path = persisted_mesh.artifact_path.clone();
        let original_fields =
            runmat_analysis_fea::pipeline::linear_static::original_static_displacement_source(
                &workload,
                Some(&solver_mesh),
                numerical.solver_result_identity,
                &numerical.result.run,
            )
            .map_err(|error| {
                intent_completion_error(
                    &intent,
                    "RM.FEA.RUN_STUDY.ORIGINAL_FIELDS_INVALID",
                    OperationErrorType::Contract,
                    OperationErrorSeverity::Fatal,
                    error.to_string(),
                )
            })?;
        let prepared_linear = PreparedStructuralLinearRun {
            workload,
            solver_mesh: Some(solver_mesh.clone()),
            options: options.clone(),
            context: intent.context.clone(),
        };
        let assigned_run_id = completion_identity.local_run_id(
            runmat_analysis_core::PhysicsFamily::Structural,
            intent.authored_study_digest,
            &step.procedure_step.step_id,
        );
        let run = analysis_admit_linear_static_run_with_id(
            prepared_linear,
            numerical.result.run,
            Some(&assigned_run_id),
            crate::analysis::StaticSolvePlacement {
                effective_memory_ceiling_bytes,
                static_solve_path: numerical.static_solve_path,
            },
        )?;
        let mut authored_execution = step.authored_execution;
        authored_execution.solver_options =
            AnalysisSolverOptions::StructuralLinear(options.clone());
        authored_steps.push(super::PreparedStudyStep {
            authored_execution,
            procedure_step: step.procedure_step.clone(),
        });
        completion_steps.push(CompletedStudyStep {
            step_id: step.procedure_step.step_id,
            solver_result_identity: numerical.solver_result_identity,
            run: super::publication::original_fields::standard(run, original_fields),
            resolved_solver_options: AnalysisSolverOptions::StructuralLinear(options),
        });
    }
    if !numerical_by_step.is_empty() {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "remote sequence returned an unauthored procedure-step result",
        ));
    }
    if !authored_workloads.is_empty() {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "prepared linear workload inventory contains an unauthored step",
        ));
    }
    complete_study_run_prepared(
        PreparedStudyCompletion {
            thermal_prefix: None,
            context: intent.context,
            study_id: intent.study_id,
            study_fingerprint: study_fingerprint_from_digest(intent.authored_study_digest),
            authored_study_digest: intent.authored_study_digest,
            model: study_model,
            steps: authored_steps,
            operation_sequence: intent.operation_sequence,
            solver_mesh_artifact_path: persisted_mesh.artifact_path,
            meshing_evidence_artifact_path: persisted_mesh.evidence_path,
        },
        completion_steps,
        result.states,
        result.frozen_project_digest,
        preparation,
    )
}

fn complete_direct_transient(
    intent: PreparedStructuralStudyIntent,
    study_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: super::super::meshing::ResolvedStudyMesh,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    mut workload: runmat_analysis_fea::execution::StructuralTransientWorkload,
    result: StructuralStudyTransientNumericalResult,
    completion_identity: AnalysisStudyCompletionIdentity,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    if result.step_id != workload.step_id {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "remote direct-transient result names a different procedure step",
        ));
    }
    let [step] = intent.steps.as_slice() else {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "direct-transient completion requires exactly one authored procedure step",
        ));
    };
    let AnalysisSolverOptions::StructuralTransient(mut options) =
        step.authored_execution.solver_options.clone()
    else {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "direct-transient workload has non-transient authored solver options",
        ));
    };
    workload.model = study_model.clone();
    let source_producer = result.result.manifest().scope.step;
    let solver_result_identity = result.result.manifest().identity;
    let recovered = runmat_analysis_fea::pipeline::transient::recover_completed_transient_run(
        runmat_analysis_fea::pipeline::transient::CompletedStructuralTransientRunInput {
            workload: &workload,
            solver_mesh: &solver_mesh,
            modal_state: None,
            result: result.result,
            numerical_receipt: &result.numerical_receipt,
        },
    )
    .map_err(|error| {
        intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.TRANSIENT_RESULT_ADMISSION_FAILED",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            error.to_string(),
        )
    })?;
    let original_fields =
        runmat_analysis_fea::pipeline::transient::original_transient_field_source(
            &study_model,
            &solver_mesh,
            source_producer,
            solver_result_identity,
            &recovered,
        )
        .map_err(|error| {
            intent_completion_error(
                &intent,
                "RM.FEA.RUN_STUDY.ORIGINAL_FIELDS_INVALID",
                OperationErrorType::Contract,
                OperationErrorSeverity::Fatal,
                error.to_string(),
            )
        })?;
    options.solver_mesh_artifact_path = persisted_mesh.artifact_path.clone();
    let assigned_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &step.procedure_step.step_id,
    );
    let run = analysis_admit_transient_run_with_id(
        workload.backend,
        options.clone(),
        &intent.context,
        Some(solver_mesh.clone()),
        recovered,
        Some(&assigned_run_id),
    )?;
    let sequence_identity = runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
        &study_model.sequence,
    )
    .map_err(|error| {
        intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            error.to_string(),
        )
    })?;
    let producer = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        sequence_identity,
        &step.procedure_step,
    )
    .map_err(|error| {
        intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            error.to_string(),
        )
    })?;
    let topology_digest =
        runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    let state_fields = super::super::collect_semantic_state_fields(&run.data);
    let states = step
        .procedure_step
        .state_outputs
        .iter()
        .map(|declaration| {
            runmat_analysis_fea::execution::CanonicalSemanticState::new(
                runmat_analysis_fea::execution::CanonicalSemanticStateInput {
                    producer,
                    declaration: declaration.clone(),
                    topology_digest,
                    source_result: solver_result_identity,
                    payload:
                        runmat_analysis_fea::execution::CanonicalSemanticStatePayload::Fields {
                            fields: state_fields.clone(),
                        },
                },
            )
            .map_err(|error| {
                intent_completion_error(
                    &intent,
                    "RM.FEA.RUN_STUDY.STATE_INVALID",
                    OperationErrorType::Contract,
                    OperationErrorSeverity::Fatal,
                    error.to_string(),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let authored_execution = {
        let mut authored = step.authored_execution.clone();
        authored.solver_options = AnalysisSolverOptions::StructuralTransient(options.clone());
        authored
    };
    complete_study_run_prepared(
        PreparedStudyCompletion {
            thermal_prefix: None,
            context: intent.context,
            study_id: intent.study_id,
            study_fingerprint: study_fingerprint_from_digest(intent.authored_study_digest),
            authored_study_digest: intent.authored_study_digest,
            model: study_model,
            steps: vec![super::PreparedStudyStep {
                authored_execution,
                procedure_step: step.procedure_step.clone(),
            }],
            operation_sequence: intent.operation_sequence,
            solver_mesh_artifact_path: persisted_mesh.artifact_path,
            meshing_evidence_artifact_path: persisted_mesh.evidence_path,
        },
        vec![CompletedStudyStep {
            step_id: result.step_id,
            solver_result_identity,
            run: CompletedStudyStepRun::OriginalFields {
                run: Box::new(run),
                source: Box::new(original_fields),
            },
            resolved_solver_options: AnalysisSolverOptions::StructuralTransient(options),
        }],
        states,
        result.frozen_project_digest,
        preparation,
    )
}
