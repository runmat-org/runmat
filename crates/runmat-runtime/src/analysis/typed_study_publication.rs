//! Publication adapter for completed typed `.fea` studies.
//!
//! Typed compilation and numerical execution own the analysis definition. This module only projects
//! that already-resolved contract into the runtime's shared result admission and artifact store;
//! it must not reconstruct authoring defaults or change a workload.

use runmat_analysis_core::ProcedureStepId;
use runmat_execution::Digest;

#[path = "typed_study_publication/preparation_facts.rs"]
mod preparation_facts;
#[path = "typed_study_publication/preparation_storage.rs"]
mod preparation_storage;
#[path = "typed_study_publication/publication_options.rs"]
mod publication_options;

use super::study_result_manifest::{PersistedStateEvidence, StudyResultManifest};
use publication_options::publication_steps;

use super::structural_study::{
    prepare_structural_study_intent_completion, PreparedStructuralStudyIntent,
};
use super::{
    execution_operation, AnalysisStudyCompletionIdentity, AnalysisStudyRunData,
    StructuralStudyResult,
};
use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};

/// Admit and persist a placement-neutral typed result through the shared runtime publication path.
///
/// The publication identity fences retry/collision behavior only. It is deliberately absent from
/// the compiled definition digest and portable workloads.
pub fn publish_structural_study_result(
    result: &StructuralStudyResult,
    context: OperationContext,
    publication_identity: AnalysisStudyCompletionIdentity,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let publication_context = context.clone();
    let study_id = result.compiled().study_id.clone();
    prepare_structural_study_publication(result, context, publication_identity)?
        .publish_without_function_execution()
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error,
                &publication_context,
                |message| typed_publication_error(&publication_context, &study_id, message),
            )
        })
}

pub fn prepare_structural_study_publication(
    result: &StructuralStudyResult,
    context: OperationContext,
    publication_identity: AnalysisStudyCompletionIdentity,
) -> Result<crate::analysis::PreparedStudyPublication, OperationErrorEnvelope> {
    let maximum =
        crate::analysis::configuration::publication_staging_budget().map_err(|error| {
            typed_publication_error(&context, &result.compiled().study_id, error.to_string())
        })?;
    let (original_bytes, copy_bytes) =
        preparation_storage::inventory(result, maximum).map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error,
                &context,
                |message| typed_publication_error(&context, &result.compiled().study_id, message),
            )
        })?;
    let (mut preparation, completion_copies) =
        crate::analysis::study_publication::preparation::PublicationPreparation::capture_with_completion_storage(
            result.definition_digest().map_err(|message| {
                typed_publication_error(&context, &result.compiled().study_id, message)
            })?,
            original_bytes,
            copy_bytes,
            maximum,
        )
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error.context(&preparation_facts::describe(result, maximum)),
                &context,
                |message| typed_publication_error(&context, &result.compiled().study_id, message),
            )
        })?;
    preparation
        .admit_structural_sources(result.numerical())
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error,
                &context,
                |message| typed_publication_error(&context, &result.compiled().study_id, message),
            )
        })?;
    let compiled = result.compiled();
    let study_definition_digest = result
        .definition_digest()
        .map_err(|message| typed_publication_error(&context, &compiled.study_id, message))?;
    let steps = publication_steps(result.workloads(), &compiled.model.sequence.steps)
        .map_err(|message| typed_publication_error(&context, &compiled.study_id, message))?;
    let operation_sequence = steps
        .iter()
        .map(|step| {
            execution_operation(&step.authored_execution, &step.procedure_step)
                .map(|operation| operation.version().to_string())
                .ok_or_else(|| {
                    typed_publication_error(
                        &context,
                        &compiled.study_id,
                        format!(
                            "typed procedure step `{}` has no runtime publication operation",
                            step.procedure_step.step_id.as_str()
                        ),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Authenticate and retain original source objects before materializing derived
    // runs and independent suffix outputs. The catalog remains private until the
    // whole publication commits; source replay does not overlap those outputs.
    let numerical_evidence = StructuralStudyNumericalEvidence::from_result(
        result.numerical(),
        study_definition_digest,
        &preparation.staging,
    )
    .map_err(|error| {
        crate::analysis::prepared_study_publication::publication_failure(
            error,
            &context,
            |message| typed_publication_error(&context, &compiled.study_id, message),
        )
    })?;
    let mut intent = PreparedStructuralStudyIntent {
        study_id: compiled.study_id.clone(),
        geometry_source_digest: Digest::from_bytes(*result.mesh().geometry.source_digest.bytes()),
        geometry_unit_evidence: result.mesh().geometry.source_unit_evidence.clone(),
        meshing_request: Some(result.mesh().resolved_request.clone()),
        meshing_contacts: super::meshing::meshing_contact_definitions_for_model(&compiled.model)
            .collect(),
        context,
        authored_study_digest: study_definition_digest,
        intent_digest: Digest::from_bytes([0; 32]),
        model: compiled.model.clone(),
        steps,
        workloads: result.workloads().clone(),
        operation_sequence,
    };
    intent.intent_digest = intent
        .semantic_digest()
        .map_err(|message| typed_publication_error(&intent.context, &intent.study_id, message))?;
    let mut published = prepare_structural_study_intent_completion(
        intent,
        crate::meshing::MeshingExecutionOutput {
            artifact: result.mesh().clone(),
            evidence: result.meshing_evidence().clone(),
        },
        result.numerical().clone(),
        publication_identity,
        preparation,
    )?;
    let published_context = OperationContext::new(
        published.published.trace_id.clone(),
        published.published.request_id.clone(),
    );
    let invalid =
        |message| typed_publication_error(&published_context, &compiled.study_id, message);
    // Family completion consumed and dropped its owned mesh, model, workloads,
    // and numerical result. Its staged output remains charged independently.
    published
        .staging
        .release_owned_copies(completion_copies)
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error,
                &published_context,
                invalid,
            )
        })?;
    let result_manifest = StudyResultManifest::new(
        result
            .compiled()
            .definition()
            .map_err(|error| invalid(error.to_string()))?,
        result
            .workloads()
            .canonical_digest()
            .map_err(|error| invalid(error.to_string()))?,
        result.workloads(),
        result.mesh(),
        result.meshing_evidence(),
        numerical_evidence,
        &mut published,
    )
    .map_err(|error| {
        crate::analysis::prepared_study_publication::publication_failure(
            error,
            &published_context,
            invalid,
        )
    })?;
    crate::analysis::PreparedStudyPublication::new(published, result_manifest).map_err(|error| {
        crate::analysis::prepared_study_publication::publication_failure(
            error,
            &published_context,
            invalid,
        )
    })
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum StructuralStudyNumericalEvidence {
    ThermalPrestress {
        root: runmat_execution::value::ValueRef,
        catalog: runmat_execution::value::PublishedObjectCatalog,
        object_artifact_paths: Vec<String>,
        structural: Box<StructuralStudyNumericalEvidence>,
    },
    LinearSequence {
        frozen_project_digest: Option<Digest>,
        steps: Vec<StructuralLinearStepEvidence>,
        states: Vec<PersistedStateEvidence>,
    },
    StructuralModal {
        frozen_project_digest: Option<Digest>,
        step_id: ProcedureStepId,
        numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
    },
    StructuralModalTransient {
        frozen_project_digest: Option<Digest>,
        modal_step_id: ProcedureStepId,
        transient_step_id: ProcedureStepId,
        modal_numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
        transient_numerical_receipt:
            runmat_analysis_fea::execution::StructuralTransientNumericalReceipt,
        basis_state: PersistedStateEvidence,
    },
    StructuralModalFrequency {
        frozen_project_digest: Option<Digest>,
        modal_step_id: ProcedureStepId,
        frequency_step_id: ProcedureStepId,
        modal_numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
        basis_state: PersistedStateEvidence,
        /// The publication itself is persisted once with the frequency run record.
        frequency_publication_digest: Digest,
    },
    StructuralModalResponseSpectrum {
        frozen_project_digest: Option<Digest>,
        modal_step_id: ProcedureStepId,
        response_step_id: ProcedureStepId,
        modal_numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
        response_numerical_receipt:
            runmat_analysis_fea::execution::StructuralModalResponseSpectrumNumericalReceipt,
        response_accuracy: runmat_analysis_fea::results::accuracy::AccuracyArtifact,
        response_resource_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
        response_publication_baseline_usage:
            runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
        response_publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
        basis_state: PersistedStateEvidence,
        envelope_state: PersistedStateEvidence,
    },
    StructuralRandomResponse {
        frozen_project_digest: Option<Digest>,
        modal_step_id: ProcedureStepId,
        response_step_id: ProcedureStepId,
        modal_numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
        response_numerical_receipt:
            runmat_analysis_fea::execution::StructuralRandomResponseNumericalReceipt,
        response_accuracy: runmat_analysis_fea::results::accuracy::AccuracyArtifact,
        response_resource_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
        response_publication_baseline_usage:
            runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
        response_publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
        basis_state: PersistedStateEvidence,
        statistics_state: PersistedStateEvidence,
    },
    StructuralBuckling {
        frozen_project_digest: Option<Digest>,
        preload_step_id: ProcedureStepId,
        buckling_step_id: ProcedureStepId,
        preload_solver_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
        preload_numerical_receipt:
            runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
        buckling_numerical_receipt:
            runmat_analysis_fea::execution::StructuralBucklingNumericalReceipt,
        prestress_state: PersistedStateEvidence,
    },
    StructuralPrestressedModal {
        frozen_project_digest: Option<Digest>,
        preload_step_id: ProcedureStepId,
        modal_step_id: ProcedureStepId,
        preload_solver_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
        preload_numerical_receipt:
            runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
        modal_numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
        prestress_state: PersistedStateEvidence,
    },
    StructuralPrestressedFrequency {
        frozen_project_digest: Option<Digest>,
        preload_step_id: ProcedureStepId,
        frequency_step_id: ProcedureStepId,
        preload_solver_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
        preload_numerical_receipt:
            runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
        prestress_state: PersistedStateEvidence,
        modal_reduction: Option<StructuralModalReductionEvidence>,
        /// The publication itself is persisted once with the frequency run record.
        frequency_publication_digest: Digest,
    },
    StructuralTransient {
        frozen_project_digest: Option<Digest>,
        step_id: ProcedureStepId,
        numerical_receipt: runmat_analysis_fea::execution::StructuralTransientNumericalReceipt,
    },
    StructuralFrequency {
        frozen_project_digest: Option<Digest>,
        step_id: ProcedureStepId,
        /// The publication itself is persisted once with the frequency run record.
        publication_digest: Digest,
    },
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralLinearStepEvidence {
    step_id: ProcedureStepId,
    solver_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    numerical_receipt: runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralModalReductionEvidence {
    modal_step_id: ProcedureStepId,
    modal_numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
    basis_state: PersistedStateEvidence,
}

impl StructuralStudyNumericalEvidence {
    fn from_result(
        result: &super::StructuralStudyNumericalResult,
        study_digest: Digest,
        staging: &crate::analysis::storage::AnalysisPublicationBuilder,
    ) -> Result<Self, crate::analysis::storage::AnalysisPublicationError> {
        Ok(match result {
            super::StructuralStudyNumericalResult::ThermalPrestress {
                thermal,
                thermal_source,
                structural,
            } => Self::ThermalPrestress {
                object_artifact_paths: thermal_source.persist(thermal, study_digest, staging)?,
                root: thermal.root.clone(),
                catalog: thermal.catalog.clone(),
                structural: Box::new(Self::from_result(structural, study_digest, staging)?),
            },
            super::StructuralStudyNumericalResult::LinearSequence(result) => Self::LinearSequence {
                frozen_project_digest: result.frozen_project_digest,
                steps: result
                    .steps
                    .iter()
                    .map(|step| StructuralLinearStepEvidence {
                        step_id: step.step_id.clone(),
                        solver_result_identity: step.solver_result_identity,
                        numerical_receipt: step.numerical_receipt.clone(),
                    })
                    .collect(),
                states: result
                    .states
                    .iter()
                    .map(|state| PersistedStateEvidence::persist(state, study_digest, staging))
                    .collect::<Result<_, _>>()?,
            },
            super::StructuralStudyNumericalResult::StructuralModal(result) => {
                Self::StructuralModal {
                    frozen_project_digest: result.frozen_project_digest,
                    step_id: result.step_id.clone(),
                    numerical_receipt: result.numerical_receipt.clone(),
                }
            }
            super::StructuralStudyNumericalResult::StructuralModalTransient(result) => {
                Self::StructuralModalTransient {
                    frozen_project_digest: result.frozen_project_digest,
                    modal_step_id: result.modal_step_id.clone(),
                    transient_step_id: result.transient_step_id.clone(),
                    modal_numerical_receipt: result.modal_numerical_receipt.clone(),
                    transient_numerical_receipt: result.transient_numerical_receipt.clone(),
                    basis_state: PersistedStateEvidence::persist(
                        &result.basis_state,
                        study_digest,
                        staging,
                    )?,
                }
            }
            super::StructuralStudyNumericalResult::StructuralModalFrequency(result) => {
                Self::StructuralModalFrequency {
                    frozen_project_digest: result.frozen_project_digest,
                    modal_step_id: result.modal_step_id.clone(),
                    frequency_step_id: result.frequency_step_id.clone(),
                    modal_numerical_receipt: result.modal_numerical_receipt.clone(),
                    basis_state: PersistedStateEvidence::persist(
                        &result.basis_state,
                        study_digest,
                        staging,
                    )?,
                    frequency_publication_digest: frequency_publication_digest(
                        &result.frequency.publication,
                    )?,
                }
            }
            super::StructuralStudyNumericalResult::StructuralModalResponseSpectrum(result) => {
                Self::StructuralModalResponseSpectrum {
                    frozen_project_digest: result.frozen_project_digest,
                    modal_step_id: result.modal_step_id.clone(),
                    response_step_id: result.response_step_id.clone(),
                    modal_numerical_receipt: result.modal_numerical_receipt.clone(),
                    response_numerical_receipt: result.response_numerical_receipt.clone(),
                    response_accuracy: result.response_accuracy.clone(),
                    response_resource_usage: result.response_resource_usage.clone(),
                    response_publication_baseline_usage: result
                        .response_publication_baseline_usage
                        .clone(),
                    response_publication: result.response_publication.clone(),
                    basis_state: PersistedStateEvidence::persist(
                        &result.basis_state,
                        study_digest,
                        staging,
                    )?,
                    envelope_state: PersistedStateEvidence::persist(
                        &result.envelope_state,
                        study_digest,
                        staging,
                    )?,
                }
            }
            super::StructuralStudyNumericalResult::StructuralRandomResponse(result) => {
                Self::StructuralRandomResponse {
                    frozen_project_digest: result.frozen_project_digest,
                    modal_step_id: result.modal_step_id.clone(),
                    response_step_id: result.response_step_id.clone(),
                    modal_numerical_receipt: result.modal_numerical_receipt.clone(),
                    response_numerical_receipt: result.response_numerical_receipt.clone(),
                    response_accuracy: result.response_accuracy.clone(),
                    response_resource_usage: result.response_resource_usage.clone(),
                    response_publication_baseline_usage: result
                        .response_publication_baseline_usage
                        .clone(),
                    response_publication: result.response_publication.clone(),
                    basis_state: PersistedStateEvidence::persist(
                        &result.basis_state,
                        study_digest,
                        staging,
                    )?,
                    statistics_state: PersistedStateEvidence::persist(
                        &result.statistics_state,
                        study_digest,
                        staging,
                    )?,
                }
            }
            super::StructuralStudyNumericalResult::StructuralBuckling(result) => {
                Self::StructuralBuckling {
                    frozen_project_digest: result.frozen_project_digest,
                    preload_step_id: result.preload_step_id.clone(),
                    buckling_step_id: result.buckling_step_id.clone(),
                    preload_solver_result_identity: result.preload_solver_result_identity,
                    preload_numerical_receipt: result.preload_numerical_receipt.clone(),
                    buckling_numerical_receipt: result.buckling_numerical_receipt.clone(),
                    prestress_state: PersistedStateEvidence::persist(
                        &result.prestress_state,
                        study_digest,
                        staging,
                    )?,
                }
            }
            super::StructuralStudyNumericalResult::StructuralPrestressedModal(result) => {
                Self::StructuralPrestressedModal {
                    frozen_project_digest: result.frozen_project_digest,
                    preload_step_id: result.preload_step_id.clone(),
                    modal_step_id: result.modal_step_id.clone(),
                    preload_solver_result_identity: result.preload_solver_result_identity,
                    preload_numerical_receipt: result.preload_numerical_receipt.clone(),
                    modal_numerical_receipt: result.modal_numerical_receipt.clone(),
                    prestress_state: PersistedStateEvidence::persist(
                        &result.prestress_state,
                        study_digest,
                        staging,
                    )?,
                }
            }
            super::StructuralStudyNumericalResult::StructuralPrestressedFrequency(result) => {
                let modal_reduction = match &result.reduction {
                    super::StructuralStudyPrestressedFrequencyReductionResult::Direct => None,
                    super::StructuralStudyPrestressedFrequencyReductionResult::ModalReduced(
                        reduction,
                    ) => Some(StructuralModalReductionEvidence {
                        modal_step_id: reduction.modal_step_id.clone(),
                        modal_numerical_receipt: reduction.modal_numerical_receipt.clone(),
                        basis_state: PersistedStateEvidence::persist(
                            &reduction.basis_state,
                            study_digest,
                            staging,
                        )?,
                    }),
                };
                Self::StructuralPrestressedFrequency {
                    frozen_project_digest: result.frozen_project_digest,
                    preload_step_id: result.preload_step_id.clone(),
                    frequency_step_id: result.frequency_step_id.clone(),
                    preload_solver_result_identity: result.preload_solver_result_identity,
                    preload_numerical_receipt: result.preload_numerical_receipt.clone(),
                    prestress_state: PersistedStateEvidence::persist(
                        &result.prestress_state,
                        study_digest,
                        staging,
                    )?,
                    modal_reduction,
                    frequency_publication_digest: frequency_publication_digest(
                        &result.frequency.publication,
                    )?,
                }
            }
            super::StructuralStudyNumericalResult::StructuralTransient(result) => {
                Self::StructuralTransient {
                    frozen_project_digest: result.frozen_project_digest,
                    step_id: result.step_id.clone(),
                    numerical_receipt: result.numerical_receipt.clone(),
                }
            }
            super::StructuralStudyNumericalResult::StructuralFrequency(result) => {
                Self::StructuralFrequency {
                    frozen_project_digest: result.frozen_project_digest,
                    step_id: result.step_id.clone(),
                    publication_digest: frequency_publication_digest(&result.publication)?,
                }
            }
        })
    }
}

fn frequency_publication_digest(
    publication: &runmat_analysis_fea::execution::StructuralFrequencyPublicationManifest,
) -> Result<Digest, crate::analysis::storage::AnalysisPublicationError> {
    let bytes = publication
        .canonical_bytes()
        .map_err(|error| error.to_string())?;
    Ok(Digest::sha256(bytes))
}

fn typed_publication_error(
    context: &OperationContext,
    study_id: &str,
    message: impl Into<String>,
) -> OperationErrorEnvelope {
    crate::operations::operation_error(
        super::ANALYSIS_RUN_STUDY_OPERATION,
        super::ANALYSIS_RUN_STUDY_OP_VERSION,
        context,
        crate::operations::OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY.INVALID_TYPED_RESULT",
            error_type: crate::operations::OperationErrorType::Contract,
            retryable: false,
            severity: crate::operations::OperationErrorSeverity::Fatal,
        },
        message,
        std::collections::BTreeMap::from([("study_id".to_string(), study_id.to_string())]),
    )
}
