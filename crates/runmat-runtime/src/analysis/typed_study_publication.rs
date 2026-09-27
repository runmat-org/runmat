//! Publication adapter for completed typed `.fea` studies.
//!
//! Typed compilation and numerical execution own the analysis definition. This module only projects
//! that already-resolved contract into the runtime's shared result admission and artifact store;
//! it must not reconstruct authoring defaults or change a workload.

use runmat_analysis_core::ProcedureStepId;
use runmat_analysis_fea::execution::StructuralStudyWorkloads;
use runmat_execution::Digest;

mod publication_options;

use publication_options::publication_steps;

use super::structural_study::{
    analysis_complete_structural_study_intent_op, PreparedStructuralStudyIntent,
};
use super::{
    execution_operation, AnalysisStudyRunData, StructuralStudyCompletionIdentity,
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
    publication_identity: StructuralStudyCompletionIdentity,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
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
    let mut intent = PreparedStructuralStudyIntent {
        study_id: compiled.study_id.clone(),
        geometry_source_digest: Digest::from_bytes(*result.mesh().geometry.source_digest.bytes()),
        geometry_units: compiled.model.units,
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
    let mut published = analysis_complete_structural_study_intent_op(
        intent,
        crate::meshing::MeshingExecutionOutput {
            artifact: result.mesh().clone(),
            evidence: result.meshing_evidence().clone(),
            region_aliases: Default::default(),
        },
        result.numerical().clone(),
        publication_identity,
    )?;
    let result_manifest = StructuralStudyResultManifest::from_result(
        result,
        study_definition_digest,
        published.data.clone(),
    )
    .map_err(|message| {
        typed_publication_error(
            &OperationContext::new(published.trace_id.clone(), published.request_id.clone()),
            &compiled.study_id,
            message,
        )
    })?;
    let published_context =
        OperationContext::new(published.trace_id.clone(), published.request_id.clone());
    let manifest_value = serde_json::to_value(&result_manifest).map_err(|error| {
        typed_publication_error(
            &published_context,
            &compiled.study_id,
            format!("failed to encode typed study result manifest: {error}"),
        )
    })?;
    let manifest_path = super::structural_study::persist_content_addressed_study_evidence(
        &study_definition_digest.to_string(),
        "study-result-manifest",
        manifest_value,
    )
    .map_err(|message| typed_publication_error(&published_context, &compiled.study_id, message))?;
    published.data.result_manifest_artifact_path = Some(manifest_path);
    Ok(published)
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralStudyResultManifest {
    schema_version: u16,
    study_definition_digest: Digest,
    compiled: runmat_analysis_fea::study::CompiledAnalysisStudy,
    workload_digest: Digest,
    workloads: StructuralStudyWorkloads,
    solver_mesh_digest: runmat_meshing_core::StableDigest,
    meshing_evidence: runmat_meshing_core::MeshingEvidence,
    numerical_evidence: StructuralStudyNumericalEvidence,
    public_run: AnalysisStudyRunData,
}

impl StructuralStudyResultManifest {
    fn from_result(
        result: &StructuralStudyResult,
        study_definition_digest: Digest,
        public_run: AnalysisStudyRunData,
    ) -> Result<Self, String> {
        Ok(Self {
            schema_version: 1,
            study_definition_digest,
            compiled: result.compiled().clone(),
            workload_digest: result
                .workloads()
                .canonical_digest()
                .map_err(|error| error.to_string())?,
            workloads: result.workloads().clone(),
            solver_mesh_digest: result.mesh().canonical_digest,
            meshing_evidence: result.meshing_evidence().clone(),
            numerical_evidence: StructuralStudyNumericalEvidence::from_result(
                result.numerical(),
                study_definition_digest,
            )?,
            public_run,
        })
    }
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
        states: Vec<runmat_analysis_fea::execution::CanonicalSemanticState>,
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
        basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
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
        basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
        envelope_state: runmat_analysis_fea::execution::CanonicalSemanticState,
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
        basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
        statistics_state: runmat_analysis_fea::execution::CanonicalSemanticState,
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
        prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    },
    StructuralPrestressedModal {
        frozen_project_digest: Option<Digest>,
        preload_step_id: ProcedureStepId,
        modal_step_id: ProcedureStepId,
        preload_solver_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
        preload_numerical_receipt:
            runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
        modal_numerical_receipt: runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
        prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    },
    StructuralPrestressedFrequency {
        frozen_project_digest: Option<Digest>,
        preload_step_id: ProcedureStepId,
        frequency_step_id: ProcedureStepId,
        preload_solver_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
        preload_numerical_receipt:
            runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
        prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
        modal_reduction: Option<StructuralModalReductionEvidence>,
        frequency_publication:
            Box<runmat_analysis_fea::execution::StructuralFrequencyPublicationManifest>,
    },
    StructuralTransient {
        frozen_project_digest: Option<Digest>,
        step_id: ProcedureStepId,
        numerical_receipt: runmat_analysis_fea::execution::StructuralTransientNumericalReceipt,
    },
    StructuralFrequency {
        frozen_project_digest: Option<Digest>,
        step_id: ProcedureStepId,
        publication: runmat_analysis_fea::execution::StructuralFrequencyPublicationManifest,
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
    basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
}

impl StructuralStudyNumericalEvidence {
    fn from_result(
        result: &super::StructuralStudyNumericalResult,
        study_digest: Digest,
    ) -> Result<Self, String> {
        Ok(match result {
            super::StructuralStudyNumericalResult::ThermalPrestress {
                thermal,
                thermal_source,
                structural,
            } => Self::ThermalPrestress {
                object_artifact_paths: thermal_source.persist(thermal, study_digest)?,
                root: thermal.root.clone(),
                catalog: thermal.catalog.clone(),
                structural: Box::new(Self::from_result(structural, study_digest)?),
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
                states: result.states.clone(),
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
                    basis_state: result.basis_state.clone(),
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
                    basis_state: result.basis_state.clone(),
                    envelope_state: result.envelope_state.clone(),
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
                    basis_state: result.basis_state.clone(),
                    statistics_state: result.statistics_state.clone(),
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
                    prestress_state: result.prestress_state.clone(),
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
                    prestress_state: result.prestress_state.clone(),
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
                        basis_state: reduction.basis_state.clone(),
                    }),
                };
                Self::StructuralPrestressedFrequency {
                    frozen_project_digest: result.frozen_project_digest,
                    preload_step_id: result.preload_step_id.clone(),
                    frequency_step_id: result.frequency_step_id.clone(),
                    preload_solver_result_identity: result.preload_solver_result_identity,
                    preload_numerical_receipt: result.preload_numerical_receipt.clone(),
                    prestress_state: result.prestress_state.clone(),
                    modal_reduction,
                    frequency_publication: Box::new(result.frequency.publication.clone()),
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
                    publication: result.publication.clone(),
                }
            }
        })
    }
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
