use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_execution::{
    CompletedStructuralSequence, StructuralSequencePreparation, StructuralSequenceResultClosure,
    StructuralStudyWorkloads,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, StructuralBucklingNumericalReceipt,
    StructuralBucklingSolverResult, StructuralBucklingWorkload,
    StructuralFrequencyNumericalReceipt, StructuralFrequencySolverResult,
    StructuralFrequencyWorkload, StructuralLinearWorkload, StructuralModalNumericalReceipt,
    StructuralModalSolverResult, StructuralModalWorkload, StructuralTransientNumericalReceipt,
    StructuralTransientSolverResult, StructuralTransientWorkload, StudyIdentity,
};
use runmat_canonical_codec::{decode_contract, encode_contract};
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, LogicalObject, ObjectNamespace, ProgramExecutionObject,
    ProgramExecutionRequest, ProgramExecutionResponse, PublishedObjectCatalog,
};
use runmat_meshing_core::{CanonicalMeshingContract, MeshingRequest};
use runmat_meshing_execution::ExactMeshingDagRunResult;
use serde::{Deserialize, Serialize};

use super::contract::{
    structural_workload_inventory_digest, validate_analysis_accelerator_requests,
    DecodedExactStructuralStudyRoot,
};
use super::region_resolution::TopologyRegionResolution;
use super::{
    protocol, read, CONTRACT_LIMITS, PUBLICATION_DOMAIN, PUBLICATION_MEDIA_TYPE,
    PUBLICATION_PREFIX, PUBLICATION_VALUE_SCHEMA,
};
use crate::{CompletedStructuralTransientSequence, NativeExecutionResult};

mod buckling_import;
mod frequency_closure_import;
mod frequency_import;
mod inventory;
mod modal_import;
mod packaging;
mod prestressed_frequency_import;
mod prestressed_modal_frequency_import;
mod prestressed_modal_import;
mod random_response_import;
mod response_spectrum_import;
mod result_import;
mod source_result_import;
mod specialized_publication_import;

use inventory::*;
pub(crate) use packaging::package_exact_structural_study_result;
pub use result_import::import_exact_structural_study_result;
pub use source_result_import::import_structural_study_result;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactStructuralStudyPublication {
    schema_version: u16,
    program_revision: ProgramRevision,
    authored_study_digest: Digest,
    compiled_study: Option<runmat_analysis_fea::study::CompiledAnalysisStudy>,
    compiled_geometry_output: Option<runmat_analysis_execution::GeometryMaterializationOutput>,
    unresolved_workloads_digest: Digest,
    meshing_request_digest: Digest,
    meshing_root: ValueRef,
    meshing_objects: Vec<ValueRef>,
    region_resolution_root: ValueRef,
    region_resolution_objects: Vec<ValueRef>,
    input_roots: Vec<ValueRef>,
    input_objects: Vec<ValueRef>,
    analysis: ExactStructuralStudyAnalysisClosure,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ExactStructuralStudyAnalysisClosure {
    LinearSequence {
        closures: Vec<StructuralSequenceResultClosure>,
        objects: Vec<ValueRef>,
    },
    StructuralTransient {
        result_root: Box<ValueRef>,
        receipt_root: Box<ValueRef>,
        objects: Vec<ValueRef>,
    },
    StructuralModal {
        result_root: Box<ValueRef>,
        receipt_root: Box<ValueRef>,
        objects: Vec<ValueRef>,
    },
    StructuralModalTransient {
        modal_result_root: Box<ValueRef>,
        modal_receipt_root: Box<ValueRef>,
        basis_state_root: Box<ValueRef>,
        transient_result_root: Box<ValueRef>,
        transient_receipt_root: Box<ValueRef>,
        objects: Vec<ValueRef>,
    },
    StructuralModalResponseSpectrum {
        modal_result_root: Box<ValueRef>,
        modal_receipt_root: Box<ValueRef>,
        basis_state_root: Box<ValueRef>,
        response_result_root: Box<ValueRef>,
        response_receipt_root: Box<ValueRef>,
        response_accuracy_root: Box<ValueRef>,
        response_resource_usage_root: Box<ValueRef>,
        response_publication_root: Box<ValueRef>,
        field_recovery_root: Box<ValueRef>,
        envelope_state_root: Box<ValueRef>,
        objects: Vec<ValueRef>,
    },
    StructuralRandomResponse {
        modal_result_root: Box<ValueRef>,
        modal_receipt_root: Box<ValueRef>,
        basis_state_root: Box<ValueRef>,
        response_result_root: Box<ValueRef>,
        response_receipt_root: Box<ValueRef>,
        response_accuracy_root: Box<ValueRef>,
        response_resource_usage_root: Box<ValueRef>,
        response_publication_root: Box<ValueRef>,
        statistics_state_root: Box<ValueRef>,
        objects: Vec<ValueRef>,
    },
    StructuralBuckling {
        preload_result_root: Box<ValueRef>,
        preload_receipt_root: Box<ValueRef>,
        prestress_state_root: Box<ValueRef>,
        buckling_result_root: Box<ValueRef>,
        buckling_receipt_root: Box<ValueRef>,
        objects: Vec<ValueRef>,
    },
    StructuralPrestressedModal {
        preload_result_root: Box<ValueRef>,
        preload_receipt_root: Box<ValueRef>,
        prestress_state_root: Box<ValueRef>,
        modal_result_root: Box<ValueRef>,
        modal_receipt_root: Box<ValueRef>,
        objects: Vec<ValueRef>,
    },
    StructuralPrestressedFrequency {
        preload_result_root: Box<ValueRef>,
        preload_receipt_root: Box<ValueRef>,
        prestress_state_root: Box<ValueRef>,
        frequency: Box<ExactStructuralFrequencyClosure>,
        objects: Vec<ValueRef>,
    },
    StructuralPrestressedModalFrequency {
        preload_result_root: Box<ValueRef>,
        preload_receipt_root: Box<ValueRef>,
        prestress_state_root: Box<ValueRef>,
        modal_result_root: Box<ValueRef>,
        modal_receipt_root: Box<ValueRef>,
        basis_state_root: Box<ValueRef>,
        frequency: Box<ExactStructuralFrequencyClosure>,
        objects: Vec<ValueRef>,
    },
    StructuralFrequency {
        frequency: Box<ExactStructuralFrequencyClosure>,
        objects: Vec<ValueRef>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactStructuralFrequencyClosure {
    result_root: Box<ValueRef>,
    receipt_root: Box<ValueRef>,
    publication: Box<runmat_analysis_fea::execution::StructuralFrequencyPublicationManifest>,
    entries: Vec<ExactStructuralFrequencyEntryClosure>,
    histories: Vec<ExactStructuralFrequencyHistoryClosure>,
    resource_usage_root: Box<ValueRef>,
    objects: Vec<ValueRef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactStructuralFrequencyEntryClosure {
    entry: runmat_analysis_fea::frequency::FrequencyResponsePlanEntry,
    source_entry: runmat_analysis_fea::execution::StructuralFrequencyEntryResultManifest,
    physical_sample: runmat_analysis_fea::execution::StructuralFrequencyPhysicalSampleIdentity,
    sample: runmat_analysis_fea::results::field::RecoveredFieldSample,
    fields_root: ValueRef,
    observations_root: Option<ValueRef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactStructuralFrequencyHistoryClosure {
    observation_id: runmat_analysis_core::ObservationId,
    checkpoint_root: ValueRef,
}

impl ExactStructuralStudyAnalysisClosure {
    fn objects(&self) -> &[ValueRef] {
        match self {
            Self::LinearSequence { objects, .. }
            | Self::StructuralTransient { objects, .. }
            | Self::StructuralModal { objects, .. }
            | Self::StructuralModalTransient { objects, .. }
            | Self::StructuralModalResponseSpectrum { objects, .. }
            | Self::StructuralRandomResponse { objects, .. }
            | Self::StructuralBuckling { objects, .. }
            | Self::StructuralPrestressedModal { objects, .. }
            | Self::StructuralPrestressedFrequency { objects, .. }
            | Self::StructuralPrestressedModalFrequency { objects, .. }
            | Self::StructuralFrequency { objects, .. } => objects,
        }
    }

    fn validate(&self) -> NativeExecutionResult<()> {
        match self {
            Self::LinearSequence {
                closures, objects, ..
            } => {
                let mut nodes = BTreeSet::new();
                for closure in closures {
                    if !nodes.insert(closure.node.clone()) {
                        return Err(protocol(
                            "study publication contains a duplicate sequence node",
                        ));
                    }
                    validate_externalized_response(&closure.response, objects)?;
                }
                let mut closure_objects = Vec::new();
                for closure in closures {
                    closure_objects.extend(response_objects(&closure.response)?);
                }
                let closure_objects = unique_references(closure_objects)?;
                if reference_inventory(&closure_objects)? != reference_inventory(objects)? {
                    return Err(protocol(
                        "analysis inventory contains unreferenced or missing sequence objects",
                    ));
                }
            }
            Self::StructuralTransient {
                result_root,
                receipt_root,
                objects,
            } => {
                if result_root == receipt_root
                    || !objects.contains(result_root)
                    || !objects.contains(receipt_root)
                {
                    return Err(protocol(
                        "structural numerical closure omits or aliases a declared root",
                    ));
                }
                reference_inventory(objects)?;
            }
            Self::StructuralModal {
                result_root,
                receipt_root,
                objects,
            } => {
                validate_declared_roots(&[result_root, receipt_root], objects)?;
            }
            Self::StructuralModalTransient {
                modal_result_root,
                modal_receipt_root,
                basis_state_root,
                transient_result_root,
                transient_receipt_root,
                objects,
            } => {
                validate_declared_roots(
                    &[
                        modal_result_root,
                        modal_receipt_root,
                        basis_state_root,
                        transient_result_root,
                        transient_receipt_root,
                    ],
                    objects,
                )?;
            }
            Self::StructuralModalResponseSpectrum {
                modal_result_root,
                modal_receipt_root,
                basis_state_root,
                response_result_root,
                response_receipt_root,
                response_accuracy_root,
                response_resource_usage_root,
                response_publication_root,
                field_recovery_root,
                envelope_state_root,
                objects,
            } => {
                validate_declared_roots(
                    &[
                        modal_result_root,
                        modal_receipt_root,
                        basis_state_root,
                        response_result_root,
                        response_receipt_root,
                        response_accuracy_root,
                        response_resource_usage_root,
                        response_publication_root,
                        field_recovery_root,
                        envelope_state_root,
                    ],
                    objects,
                )?;
            }
            Self::StructuralRandomResponse {
                modal_result_root,
                modal_receipt_root,
                basis_state_root,
                response_result_root,
                response_receipt_root,
                response_accuracy_root,
                response_resource_usage_root,
                response_publication_root,
                statistics_state_root,
                objects,
            } => {
                validate_declared_roots(
                    &[
                        modal_result_root,
                        modal_receipt_root,
                        basis_state_root,
                        response_result_root,
                        response_receipt_root,
                        response_accuracy_root,
                        response_resource_usage_root,
                        response_publication_root,
                        statistics_state_root,
                    ],
                    objects,
                )?;
            }
            Self::StructuralBuckling {
                preload_result_root,
                preload_receipt_root,
                prestress_state_root,
                buckling_result_root,
                buckling_receipt_root,
                objects,
            } => validate_declared_roots(
                &[
                    preload_result_root,
                    preload_receipt_root,
                    prestress_state_root,
                    buckling_result_root,
                    buckling_receipt_root,
                ],
                objects,
            )?,
            Self::StructuralPrestressedModal {
                preload_result_root,
                preload_receipt_root,
                prestress_state_root,
                modal_result_root,
                modal_receipt_root,
                objects,
            } => validate_declared_roots(
                &[
                    preload_result_root,
                    preload_receipt_root,
                    prestress_state_root,
                    modal_result_root,
                    modal_receipt_root,
                ],
                objects,
            )?,
            Self::StructuralPrestressedFrequency {
                preload_result_root,
                preload_receipt_root,
                prestress_state_root,
                frequency,
                objects,
            } => {
                validate_frequency_closure(frequency, objects)?;
                validate_declared_roots(
                    &[
                        preload_result_root,
                        preload_receipt_root,
                        prestress_state_root,
                        &frequency.result_root,
                        &frequency.receipt_root,
                        &frequency.resource_usage_root,
                    ],
                    objects,
                )?;
            }
            Self::StructuralPrestressedModalFrequency {
                preload_result_root,
                preload_receipt_root,
                prestress_state_root,
                modal_result_root,
                modal_receipt_root,
                basis_state_root,
                frequency,
                objects,
            } => {
                validate_frequency_closure(frequency, objects)?;
                validate_declared_roots(
                    &[
                        preload_result_root,
                        preload_receipt_root,
                        prestress_state_root,
                        modal_result_root,
                        modal_receipt_root,
                        basis_state_root,
                        &frequency.result_root,
                        &frequency.receipt_root,
                        &frequency.resource_usage_root,
                    ],
                    objects,
                )?;
            }
            Self::StructuralFrequency { frequency, objects } => {
                validate_frequency_closure(frequency, objects)?
            }
        }
        Ok(())
    }
}

fn validate_frequency_closure(
    frequency: &ExactStructuralFrequencyClosure,
    objects: &[ValueRef],
) -> NativeExecutionResult<()> {
    frequency.publication.validate().map_err(protocol)?;
    if frequency
        .objects
        .iter()
        .any(|reference| !objects.contains(reference))
    {
        return Err(protocol(
            "structural frequency object closure escapes its enclosing result",
        ));
    }
    let objects = frequency.objects.as_slice();
    if frequency.result_root == frequency.receipt_root
        || frequency.entries.is_empty()
        || !objects.contains(&frequency.result_root)
        || !objects.contains(&frequency.receipt_root)
        || !objects.contains(&frequency.resource_usage_root)
        || frequency.entries.iter().any(|entry| {
            !objects.contains(&entry.fields_root)
                || entry
                    .observations_root
                    .as_ref()
                    .is_some_and(|root| !objects.contains(root))
        })
        || frequency
            .histories
            .iter()
            .any(|history| !objects.contains(&history.checkpoint_root))
    {
        return Err(protocol(
            "structural frequency closure omits a declared artifact root",
        ));
    }
    let history_ids = frequency
        .histories
        .iter()
        .map(|history| &history.observation_id)
        .collect::<BTreeSet<_>>();
    if history_ids.len() != frequency.histories.len() {
        return Err(protocol(
            "structural frequency closure contains duplicate histories",
        ));
    }
    reference_inventory(objects)?;
    Ok(())
}

impl ExactStructuralStudyPublication {
    fn validate(&self, access: &ArtifactAccess) -> NativeExecutionResult<()> {
        if self.schema_version != 1
            || self.authored_study_digest.bytes() == &[0; 32]
            || self.unresolved_workloads_digest.bytes() == &[0; 32]
            || self.meshing_request_digest.bytes() == &[0; 32]
            || self.meshing_objects.is_empty()
            || self.region_resolution_objects.is_empty()
            || self.analysis.objects().is_empty()
            || self.input_roots.len() > self.input_objects.len()
            || self
                .input_roots
                .iter()
                .any(|root| !self.input_objects.contains(root))
        {
            return Err(protocol("exact structural study publication is invalid"));
        }
        if self.compiled_study.is_some() != self.compiled_geometry_output.is_some() {
            return Err(protocol(
                "compiled study publication has an incomplete source contract",
            ));
        }
        for reference in self
            .meshing_objects
            .iter()
            .chain(&self.region_resolution_objects)
            .chain(&self.input_objects)
            .chain(self.analysis.objects())
        {
            if reference.kind != ValueRefKind::ResultObject
                || reference.authorization_scope != access.authorization_scope
                || reference.encryption_context != access.encryption_context
            {
                return Err(protocol(
                    "study publication object is outside its result authority",
                ));
            }
        }
        for (root, inventory) in [
            (&self.meshing_root, &self.meshing_objects),
            (
                &self.region_resolution_root,
                &self.region_resolution_objects,
            ),
        ] {
            if !inventory.contains(root) {
                return Err(protocol(
                    "study publication closure omits a declared component root",
                ));
            }
            reference_inventory(inventory)?;
        }
        self.analysis.validate()?;
        reference_inventory(&self.input_objects)?;
        Ok(())
    }

    fn canonical_bytes(&self, access: &ArtifactAccess) -> NativeExecutionResult<Vec<u8>> {
        self.validate(access)?;
        encode_contract(
            PUBLICATION_PREFIX,
            PUBLICATION_DOMAIN,
            self,
            CONTRACT_LIMITS,
        )
        .map_err(protocol)
    }

    fn decode(bytes: &[u8], access: &ArtifactAccess) -> NativeExecutionResult<Self> {
        let publication = decode_contract(
            PUBLICATION_PREFIX,
            PUBLICATION_DOMAIN,
            bytes,
            CONTRACT_LIMITS,
        )
        .map_err(protocol)?;
        Self::validate(&publication, access)?;
        Ok(publication)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralStudyResult {
    pub frozen_project_digest: Option<Digest>,
    pub compiled_study: Option<runmat_analysis_fea::study::CompiledAnalysisStudy>,
    pub mesh: ExactMeshingDagRunResult,
    pub region_resolution: TopologyRegionResolution,
    pub resolved_workloads: StructuralStudyWorkloads,
    pub analysis: ExactStructuralStudyAnalysisResult,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExactStructuralStudyAnalysisResult {
    LinearSequence {
        resolved_workloads:
            BTreeMap<runmat_analysis_core::ProcedureStepId, StructuralLinearWorkload>,
        sequence: CompletedStructuralSequence,
    },
    StructuralTransient {
        resolved_workload: Box<StructuralTransientWorkload>,
        result: Box<StructuralTransientSolverResult>,
        numerical_receipt: StructuralTransientNumericalReceipt,
    },
    StructuralModal {
        resolved_workload: Box<StructuralModalWorkload>,
        result: Box<StructuralModalSolverResult>,
        numerical_receipt: StructuralModalNumericalReceipt,
    },
    StructuralModalTransient(Box<ExactStructuralModalTransientAnalysisResult>),
    StructuralModalResponseSpectrum(Box<ExactStructuralModalResponseSpectrumAnalysisResult>),
    StructuralRandomResponse(Box<ExactStructuralRandomResponseAnalysisResult>),
    StructuralBuckling(Box<ExactStructuralBucklingAnalysisResult>),
    StructuralPrestressedModal(Box<ExactStructuralPrestressedModalAnalysisResult>),
    StructuralPrestressedFrequency(Box<ExactStructuralPrestressedFrequencyAnalysisResult>),
    StructuralPrestressedModalFrequency(
        Box<ExactStructuralPrestressedModalFrequencyAnalysisResult>,
    ),
    StructuralFrequency(Box<ExactStructuralFrequencyAnalysisResult>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralModalTransientAnalysisResult {
    pub modal_workload: Box<StructuralModalWorkload>,
    pub transient_workload: Box<StructuralTransientWorkload>,
    pub modal_result: Box<StructuralModalSolverResult>,
    pub modal_numerical_receipt: StructuralModalNumericalReceipt,
    pub basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub transient_result: Box<StructuralTransientSolverResult>,
    pub transient_numerical_receipt: StructuralTransientNumericalReceipt,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralModalResponseSpectrumAnalysisResult {
    pub modal_workload: Box<StructuralModalWorkload>,
    pub response_workload:
        Box<runmat_analysis_fea::execution::StructuralModalResponseSpectrumWorkload>,
    pub modal_result: Box<StructuralModalSolverResult>,
    pub modal_numerical_receipt: StructuralModalNumericalReceipt,
    pub basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub response_result:
        Box<runmat_analysis_fea::execution::StructuralModalResponseSpectrumSolverResult>,
    pub response_numerical_receipt:
        runmat_analysis_fea::execution::StructuralModalResponseSpectrumNumericalReceipt,
    pub response_accuracy: runmat_analysis_fea::results::accuracy::AccuracyArtifact,
    pub response_resource_usage: runmat_analysis_execution::PreparedResourceUsage,
    pub response_publication: runmat_analysis_execution::ImportedPublishedResult,
    pub field_recovery: runmat_analysis_fea::solve::dynamics::ResponseSpectrumFieldRecovery,
    pub envelope_state: runmat_analysis_fea::execution::CanonicalSemanticState,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralRandomResponseAnalysisResult {
    pub modal_workload: Box<StructuralModalWorkload>,
    pub response_workload: Box<runmat_analysis_fea::execution::StructuralRandomResponseWorkload>,
    pub modal_result: Box<StructuralModalSolverResult>,
    pub modal_numerical_receipt: StructuralModalNumericalReceipt,
    pub basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub response_result: Box<runmat_analysis_fea::execution::StructuralRandomResponseSolverResult>,
    pub response_numerical_receipt:
        runmat_analysis_fea::execution::StructuralRandomResponseNumericalReceipt,
    pub response_accuracy: runmat_analysis_fea::results::accuracy::AccuracyArtifact,
    pub response_resource_usage: runmat_analysis_execution::PreparedResourceUsage,
    pub response_publication: runmat_analysis_execution::ImportedPublishedResult,
    pub statistics_state: runmat_analysis_fea::execution::CanonicalSemanticState,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralBucklingAnalysisResult {
    pub preload_workload: Box<StructuralLinearWorkload>,
    pub buckling_workload: Box<StructuralBucklingWorkload>,
    pub preload_result: Box<runmat_analysis_fea::execution::StructuralLinearWorkloadResult>,
    pub preload_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub preload_numerical_receipt:
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    pub prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub buckling_result: Box<StructuralBucklingSolverResult>,
    pub buckling_numerical_receipt: StructuralBucklingNumericalReceipt,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralPrestressedModalAnalysisResult {
    pub preload_workload: Box<StructuralLinearWorkload>,
    pub modal_workload: Box<StructuralModalWorkload>,
    pub preload_result: Box<runmat_analysis_fea::execution::StructuralLinearWorkloadResult>,
    pub preload_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub preload_numerical_receipt:
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    pub prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub modal_result: Box<StructuralModalSolverResult>,
    pub modal_numerical_receipt: StructuralModalNumericalReceipt,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralPrestressedFrequencyAnalysisResult {
    pub preload_workload: Box<StructuralLinearWorkload>,
    pub preload_result: Box<runmat_analysis_fea::execution::StructuralLinearWorkloadResult>,
    pub preload_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub preload_numerical_receipt:
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    pub prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub frequency: ExactStructuralFrequencyAnalysisResult,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralPrestressedModalFrequencyAnalysisResult {
    pub preload_workload: Box<StructuralLinearWorkload>,
    pub modal_workload: Box<StructuralModalWorkload>,
    pub preload_result: Box<runmat_analysis_fea::execution::StructuralLinearWorkloadResult>,
    pub preload_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub preload_numerical_receipt:
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    pub prestress_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub modal_result: Box<StructuralModalSolverResult>,
    pub modal_numerical_receipt: StructuralModalNumericalReceipt,
    pub basis_state: runmat_analysis_fea::execution::CanonicalSemanticState,
    pub frequency: ExactStructuralFrequencyAnalysisResult,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralFrequencyAnalysisResult {
    pub resolved_workload: Box<StructuralFrequencyWorkload>,
    pub result: Box<StructuralFrequencySolverResult>,
    pub numerical_receipt: StructuralFrequencyNumericalReceipt,
    pub entries: Vec<ExactStructuralFrequencyEntryResult>,
    pub histories: Vec<ExactStructuralFrequencyHistoryResult>,
    pub resource_usage: runmat_analysis_execution::PreparedResourceUsage,
    pub publication: Box<runmat_analysis_fea::execution::StructuralFrequencyPublicationManifest>,
    pub object_catalog: PublishedObjectCatalog,
    pub artifact_roots: runmat_runtime::analysis::StructuralFrequencyArtifactRootIndex,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralFrequencyEntryResult {
    pub sample: runmat_analysis_fea::results::field::RecoveredFieldSample,
    pub recovered_fields: runmat_analysis_execution::ImportedRecoveredFields,
    pub observations: Option<runmat_analysis_execution::ImportedObservations>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralFrequencyHistoryResult {
    pub observation_id: runmat_analysis_core::ObservationId,
    pub checkpoint: runmat_analysis_execution::ImportedObservationHistoryCheckpointTask,
}

pub(crate) enum ExactStructuralAnalysisPublication<'a> {
    LinearSequence {
        inputs: &'a [crate::input_materialization::CompletedInputMaterialization],
        sequence: &'a CompletedStructuralSequence,
        unresolved_workloads_digest: Digest,
    },
    StructuralTransient {
        inputs: &'a [crate::input_materialization::CompletedInputMaterialization],
        completion: &'a CompletedStructuralTransientSequence,
        unresolved_workloads_digest: Digest,
    },
    StructuralModal {
        inputs: &'a [crate::input_materialization::CompletedInputMaterialization],
        completion: &'a crate::CompletedStructuralModalStudy,
        unresolved_workloads_digest: Digest,
    },
    StructuralModalTransient {
        inputs: &'a [crate::input_materialization::CompletedInputMaterialization],
        completion: &'a crate::CompletedStructuralModalTransientStudy,
        unresolved_workloads_digest: Digest,
    },
    StructuralModalResponseSpectrum {
        inputs: &'a [crate::input_materialization::CompletedInputMaterialization],
        completion: &'a crate::CompletedStructuralModalResponseSpectrumStudy,
        unresolved_workloads_digest: Digest,
    },
    StructuralRandomResponse {
        inputs: &'a [crate::input_materialization::CompletedInputMaterialization],
        completion: &'a crate::CompletedStructuralRandomResponseStudy,
        unresolved_workloads_digest: Digest,
    },
    StructuralBuckling {
        inputs: &'a [crate::input_materialization::CompletedInputMaterialization],
        completion: &'a crate::CompletedStructuralBucklingSequenceStudy,
        unresolved_workloads_digest: Digest,
    },
    StructuralPrestressedModal {
        inputs: &'a [crate::input_materialization::CompletedInputMaterialization],
        completion: &'a crate::CompletedStructuralPrestressedModalStudy,
        unresolved_workloads_digest: Digest,
    },
    StructuralPrestressedFrequency {
        inputs: &'a [crate::input_materialization::CompletedInputMaterialization],
        completion: &'a crate::CompletedStructuralPrestressedFrequencyStudy,
        unresolved_workloads_digest: Digest,
    },
    StructuralPrestressedModalFrequency {
        inputs: &'a [crate::input_materialization::CompletedInputMaterialization],
        completion: &'a crate::CompletedStructuralPrestressedModalFrequencyStudy,
        unresolved_workloads_digest: Digest,
    },
    StructuralFrequency {
        inputs: &'a [crate::input_materialization::CompletedInputMaterialization],
        completion: &'a crate::CompletedStructuralFrequencyStudy,
        unresolved_workloads_digest: Digest,
    },
}

impl ExactStructuralAnalysisPublication<'_> {
    fn inputs(&self) -> &[crate::input_materialization::CompletedInputMaterialization] {
        match self {
            Self::LinearSequence { inputs, .. }
            | Self::StructuralTransient { inputs, .. }
            | Self::StructuralModal { inputs, .. }
            | Self::StructuralModalTransient { inputs, .. }
            | Self::StructuralModalResponseSpectrum { inputs, .. }
            | Self::StructuralRandomResponse { inputs, .. }
            | Self::StructuralBuckling { inputs, .. }
            | Self::StructuralPrestressedModal { inputs, .. }
            | Self::StructuralPrestressedFrequency { inputs, .. }
            | Self::StructuralPrestressedModalFrequency { inputs, .. }
            | Self::StructuralFrequency { inputs, .. } => inputs,
        }
    }

    fn unresolved_workloads_digest(&self) -> Digest {
        match self {
            Self::LinearSequence {
                unresolved_workloads_digest,
                ..
            }
            | Self::StructuralTransient {
                unresolved_workloads_digest,
                ..
            }
            | Self::StructuralModal {
                unresolved_workloads_digest,
                ..
            }
            | Self::StructuralModalTransient {
                unresolved_workloads_digest,
                ..
            }
            | Self::StructuralModalResponseSpectrum {
                unresolved_workloads_digest,
                ..
            }
            | Self::StructuralRandomResponse {
                unresolved_workloads_digest,
                ..
            }
            | Self::StructuralBuckling {
                unresolved_workloads_digest,
                ..
            }
            | Self::StructuralPrestressedModal {
                unresolved_workloads_digest,
                ..
            }
            | Self::StructuralPrestressedFrequency {
                unresolved_workloads_digest,
                ..
            }
            | Self::StructuralPrestressedModalFrequency {
                unresolved_workloads_digest,
                ..
            }
            | Self::StructuralFrequency {
                unresolved_workloads_digest,
                ..
            } => *unresolved_workloads_digest,
        }
    }
}

fn validate_declared_roots(roots: &[&ValueRef], objects: &[ValueRef]) -> NativeExecutionResult<()> {
    if roots.iter().any(|root| !objects.contains(root))
        || roots
            .iter()
            .enumerate()
            .any(|(index, root)| roots[..index].contains(root))
    {
        return Err(protocol(
            "structural numerical closure omits or aliases a declared root",
        ));
    }
    reference_inventory(objects)?;
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralStudyResultExpectation {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub geometry_source_digest: Digest,
    pub workloads: StructuralStudyWorkloads,
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub analysis_accelerator_requests: Vec<runmat_execution::resource::AcceleratorRequest>,
    pub input_materializations: Vec<runmat_analysis_execution::InputMaterializationExpectation>,
    pub expected_meshing_request: MeshingRequest,
}

impl ExactStructuralStudyResultExpectation {
    pub fn study_identity(&self) -> NativeExecutionResult<StudyIdentity> {
        StudyIdentity::from_authored_digest(self.authored_study_digest, None).map_err(protocol)
    }

    pub fn validate(&self) -> NativeExecutionResult<()> {
        self.artifact_access.validate().map_err(protocol)?;
        self.workloads.validate().map_err(protocol)?;
        validate_analysis_accelerator_requests(
            &self.workloads,
            &self.analysis_accelerator_requests,
        )?;
        self.expected_meshing_request.validate().map_err(protocol)?;
        let model = self.workloads.authored_model().map_err(protocol)?;
        let authored_signals = model
            .signals
            .iter()
            .filter(|signal| signal.generator().is_some())
            .collect::<Vec<_>>();
        let authored_responses = model
            .response_spectra
            .iter()
            .filter(|spectrum| spectrum.ordinates.generator().is_some())
            .collect::<Vec<_>>();
        let authored_random = model
            .random_excitations
            .iter()
            .filter(|spectrum| spectrum.spectral_density.generator().is_some())
            .collect::<Vec<_>>();
        if authored_signals.len() + authored_responses.len() + authored_random.len()
            != self.input_materializations.len()
        {
            return Err(protocol(
                "whole-study input expectations differ from its authored workload",
            ));
        }
        let mut signals = authored_signals.into_iter();
        let mut responses = authored_responses.into_iter();
        let mut random = authored_random.into_iter();
        for materialization in &self.input_materializations {
            materialization.validate().map_err(protocol)?;
            let (revision, matches) = match materialization {
                runmat_analysis_execution::InputMaterializationExpectation::Signal(expected) => (
                    &expected.admitted.program_revision,
                    signals.next() == Some(&expected.authored),
                ),
                runmat_analysis_execution::InputMaterializationExpectation::Numerical(expected) => {
                    let matches = match &expected.authored {
                        runmat_analysis_execution::NumericalGeneratorSource::ResponseSpectrum(
                            spectrum,
                        ) => responses.next() == Some(spectrum.as_ref()),
                        runmat_analysis_execution::NumericalGeneratorSource::RandomSpectrum(
                            spectrum,
                        ) => random.next() == Some(spectrum.as_ref()),
                    };
                    (&expected.admitted.invocation.program_revision, matches)
                }
            };
            if revision != &self.program_revision || !matches {
                return Err(protocol(
                    "whole-study input expectation differs from its workload or program revision",
                ));
            }
        }
        if self.authored_study_digest.bytes() == &[0; 32]
            || self.geometry_source_digest.bytes() == &[0; 32]
            || self
                .frozen_project_digest
                .is_some_and(|digest| digest.bytes() == &[0; 32])
        {
            return Err(protocol("whole-study result expectation is invalid"));
        }
        Ok(())
    }
}
