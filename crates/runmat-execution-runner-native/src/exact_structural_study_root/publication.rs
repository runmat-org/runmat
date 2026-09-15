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

use super::boundary::BoundaryRegionResolution;
use super::contract::{
    structural_workload_inventory_digest, validate_analysis_accelerator_requests,
    DecodedExactStructuralStudyRoot,
};
use super::{
    protocol, read, CONTRACT_LIMITS, PUBLICATION_DOMAIN, PUBLICATION_MEDIA_TYPE,
    PUBLICATION_PREFIX, PUBLICATION_VALUE_SCHEMA,
};
use crate::{CompletedStructuralTransientSequence, NativeExecutionResult};

mod buckling_import;
mod frequency_import;
mod inventory;
mod modal_import;
mod packaging;
mod result_import;

use inventory::*;
pub(crate) use packaging::package_exact_structural_study_result;
pub use result_import::import_exact_structural_study_result;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactStructuralStudyPublication {
    schema_version: u16,
    program_revision: ProgramRevision,
    authored_study_digest: Digest,
    unresolved_workloads_digest: Digest,
    meshing_request_digest: Digest,
    meshing_root: ValueRef,
    meshing_objects: Vec<ValueRef>,
    boundary_root: ValueRef,
    boundary_objects: Vec<ValueRef>,
    signal_roots: Vec<ValueRef>,
    signal_objects: Vec<ValueRef>,
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
    StructuralBuckling {
        preload_result_root: Box<ValueRef>,
        preload_receipt_root: Box<ValueRef>,
        prestress_state_root: Box<ValueRef>,
        buckling_result_root: Box<ValueRef>,
        buckling_receipt_root: Box<ValueRef>,
        objects: Vec<ValueRef>,
    },
    StructuralFrequency {
        result_root: Box<ValueRef>,
        receipt_root: Box<ValueRef>,
        publication: Box<runmat_analysis_fea::execution::StructuralFrequencyPublicationManifest>,
        entries: Vec<ExactStructuralFrequencyEntryClosure>,
        histories: Vec<ExactStructuralFrequencyHistoryClosure>,
        resource_usage_root: Box<ValueRef>,
        objects: Vec<ValueRef>,
    },
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
            | Self::StructuralBuckling { objects, .. }
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
            Self::StructuralFrequency {
                result_root,
                receipt_root,
                publication,
                entries,
                histories,
                resource_usage_root,
                objects,
            } => {
                publication.validate().map_err(protocol)?;
                if result_root == receipt_root
                    || entries.is_empty()
                    || !objects.contains(result_root)
                    || !objects.contains(receipt_root)
                    || !objects.contains(resource_usage_root)
                    || entries.iter().any(|entry| {
                        !objects.contains(&entry.fields_root)
                            || entry
                                .observations_root
                                .as_ref()
                                .is_some_and(|root| !objects.contains(root))
                    })
                    || histories
                        .iter()
                        .any(|history| !objects.contains(&history.checkpoint_root))
                {
                    return Err(protocol(
                        "structural frequency closure omits a declared artifact root",
                    ));
                }
                let history_ids = histories
                    .iter()
                    .map(|history| &history.observation_id)
                    .collect::<BTreeSet<_>>();
                if history_ids.len() != histories.len() {
                    return Err(protocol(
                        "structural frequency closure contains duplicate histories",
                    ));
                }
                reference_inventory(objects)?;
            }
        }
        Ok(())
    }
}

impl ExactStructuralStudyPublication {
    fn validate(&self, access: &ArtifactAccess) -> NativeExecutionResult<()> {
        if self.schema_version != 1
            || self.authored_study_digest.bytes() == &[0; 32]
            || self.unresolved_workloads_digest.bytes() == &[0; 32]
            || self.meshing_request_digest.bytes() == &[0; 32]
            || self.meshing_objects.is_empty()
            || self.boundary_objects.is_empty()
            || self.analysis.objects().is_empty()
            || self.signal_roots.len() > self.signal_objects.len()
            || self
                .signal_roots
                .iter()
                .any(|root| !self.signal_objects.contains(root))
        {
            return Err(protocol("exact structural study publication is invalid"));
        }
        for reference in self
            .meshing_objects
            .iter()
            .chain(&self.boundary_objects)
            .chain(&self.signal_objects)
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
            (&self.boundary_root, &self.boundary_objects),
        ] {
            if !inventory.contains(root) {
                return Err(protocol(
                    "study publication closure omits a declared component root",
                ));
            }
            reference_inventory(inventory)?;
        }
        self.analysis.validate()?;
        reference_inventory(&self.signal_objects)?;
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
    pub mesh: ExactMeshingDagRunResult,
    pub boundary_resolution: BoundaryRegionResolution,
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
    StructuralBuckling(Box<ExactStructuralBucklingAnalysisResult>),
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
pub struct ExactStructuralFrequencyAnalysisResult {
    pub resolved_workload: Box<StructuralFrequencyWorkload>,
    pub result: Box<StructuralFrequencySolverResult>,
    pub numerical_receipt: StructuralFrequencyNumericalReceipt,
    pub entries: Vec<ExactStructuralFrequencyEntryResult>,
    pub histories: Vec<ExactStructuralFrequencyHistoryResult>,
    pub resource_usage: runmat_analysis_execution::PreparedStructuralResourceUsage,
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
        signals:
            &'a [crate::remote::signal_materialization::CompletedRemoteSignalMaterialization],
        sequence: &'a CompletedStructuralSequence,
        unresolved_workloads_digest: Digest,
    },
    StructuralTransient {
        signals:
            &'a [crate::remote::signal_materialization::CompletedRemoteSignalMaterialization],
        completion: &'a CompletedStructuralTransientSequence,
        unresolved_workloads_digest: Digest,
    },
    StructuralModal {
        signals:
            &'a [crate::remote::signal_materialization::CompletedRemoteSignalMaterialization],
        completion: &'a crate::CompletedStructuralModalStudy,
        unresolved_workloads_digest: Digest,
    },
    StructuralModalTransient {
        signals:
            &'a [crate::remote::signal_materialization::CompletedRemoteSignalMaterialization],
        completion: &'a crate::CompletedStructuralModalTransientStudy,
        unresolved_workloads_digest: Digest,
    },
    StructuralBuckling {
        signals:
            &'a [crate::remote::signal_materialization::CompletedRemoteSignalMaterialization],
        completion: &'a crate::CompletedStructuralBucklingSequenceStudy,
        unresolved_workloads_digest: Digest,
    },
    StructuralFrequency {
        signals:
            &'a [crate::remote::signal_materialization::CompletedRemoteSignalMaterialization],
        completion: &'a crate::CompletedStructuralFrequencyStudy,
        unresolved_workloads_digest: Digest,
    },
}

impl ExactStructuralAnalysisPublication<'_> {
    fn signals(
        &self,
    ) -> &[crate::remote::signal_materialization::CompletedRemoteSignalMaterialization] {
        match self {
            Self::LinearSequence { signals, .. }
            | Self::StructuralTransient { signals, .. }
            | Self::StructuralModal { signals, .. }
            | Self::StructuralModalTransient { signals, .. }
            | Self::StructuralBuckling { signals, .. }
            | Self::StructuralFrequency { signals, .. } => signals,
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
            | Self::StructuralBuckling {
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
    pub analysis_accelerator_requests: Vec<runmat_execution::resource::AcceleratorRequest>,
    pub signal_materializations: Vec<runmat_analysis_execution::SignalMaterializationExpectation>,
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
        let authored_signals = self
            .workloads
            .authored_model()
            .map_err(protocol)?
            .signals
            .iter()
            .filter(|signal| signal.generator().is_some())
            .collect::<Vec<_>>();
        if authored_signals.len() != self.signal_materializations.len()
            || authored_signals
                .iter()
                .zip(&self.signal_materializations)
                .any(|(signal, expected)| *signal != &expected.authored)
        {
            return Err(protocol(
                "whole-study signal expectations differ from its authored workload",
            ));
        }
        for materialization in &self.signal_materializations {
            materialization.validate().map_err(protocol)?;
            if materialization.admitted.program_revision != self.program_revision {
                return Err(protocol(
                    "whole-study signal expectation uses another program revision",
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
