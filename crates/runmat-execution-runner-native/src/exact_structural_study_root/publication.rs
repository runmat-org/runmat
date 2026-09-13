use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_execution::{
    CompletedStructuralSequence, StructuralSequencePreparation, StructuralSequenceResultClosure,
    StructuralStudyWorkloads,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, StructuralLinearWorkload, StructuralTransientNumericalReceipt,
    StructuralTransientSolverResult, StructuralTransientWorkload, StudyIdentity,
};
use runmat_canonical_codec::{decode_contract, encode_contract};
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, LogicalObject, ObjectNamespace, ProgramExecutionObject,
    ProgramExecutionRequest, ProgramExecutionResponse,
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

mod inventory;
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
    DirectTransient {
        result_root: Box<ValueRef>,
        receipt_root: Box<ValueRef>,
        objects: Vec<ValueRef>,
    },
}

impl ExactStructuralStudyAnalysisClosure {
    fn objects(&self) -> &[ValueRef] {
        match self {
            Self::LinearSequence { objects, .. } | Self::DirectTransient { objects, .. } => objects,
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
            Self::DirectTransient {
                result_root,
                receipt_root,
                objects,
                ..
            } => {
                if result_root == receipt_root
                    || !objects.contains(result_root)
                    || !objects.contains(receipt_root)
                {
                    return Err(protocol(
                        "transient result closure omits or aliases a declared root",
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
    DirectTransient {
        resolved_workload: Box<StructuralTransientWorkload>,
        result: Box<StructuralTransientSolverResult>,
        numerical_receipt: StructuralTransientNumericalReceipt,
    },
}

pub(crate) enum ExactStructuralAnalysisPublication<'a> {
    LinearSequence {
        signals:
            &'a [crate::remote::signal_materialization::CompletedRemoteSignalMaterialization],
        sequence: &'a CompletedStructuralSequence,
        unresolved_workloads_digest: Digest,
    },
    DirectTransient {
        signals:
            &'a [crate::remote::signal_materialization::CompletedRemoteSignalMaterialization],
        completion: &'a CompletedStructuralTransientSequence,
        unresolved_workloads_digest: Digest,
    },
}

impl ExactStructuralAnalysisPublication<'_> {
    fn signals(
        &self,
    ) -> &[crate::remote::signal_materialization::CompletedRemoteSignalMaterialization] {
        match self {
            Self::LinearSequence { signals, .. } | Self::DirectTransient { signals, .. } => signals,
        }
    }

    fn unresolved_workloads_digest(&self) -> Digest {
        match self {
            Self::LinearSequence {
                unresolved_workloads_digest,
                ..
            }
            | Self::DirectTransient {
                unresolved_workloads_digest,
                ..
            } => *unresolved_workloads_digest,
        }
    }
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
