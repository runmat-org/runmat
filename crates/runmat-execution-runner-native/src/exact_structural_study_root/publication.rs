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
mod closure_validation;
mod expectation;
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
mod results;
mod source_result_import;
mod specialized_publication_import;
#[cfg(test)]
mod tests;
mod thermal_prefix;

#[cfg(all(test, feature = "occt-native"))]
mod substitution_tests;
#[cfg(all(test, feature = "occt-native"))]
pub(crate) use substitution_tests::assert_foreign_thermal_substitution_rejected;

pub use expectation::ExactStructuralStudyResultExpectation;
use inventory::*;
pub(crate) use packaging::package_exact_structural_study_result;
pub use result_import::import_exact_structural_study_result;
pub use results::*;
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
    thermal: Option<thermal_prefix::ThermalPrefixClosure>,
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
        response_publication_baseline_usage_root: Box<ValueRef>,
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
        response_publication_baseline_usage_root: Box<ValueRef>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    finalization:
        Option<runmat_runtime::analysis::StructuralFrequencyFinalizedHistoryArtifactRoots>,
}

impl ExactStructuralStudyPublication {
    fn validate(&self, access: &ArtifactAccess) -> NativeExecutionResult<()> {
        if self.schema_version != 4
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
        if let Some(thermal) = &self.thermal {
            thermal.validate(access)?;
            if self.compiled_study.is_none()
                || !matches!(self.analysis,
                ExactStructuralStudyAnalysisClosure::StructuralBuckling { .. }
                | ExactStructuralStudyAnalysisClosure::StructuralPrestressedModal { .. }
                | ExactStructuralStudyAnalysisClosure::StructuralPrestressedFrequency { .. }
                | ExactStructuralStudyAnalysisClosure::StructuralPrestressedModalFrequency { .. })
            {
                return Err(protocol(
                    "thermal prefix requires a compiled prestress publication",
                ));
            }
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

pub(crate) enum ExactStructuralAnalysisPublication<'a> {
    ThermalPrestress {
        inputs: &'a [crate::input_materialization::CompletedInputMaterialization],
        completion: &'a crate::CompletedThermalPrestressStudy,
        unresolved_workloads_digest: Digest,
    },
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
            Self::ThermalPrestress { inputs, .. }
            | Self::LinearSequence { inputs, .. }
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
            Self::ThermalPrestress {
                unresolved_workloads_digest,
                ..
            }
            | Self::LinearSequence {
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
