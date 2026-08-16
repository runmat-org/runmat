//! Adapter-neutral meshing work and detailed-progress contracts.
//!
//! These records describe geometry work. The execution bridge maps domain requirements onto the
//! shared resource/capability model; scheduling, attempts, retries, fencing, and workers do not
//! enter these contracts.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{
    identity::validate_digest_list, validate_token, MeshElementOrderV2, MeshingContractError,
    MeshingFailure, MeshingPartitionDescriptorV2, MeshingStageV2, StableDigest,
};

pub const MESHING_WORKLOAD_SCHEMA_VERSION: u16 = 2;
pub const MESHING_PROGRESS_SCHEMA_VERSION: u16 = 2;
const MAX_INPUT_MANIFESTS: usize = 64;
const MAX_CAPABILITIES: usize = 16;
const MAX_PROGRESS_COUNTS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeshingCapabilityRequirementV2 {
    HostWorkload { abi: String },
    ExactCadKernel { abi: String },
    MeshingAlgorithm { version: String },
    ElementOrder { order: MeshElementOrderV2 },
    DeterministicPlatformCohort { cohort: String },
}

impl MeshingCapabilityRequirementV2 {
    fn validate(&self) -> Result<(), MeshingContractError> {
        match self {
            Self::HostWorkload { abi } => validate_token("meshing host ABI", abi, 128),
            Self::ExactCadKernel { abi } => validate_token("exact CAD kernel ABI", abi, 128),
            Self::MeshingAlgorithm { version } => {
                validate_token("meshing algorithm version", version, 128)
            }
            Self::ElementOrder { .. } => Ok(()),
            Self::DeterministicPlatformCohort { cohort } => {
                validate_token("deterministic platform cohort", cohort, 128)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingWorkloadRequestV2 {
    pub schema_version: u16,
    pub stage: MeshingStageV2,
    pub stage_identity_digest: StableDigest,
    pub partition: MeshingPartitionDescriptorV2,
    pub input_manifest_digests: Vec<StableDigest>,
    pub required_capabilities: Vec<MeshingCapabilityRequirementV2>,
}

impl MeshingWorkloadRequestV2 {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != MESHING_WORKLOAD_SCHEMA_VERSION {
            return Err(MeshingContractError::invalid(
                "meshing workload schema version",
                format!("expected {MESHING_WORKLOAD_SCHEMA_VERSION}"),
            ));
        }
        self.stage_identity_digest
            .validate_nonzero("workload stage identity digest")?;
        self.partition.validate()?;
        if matches!(
            self.partition.kind,
            super::MeshingPartitionKindV2::CanonicalEntityBatch
        ) && !matches!(
            self.stage,
            MeshingStageV2::Sizing | MeshingStageV2::CurveMesh | MeshingStageV2::SurfaceMesh
        ) {
            return Err(MeshingContractError::invalid(
                "meshing workload partition",
                "entity batching is limited to independently composable sizing, curve, and surface work",
            ));
        }
        validate_digest_list(
            "workload input manifests",
            &self.input_manifest_digests,
            MAX_INPUT_MANIFESTS,
            true,
        )?;
        if self.stage != MeshingStageV2::GeometryAdmission && self.input_manifest_digests.is_empty()
        {
            return Err(MeshingContractError::invalid(
                "workload input manifests",
                "every stage after geometry admission requires an artifact dependency",
            ));
        }
        if self.required_capabilities.len() > MAX_CAPABILITIES
            || !self
                .required_capabilities
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        {
            return Err(MeshingContractError::invalid(
                "meshing workload capabilities",
                "must be bounded, unique, and canonically ordered",
            ));
        }
        for capability in &self.required_capabilities {
            capability.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeshingWorkloadResultV2 {
    Validated { stage_manifest_digest: StableDigest },
    Failed { failure: MeshingFailure },
}

impl MeshingWorkloadResultV2 {
    pub(super) fn validate_standalone(&self) -> Result<(), MeshingContractError> {
        match self {
            Self::Validated {
                stage_manifest_digest,
            } => stage_manifest_digest.validate_nonzero("workload result manifest digest"),
            Self::Failed { failure } => failure.validate(),
        }
    }

    pub fn validate_against(
        &self,
        request: &MeshingWorkloadRequestV2,
    ) -> Result<(), MeshingContractError> {
        request.validate()?;
        self.validate_standalone()?;
        match self {
            Self::Validated { .. } => Ok(()),
            Self::Failed { failure } => {
                if failure.stage != request.stage {
                    return Err(MeshingContractError::invalid(
                        "workload failure stage",
                        "failure must belong to the requested meshing stage",
                    ));
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingProgressV2 {
    pub schema_version: u16,
    pub stage: MeshingStageV2,
    pub partition_index: u32,
    pub sequence: u64,
    pub completed_work: u64,
    pub estimated_work: u64,
    pub entity_counts: BTreeMap<String, u64>,
    pub peak_memory_bytes: u64,
    pub elapsed_time_ms: u64,
    pub consumed_search_work: u64,
    pub cancellation_checkpoint: u64,
}

impl MeshingProgressV2 {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != MESHING_PROGRESS_SCHEMA_VERSION {
            return Err(MeshingContractError::invalid(
                "meshing progress schema version",
                format!("expected {MESHING_PROGRESS_SCHEMA_VERSION}"),
            ));
        }
        if self.completed_work > self.estimated_work
            || self.entity_counts.len() > MAX_PROGRESS_COUNTS
        {
            return Err(MeshingContractError::invalid(
                "meshing progress",
                "work bounds or entity counter bounds are invalid",
            ));
        }
        for key in self.entity_counts.keys() {
            validate_token("meshing progress entity counter", key, 128)?;
        }
        Ok(())
    }

    pub fn validate_after(&self, previous: &Self) -> Result<(), MeshingContractError> {
        self.validate()?;
        previous.validate()?;
        if self.stage != previous.stage
            || self.partition_index != previous.partition_index
            || self.sequence <= previous.sequence
            || self.completed_work < previous.completed_work
            || self.estimated_work < previous.estimated_work
            || self.peak_memory_bytes < previous.peak_memory_bytes
            || self.elapsed_time_ms < previous.elapsed_time_ms
            || self.consumed_search_work < previous.consumed_search_work
            || self.cancellation_checkpoint < previous.cancellation_checkpoint
            || previous
                .entity_counts
                .iter()
                .any(|(name, count)| self.entity_counts.get(name).is_none_or(|next| next < count))
        {
            return Err(MeshingContractError::invalid(
                "meshing progress transition",
                "progress for one partition must be strictly sequenced and monotone",
            ));
        }
        Ok(())
    }
}
