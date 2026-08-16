//! Logical meshing identities exclude execution attempts and physical placement.
//!
//! Callers hash these validated records with the domain-separated canonical codec. Partition
//! indices and join inputs use canonical entity order, never worker completion order.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{validate_token, MeshingContractError};

pub const MESHING_IDENTITY_SCHEMA_VERSION: u16 = 2;
const MAX_PREREQUISITE_DIGESTS: usize = 64;
const MAX_JOIN_PARTITIONS: usize = 4096;
const MAX_RESULT_CHUNKS: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StableDigest(pub [u8; 32]);

impl StableDigest {
    pub const ZERO: Self = Self([0; 32]);

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn validate_nonzero(&self, field: &str) -> Result<(), MeshingContractError> {
        if *self == Self::ZERO {
            return Err(MeshingContractError::invalid(
                field,
                "digest must not be all zeroes",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryRevisionRef {
    pub source_digest: StableDigest,
    pub geometry_revision: u64,
    pub persistent_mapping_version: u32,
}

impl GeometryRevisionRef {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        self.source_digest
            .validate_nonzero("geometry.source_digest")?;
        if self.geometry_revision == 0 || self.persistent_mapping_version == 0 {
            return Err(MeshingContractError::invalid(
                "geometry revision",
                "geometry revision and persistent mapping version must be non-zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersistentEntityKind {
    Assembly,
    Instance,
    Body,
    Lump,
    Solid,
    Shell,
    Face,
    Wire,
    Coedge,
    Edge,
    Vertex,
    Region,
    Contact,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersistentEntityId {
    pub kind: PersistentEntityKind,
    pub source_topology_id: String,
    #[serde(default)]
    pub assembly_path: Vec<String>,
}

impl PersistentEntityId {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        validate_token(
            "persistent entity source topology id",
            &self.source_topology_id,
            512,
        )?;
        if self.assembly_path.len() > 256 {
            return Err(MeshingContractError::invalid(
                "persistent entity assembly path",
                "must contain at most 256 segments",
            ));
        }
        for segment in &self.assembly_path {
            validate_token("persistent entity assembly path segment", segment, 256)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalEntityRangeV2 {
    pub first: PersistentEntityId,
    pub last: PersistentEntityId,
    pub entity_count: u64,
}

impl CanonicalEntityRangeV2 {
    fn validate(&self) -> Result<(), MeshingContractError> {
        self.first.validate()?;
        self.last.validate()?;
        if self.entity_count == 0 || self.first.kind != self.last.kind || self.first > self.last {
            return Err(MeshingContractError::invalid(
                "canonical entity range",
                "range must be non-empty, ordered, and contain one entity kind",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshingPartitionKindV2 {
    WholeStage,
    CanonicalEntityBatch,
    DisconnectedComponent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingPartitionDescriptorV2 {
    pub kind: MeshingPartitionKindV2,
    pub partition_index: u32,
    pub partition_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_range: Option<CanonicalEntityRangeV2>,
}

impl MeshingPartitionDescriptorV2 {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.partition_count == 0 || self.partition_index >= self.partition_count {
            return Err(MeshingContractError::invalid(
                "meshing partition descriptor",
                "partition index must be within a non-empty partition set",
            ));
        }
        match (&self.kind, &self.entity_range) {
            (MeshingPartitionKindV2::WholeStage, None)
                if self.partition_index == 0 && self.partition_count == 1 => {}
            (MeshingPartitionKindV2::CanonicalEntityBatch, Some(range))
            | (MeshingPartitionKindV2::DisconnectedComponent, Some(range)) => range.validate()?,
            _ => {
                return Err(MeshingContractError::invalid(
                    "meshing partition descriptor",
                    "whole-stage and partitioned work have inconsistent entity ranges",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingStageIdentityV2 {
    pub schema_version: u16,
    pub stage: super::MeshingStageV2,
    pub geometry: GeometryRevisionRef,
    pub resolved_request_digest: StableDigest,
    pub tolerance_policy_digest: StableDigest,
    pub metric_policy_digest: StableDigest,
    pub algorithm_set_digest: StableDigest,
    pub deterministic_seed: u64,
    pub prerequisite_artifact_digests: Vec<StableDigest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability_cohort: Option<String>,
}

impl MeshingStageIdentityV2 {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != MESHING_IDENTITY_SCHEMA_VERSION {
            return Err(MeshingContractError::invalid(
                "meshing identity schema version",
                format!("expected {MESHING_IDENTITY_SCHEMA_VERSION}"),
            ));
        }
        self.geometry.validate()?;
        for (field, digest) in [
            ("resolved request digest", self.resolved_request_digest),
            ("tolerance policy digest", self.tolerance_policy_digest),
            ("metric policy digest", self.metric_policy_digest),
            ("algorithm set digest", self.algorithm_set_digest),
        ] {
            digest.validate_nonzero(field)?;
        }
        validate_digest_list(
            "prerequisite artifact digests",
            &self.prerequisite_artifact_digests,
            MAX_PREREQUISITE_DIGESTS,
            true,
        )?;
        if let Some(cohort) = &self.capability_cohort {
            validate_token("capability cohort", cohort, 128)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingPartitionIdentityV2 {
    pub schema_version: u16,
    pub stage_identity_digest: StableDigest,
    pub partition: MeshingPartitionDescriptorV2,
}

impl MeshingPartitionIdentityV2 {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        validate_identity_version(self.schema_version)?;
        self.stage_identity_digest
            .validate_nonzero("stage identity digest")?;
        self.partition.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingPartitionResultRefV2 {
    pub partition_index: u32,
    pub result_digest: StableDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingJoinIdentityV2 {
    pub schema_version: u16,
    pub stage_identity_digest: StableDigest,
    pub join_algorithm_version: String,
    pub ordered_partition_results: Vec<MeshingPartitionResultRefV2>,
}

impl MeshingJoinIdentityV2 {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        validate_identity_version(self.schema_version)?;
        self.stage_identity_digest
            .validate_nonzero("join stage identity digest")?;
        validate_token("join algorithm version", &self.join_algorithm_version, 128)?;
        if self.ordered_partition_results.is_empty()
            || self.ordered_partition_results.len() > MAX_JOIN_PARTITIONS
        {
            return Err(MeshingContractError::invalid(
                "join partition results",
                "must contain a bounded, non-empty partition set",
            ));
        }
        for (expected, result) in self.ordered_partition_results.iter().enumerate() {
            if result.partition_index != expected as u32 {
                return Err(MeshingContractError::invalid(
                    "join partition results",
                    "partition indices must be contiguous and canonically ordered",
                ));
            }
            result
                .result_digest
                .validate_nonzero("partition result digest")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingStageResultIdentityV2 {
    pub schema_version: u16,
    pub stage: super::MeshingStageV2,
    pub producer_identity_digest: StableDigest,
    pub ordered_chunk_digests: Vec<StableDigest>,
    pub invariant_summary_digest: StableDigest,
}

impl MeshingStageResultIdentityV2 {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        validate_identity_version(self.schema_version)?;
        self.producer_identity_digest
            .validate_nonzero("stage result producer identity")?;
        self.invariant_summary_digest
            .validate_nonzero("stage result invariant summary")?;
        if self.ordered_chunk_digests.is_empty()
            || self.ordered_chunk_digests.len() > MAX_RESULT_CHUNKS
            || self
                .ordered_chunk_digests
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != self.ordered_chunk_digests.len()
        {
            return Err(MeshingContractError::invalid(
                "stage result chunks",
                "must contain a bounded, non-empty ordered chunk list",
            ));
        }
        for digest in &self.ordered_chunk_digests {
            digest.validate_nonzero("stage result chunk digest")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingValidationIdentityV2 {
    pub schema_version: u16,
    pub subject_stage_result_digest: StableDigest,
    pub geometry: GeometryRevisionRef,
    pub resolved_request_digest: StableDigest,
    pub validation_algorithm_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability_cohort: Option<String>,
}

impl MeshingValidationIdentityV2 {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        validate_identity_version(self.schema_version)?;
        self.subject_stage_result_digest
            .validate_nonzero("validated stage result digest")?;
        self.geometry.validate()?;
        self.resolved_request_digest
            .validate_nonzero("validation request digest")?;
        validate_token(
            "validation algorithm version",
            &self.validation_algorithm_version,
            128,
        )?;
        if let Some(cohort) = &self.capability_cohort {
            validate_token("validation capability cohort", cohort, 128)?;
        }
        Ok(())
    }
}

pub(super) fn validate_digest_list(
    field: &str,
    digests: &[StableDigest],
    maximum: usize,
    allow_empty: bool,
) -> Result<(), MeshingContractError> {
    if (!allow_empty && digests.is_empty())
        || digests.len() > maximum
        || !digests.windows(2).all(|pair| pair[0] < pair[1])
    {
        return Err(MeshingContractError::invalid(
            field,
            "must be bounded, unique, and canonically ordered",
        ));
    }
    for digest in digests {
        digest.validate_nonzero(field)?;
    }
    Ok(())
}

fn validate_identity_version(version: u16) -> Result<(), MeshingContractError> {
    if version != MESHING_IDENTITY_SCHEMA_VERSION {
        return Err(MeshingContractError::invalid(
            "meshing identity schema version",
            format!("expected {MESHING_IDENTITY_SCHEMA_VERSION}"),
        ));
    }
    Ok(())
}
