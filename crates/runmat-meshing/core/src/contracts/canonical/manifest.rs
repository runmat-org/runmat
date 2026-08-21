//! Meshing-owned logical manifests over execution-artifact-owned content objects.
//!
//! Descriptors carry no path, worker, encryption, retention, or transfer state. The execution
//! artifact system remains authoritative for those concerns and independently verifies bytes.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{identity::validate_digest_list, MeshingContractError, MeshingStageKind, StableDigest};

pub const MESHING_STAGE_MANIFEST_SCHEMA_VERSION: u16 = 3;
const MAX_CHUNKS_PER_MANIFEST: usize = 65_536;
const MAX_MANIFEST_PREREQUISITES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshingChunkMediaType {
    GeometrySource,
    ExactGeometry,
    FacetedGeometry,
    MetricField,
    CurvePartitions,
    CurveMesh,
    SurfacePartitions,
    SurfaceMesh,
    ProtectedBoundaryComplex,
    VolumeTopology,
    MeshNodes,
    MeshElements,
    MeshClassification,
    ValidationEvidence,
    DiagnosticEvidence,
    AnalysisMeshArtifact,
    MeshingEvidence,
}

impl MeshingChunkMediaType {
    pub const fn media_type(self) -> &'static str {
        match self {
            Self::GeometrySource => "application/vnd.runmat.geometry-source.v2",
            Self::ExactGeometry => "application/vnd.runmat.exact-geometry.v2",
            Self::FacetedGeometry => "application/vnd.runmat.faceted-solid.v2",
            Self::MetricField => "application/vnd.runmat.metric-field.v2",
            Self::CurvePartitions => "application/vnd.runmat.mesh-curves.v2",
            Self::CurveMesh => "application/vnd.runmat.shared-curve-mesh.v4",
            Self::SurfacePartitions => "application/vnd.runmat.mesh-surfaces.v2",
            Self::SurfaceMesh => "application/vnd.runmat.exact-surface-mesh.v1",
            Self::ProtectedBoundaryComplex => "application/vnd.runmat.mesh-plc.v2",
            Self::VolumeTopology => "application/vnd.runmat.mesh-volume.v2",
            Self::MeshNodes => "application/vnd.runmat.mesh-nodes.v2",
            Self::MeshElements => "application/vnd.runmat.mesh-elements.v2",
            Self::MeshClassification => "application/vnd.runmat.mesh-classification.v2",
            Self::ValidationEvidence => "application/vnd.runmat.mesh-validation.v2",
            Self::DiagnosticEvidence => "application/vnd.runmat.mesh-diagnostics.v2",
            Self::AnalysisMeshArtifact => "application/vnd.runmat.analysis-mesh.v2",
            Self::MeshingEvidence => "application/vnd.runmat.meshing-evidence.v2",
        }
    }

    pub(super) fn from_media_type(value: &str) -> Option<Self> {
        [
            Self::GeometrySource,
            Self::ExactGeometry,
            Self::FacetedGeometry,
            Self::MetricField,
            Self::CurvePartitions,
            Self::CurveMesh,
            Self::SurfacePartitions,
            Self::SurfaceMesh,
            Self::ProtectedBoundaryComplex,
            Self::VolumeTopology,
            Self::MeshNodes,
            Self::MeshElements,
            Self::MeshClassification,
            Self::ValidationEvidence,
            Self::DiagnosticEvidence,
            Self::AnalysisMeshArtifact,
            Self::MeshingEvidence,
        ]
        .into_iter()
        .find(|candidate| candidate.media_type() == value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingChunkDescriptor {
    pub ordinal: u32,
    pub first_logical_entity_ordinal: u64,
    pub digest: StableDigest,
    pub media_type: MeshingChunkMediaType,
    pub schema_version: u16,
    pub encoded_length: u64,
    pub decoded_length: u64,
    pub logical_entity_count: u64,
}

impl MeshingChunkDescriptor {
    fn validate(&self, expected_ordinal: usize) -> Result<(), MeshingContractError> {
        if self.ordinal != expected_ordinal as u32
            || self.schema_version == 0
            || self.encoded_length == 0
            || self.decoded_length == 0
            || self.logical_entity_count == 0
        {
            return Err(MeshingContractError::invalid(
                "meshing chunk descriptor",
                "ordinal, schema version, and byte lengths must be valid",
            ));
        }
        self.digest.validate_nonzero("meshing chunk digest")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshingStageResultKind {
    WholeStage,
    Partition,
    DeterministicJoin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshingManifestDisposition {
    ValidatedDependency,
    DiagnosticOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingStageManifest {
    pub schema_version: u16,
    pub stage: MeshingStageKind,
    pub result_kind: MeshingStageResultKind,
    pub logical_result_identity: StableDigest,
    pub disposition: MeshingManifestDisposition,
    pub prerequisite_manifest_digests: Vec<StableDigest>,
    pub invariant_summary_digest: StableDigest,
    pub chunks: Vec<MeshingChunkDescriptor>,
    pub total_encoded_length: u64,
}

impl MeshingStageManifest {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != MESHING_STAGE_MANIFEST_SCHEMA_VERSION {
            return Err(MeshingContractError::invalid(
                "meshing stage manifest schema version",
                format!("expected {MESHING_STAGE_MANIFEST_SCHEMA_VERSION}"),
            ));
        }
        self.logical_result_identity
            .validate_nonzero("logical stage result identity")?;
        self.invariant_summary_digest
            .validate_nonzero("stage invariant summary digest")?;
        validate_digest_list(
            "prerequisite manifest digests",
            &self.prerequisite_manifest_digests,
            MAX_MANIFEST_PREREQUISITES,
            true,
        )?;
        if self.chunks.is_empty() || self.chunks.len() > MAX_CHUNKS_PER_MANIFEST {
            return Err(MeshingContractError::invalid(
                "meshing stage manifest chunks",
                "manifest must contain a bounded, non-empty chunk inventory",
            ));
        }
        let mut chunk_digests = BTreeSet::new();
        let mut total = 0_u64;
        let mut next_entity_ordinal = 0_u64;
        for (ordinal, chunk) in self.chunks.iter().enumerate() {
            chunk.validate(ordinal)?;
            if chunk.first_logical_entity_ordinal != next_entity_ordinal
                || !chunk_digests.insert(chunk.digest)
            {
                return Err(MeshingContractError::invalid(
                    "meshing stage manifest chunks",
                    "chunk entity ranges must be contiguous and digests must be unique",
                ));
            }
            next_entity_ordinal = next_entity_ordinal
                .checked_add(chunk.logical_entity_count)
                .ok_or_else(|| {
                    MeshingContractError::invalid(
                        "meshing stage manifest chunks",
                        "logical entity count overflow",
                    )
                })?;
            total = total.checked_add(chunk.encoded_length).ok_or_else(|| {
                MeshingContractError::invalid(
                    "meshing stage manifest length",
                    "encoded chunk length overflow",
                )
            })?;
        }
        if total != self.total_encoded_length {
            return Err(MeshingContractError::invalid(
                "meshing stage manifest length",
                "declared length must equal the ordered chunk inventory",
            ));
        }
        Ok(())
    }

    pub const fn is_dependency_eligible(&self) -> bool {
        matches!(
            self.disposition,
            MeshingManifestDisposition::ValidatedDependency
        )
    }
}
