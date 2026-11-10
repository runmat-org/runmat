use crate::topology::storage::{self, TopologyStorageError};

use runmat_analysis_core::StructuralModel;
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{
    CanonicalMeshingContract, PersistentEntityKind, SolverMeshArtifact, StableDigest,
};
use serde::Serialize;
use thiserror::Error;

use super::{
    explicit_node_identity, explicit_topology_digest, CanonicalStructuralTopologyIdentity,
};

mod construction;
mod resources;
pub(crate) use resources::node_space_workspace_profile;

const IDENTITY_PREFIX: &[u8] = b"runmat-fea-mixed-structural-topology/v1\0";
const IDENTITY_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(4_000_000, 64);

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct CanonicalStructuralNode {
    pub identity: StableDigest,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
    pub owns_solid_translation: bool,
    pub owns_explicit_rotation: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct CanonicalStructuralNodeSpace {
    pub nodes: Vec<CanonicalStructuralNode>,
    pub solid_node_count: usize,
    pub explicit_node_indices: Vec<usize>,
    pub identity: CanonicalStructuralTopologyIdentity,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub(crate) enum StructuralNodeSpaceError {
    #[error(transparent)]
    Storage(#[from] TopologyStorageError),
    #[error("mixed structural topology requires a valid canonical solver mesh: {0}")]
    InvalidMesh(String),
    #[error(
        "explicit node {node_id} attachment `{attachment_id}` did not resolve to one solid-mesh node"
    )]
    MissingAttachment { node_id: u32, attachment_id: String },
    #[error(
        "explicit node {node_id} attachment `{attachment_id}` resolved to {matched_node_count} solid-mesh nodes"
    )]
    AmbiguousAttachment {
        node_id: u32,
        attachment_id: String,
        matched_node_count: usize,
    },
    #[error(
        "explicit node {node_id} is {distance_m} m from attachment `{attachment_id}`, exceeding topology tolerance {tolerance_m} m"
    )]
    AttachmentCoordinateMismatch {
        node_id: u32,
        attachment_id: String,
        distance_m: f64,
        tolerance_m: f64,
    },
    #[error(
        "explicit nodes {first_node_id} and {second_node_id} both attach to the same solid-mesh node"
    )]
    DuplicateAttachment {
        first_node_id: u32,
        second_node_id: u32,
    },
    #[error("mixed structural topology identity encoding failed: {0}")]
    Identity(String),
}

fn attachment_matches(
    attachment: &runmat_analysis_core::ConformalSolidAttachment,
    candidate: &runmat_meshing_core::SolverMeshNode,
) -> bool {
    match attachment {
        runmat_analysis_core::ConformalSolidAttachment::GeometryVertex { entity } => {
            entity.kind == PersistentEntityKind::Vertex
                && candidate.provenance.iter().any(|value| value == entity)
        }
        runmat_analysis_core::ConformalSolidAttachment::SolverNode { stable_identity } => {
            candidate.stable_identity.0 == *stable_identity
        }
    }
}

fn attachment_label(attachment: &runmat_analysis_core::ConformalSolidAttachment) -> String {
    match attachment {
        runmat_analysis_core::ConformalSolidAttachment::GeometryVertex { entity } => entity
            .analysis_selection_id()
            .unwrap_or_else(|_| entity.source_topology_id.clone()),
        runmat_analysis_core::ConformalSolidAttachment::SolverNode { stable_identity } => {
            stable_identity
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        }
    }
}

fn identity(
    mesh_digest: StableDigest,
    structural_topology_digest: Digest,
    nodes: &[CanonicalStructuralNode],
    explicit_node_indices: &[usize],
) -> Result<Digest, StructuralNodeSpaceError> {
    encode_contract(
        IDENTITY_PREFIX,
        "mixed-structural-topology",
        &(
            mesh_digest,
            structural_topology_digest,
            nodes,
            explicit_node_indices,
        ),
        IDENTITY_LIMITS,
    )
    .map(Digest::sha256)
    .map_err(|error| StructuralNodeSpaceError::Identity(error.to_string()))
}

fn model_scale(mesh: &SolverMeshArtifact) -> f64 {
    let bounds = std::array::from_fn::<_, 3, _>(|axis| {
        mesh.topology.nodes.iter().fold(
            [f64::INFINITY, f64::NEG_INFINITY],
            |[minimum, maximum], node| {
                [
                    minimum.min(node.coordinates_m.canonical_value()[axis]),
                    maximum.max(node.coordinates_m.canonical_value()[axis]),
                ]
            },
        )
    });
    bounds
        .into_iter()
        .map(|[minimum, maximum]| (maximum - minimum).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests;
