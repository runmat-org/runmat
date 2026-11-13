//! Canonical explicit-topology identity and its shared allocation estimate. Borrowed contracts
//! preserve the established bytes without cloning topology, connectivity, or attachments.
use crate::topology::storage::{self, TopologyStorageError};

use runmat_analysis_core::{
    ConformalSolidAttachment, StructuralElement, StructuralElementKind, StructuralModel,
    StructuralNode,
};
use runmat_canonical_codec::{
    encode_contract, serialized_materialization_resident_bytes_estimate, CanonicalLimits,
};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Serialize, Serializer};
use thiserror::Error;

const PREFIX: &[u8] = b"runmat-fea-explicit-structural-topology/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::platform_content(4_000_000, 64);
const PROFILE_LIMITS: CanonicalLimits =
    CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);

#[derive(Serialize)]
struct TopologyContract<'a> {
    #[serde(serialize_with = "serialize_nodes")]
    nodes: &'a [StructuralNode],
    #[serde(serialize_with = "serialize_elements")]
    elements: &'a [StructuralElement],
}
#[derive(Serialize)]
struct NodeContract<'a> {
    node_id: u32,
    // Canonical identity encoding of checked physical positions, not a numeric quantity.
    #[serde(rename = "coordinates_m")]
    coordinate_bits: [u64; 3],
    solid_attachment: Option<&'a ConformalSolidAttachment>,
}
#[derive(Serialize)]
struct ElementContract<'a> {
    element_id: &'a str,
    region_id: &'a str,
    node_ids: &'a [u32],
}
fn serialize_nodes<S: Serializer>(
    nodes: &[StructuralNode],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(nodes.iter().map(|node| NodeContract {
        node_id: node.node_id,
        coordinate_bits: node.coordinates_m.canonical_value().map(f64::to_bits),
        solid_attachment: node.solid_attachment.as_ref(),
    }))
}
fn serialize_elements<S: Serializer>(
    elements: &[StructuralElement],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(elements.iter().map(|element| ElementContract {
        element_id: &element.element_id,
        region_id: &element.region_id,
        node_ids: node_ids(&element.kind),
    }))
}
fn node_ids(kind: &StructuralElementKind) -> &[u32] {
    match kind {
        StructuralElementKind::Beam(beam) => &beam.node_ids,
        StructuralElementKind::Shell(shell) => &shell.node_ids,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum ExplicitStructuralTopologyError {
    #[error(transparent)]
    Storage(#[from] TopologyStorageError),
    #[error("explicit structural node id {node_id} is duplicated")]
    DuplicateNode { node_id: u32 },
    #[error("explicit structural node {node_id} has non-finite coordinates")]
    NonFiniteNode { node_id: u32 },
    #[error("explicit structural element id `{element_id}` is duplicated")]
    DuplicateElement { element_id: String },
    #[error("explicit structural element `{element_id}` references absent node {node_id}")]
    MissingElementNode { element_id: String, node_id: u32 },
    #[error("explicit structural topology identity encoding failed: {0}")]
    Identity(String),
}

fn validate(structural: &StructuralModel) -> Result<(), ExplicitStructuralTopologyError> {
    let mut nodes = storage::reserve(structural.nodes.len())?;
    nodes.extend(
        structural
            .nodes
            .iter()
            .enumerate()
            .map(|(ordinal, node)| (node.node_id, ordinal)),
    );
    nodes.sort_unstable();
    for (ordinal, node) in structural.nodes.iter().enumerate() {
        storage::check_cancelled()?;
        let first = nodes.partition_point(|(id, _)| *id < node.node_id);
        if nodes[first].1 < ordinal {
            return Err(ExplicitStructuralTopologyError::DuplicateNode {
                node_id: node.node_id,
            });
        }
        if node
            .coordinates_m
            .canonical_value()
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(ExplicitStructuralTopologyError::NonFiniteNode {
                node_id: node.node_id,
            });
        }
    }
    let mut elements = storage::reserve(structural.elements.len())?;
    elements.extend(
        structural
            .elements
            .iter()
            .enumerate()
            .map(|(ordinal, element)| (element.element_id.as_str(), ordinal)),
    );
    elements.sort_unstable();
    for (ordinal, element) in structural.elements.iter().enumerate() {
        storage::check_cancelled()?;
        let first = elements.partition_point(|(id, _)| *id < element.element_id.as_str());
        if elements[first].1 < ordinal {
            return Err(ExplicitStructuralTopologyError::DuplicateElement {
                element_id: element.element_id.clone(),
            });
        }
        if let Some(node_id) = node_ids(&element.kind)
            .iter()
            .find(|id| nodes.binary_search_by_key(id, |(key, _)| key).is_err())
        {
            return Err(ExplicitStructuralTopologyError::MissingElementNode {
                element_id: element.element_id.clone(),
                node_id: *node_id,
            });
        }
    }
    Ok(()) // Both validation indexes are released before canonical tree/byte materialization.
}

pub(crate) fn explicit_topology_digest(
    structural: &StructuralModel,
) -> Result<Digest, ExplicitStructuralTopologyError> {
    validate(structural)?;
    encode_contract(
        PREFIX,
        "explicit-structural-topology",
        &TopologyContract {
            nodes: &structural.nodes,
            elements: &structural.elements,
        },
        LIMITS,
    )
    .map(Digest::sha256)
    .map_err(identity_error)
}

pub(crate) fn explicit_node_identity(
    structural_topology_digest: Digest,
    node_id: u32,
    coordinates_m: [f64; 3],
) -> Result<StableDigest, ExplicitStructuralTopologyError> {
    encode_contract(
        PREFIX,
        "explicit-structural-node",
        &(
            structural_topology_digest,
            node_id,
            coordinates_m.map(f64::to_bits),
        ),
        LIMITS,
    )
    .map(stable_hash)
    .map_err(identity_error)
}

pub(crate) fn explicit_element_identity(
    structural_topology_digest: Digest,
    element_id: &str,
    region_id: &str,
    kind: &StructuralElementKind,
) -> Result<StableDigest, ExplicitStructuralTopologyError> {
    encode_contract(
        PREFIX,
        "explicit-structural-element",
        &(
            structural_topology_digest,
            element_id,
            region_id,
            node_ids(kind),
        ),
        LIMITS,
    )
    .map(stable_hash)
    .map_err(identity_error)
}

/// Conservative shared-codec requested-heap estimate before validation or encoding. Encoding
/// retains its actual canonical limits; the profiler must not reject a valid payload merely
/// because its encoded-size upper bound exceeds the canonical byte limit. Does not include
/// caller-owned input or a retained per-element identity cache, allocator bookkeeping, or RSS.
pub(crate) fn explicit_identity_resource_bytes(
    structural: &StructuralModel,
) -> Result<u64, ExplicitStructuralTopologyError> {
    let topology = serialized_materialization_resident_bytes_estimate(
        &TopologyContract {
            nodes: &structural.nodes,
            elements: &structural.elements,
        },
        PROFILE_LIMITS,
    )
    .map_err(identity_error)?;
    let mut peak = topology;
    for element in &structural.elements {
        storage::check_cancelled()?;
        // Digest's serialized width is fixed. This placeholder is used only for counting;
        // it is never published or used to construct a topology/element identity.
        let bytes = serialized_materialization_resident_bytes_estimate(
            &(
                Digest::from_bytes([0; 32]),
                element.element_id.as_str(),
                element.region_id.as_str(),
                node_ids(&element.kind),
            ),
            PROFILE_LIMITS,
        )
        .map_err(identity_error)?;
        peak = peak.max(bytes);
    }
    let validation = storage::add(
        storage::slots::<(u32, usize)>(structural.nodes.len())?,
        storage::slots::<(&str, usize)>(structural.elements.len())?,
    )?;
    Ok(peak.max(validation))
}

fn identity_error(error: impl std::fmt::Display) -> ExplicitStructuralTopologyError {
    ExplicitStructuralTopologyError::Identity(error.to_string())
}
fn stable_hash(encoded: Vec<u8>) -> StableDigest {
    StableDigest::from_bytes(*Digest::sha256(encoded).bytes())
}

#[cfg(test)]
mod tests;
