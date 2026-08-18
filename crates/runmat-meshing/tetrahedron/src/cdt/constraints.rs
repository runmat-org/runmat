use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{MeshingStage, ProtectedBoundaryComplex, TopologyEntityId},
    MeshingCancellationSignal, StableDigest,
};
use runmat_meshing_plc::validate::validate_protected_boundary_complex;
use sha2::{Digest as _, Sha256};

use super::DelaunayVolumeNode;

const NODE_IDENTITY_DOMAIN: &[u8] = b"runmat-meshing-cdt-plc-node/v1\0";
const MAXIMUM_ENTITY_ID_BYTES: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayConstraintOptions {
    pub maximum_nodes: u64,
    pub maximum_segments: u64,
    pub maximum_facets: u64,
    pub cancellation_check_interval: u64,
}

impl Default for DelaunayConstraintOptions {
    fn default() -> Self {
        Self {
            maximum_nodes: 1_000_000_000,
            maximum_segments: 3_000_000_000,
            maximum_facets: 2_000_000_000,
            cancellation_check_interval: 1_024,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayConstraintNode {
    pub identity: StableDigest,
    pub source_node_id: TopologyEntityId,
    pub coordinates_m: [f64; 3],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayConstraintSegment {
    pub vertex_indices: [u32; 2],
    pub protected_edge_id: Option<TopologyEntityId>,
    pub source_edge_id: Option<TopologyEntityId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayConstraintFacet {
    pub facet_id: TopologyEntityId,
    pub vertex_indices: [u32; 3],
    pub source_face_id: TopologyEntityId,
    pub material_interface_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayConstraints {
    pub nodes: Vec<DelaunayConstraintNode>,
    pub segments: Vec<DelaunayConstraintSegment>,
    pub facets: Vec<DelaunayConstraintFacet>,
}

impl DelaunayConstraints {
    pub fn volume_nodes(&self) -> Vec<DelaunayVolumeNode> {
        self.nodes
            .iter()
            .map(|node| DelaunayVolumeNode {
                identity: node.identity,
                coordinates_m: node.coordinates_m,
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayConstraintErrorKind {
    InvalidOptions,
    InvalidPlc,
    InvalidIdentity,
    IdentityCollision,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayConstraintError {
    pub kind: DelaunayConstraintErrorKind,
    pub reason: String,
}

impl std::fmt::Display for DelaunayConstraintError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "3D Delaunay constraints {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunayConstraintError {}

pub fn build_delaunay_constraints(
    plc: &ProtectedBoundaryComplex,
    options: DelaunayConstraintOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayConstraints, DelaunayConstraintError> {
    validate_options(options)?;
    validate_protected_boundary_complex(plc).map_err(|validation| {
        error(
            DelaunayConstraintErrorKind::InvalidPlc,
            validation.to_string(),
        )
    })?;
    if plc.nodes.len() as u64 > options.maximum_nodes
        || plc.facets.len() as u64 > options.maximum_facets
    {
        return Err(resource(
            "PLC node or facet inventory exceeds its hard limit",
        ));
    }

    let mut nodes = Vec::with_capacity(plc.nodes.len());
    for (index, node) in plc.nodes.iter().enumerate() {
        checkpoint(index, options, cancellation)?;
        validate_entity_id(&node.node_id)?;
        nodes.push(DelaunayConstraintNode {
            identity: node_identity(&node.node_id),
            source_node_id: node.node_id.clone(),
            coordinates_m: node.coordinates_m,
        });
    }
    nodes.sort_by_key(|node| node.identity);
    if nodes
        .windows(2)
        .any(|pair| pair[0].identity == pair[1].identity)
    {
        return Err(error(
            DelaunayConstraintErrorKind::IdentityCollision,
            "distinct PLC nodes produced the same stable CDT identity",
        ));
    }
    let node_index = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.source_node_id.clone(), index as u32))
        .collect::<BTreeMap<_, _>>();

    let protected = plc
        .protected_edges
        .iter()
        .map(|edge| {
            validate_entity_id(&edge.edge_id)?;
            validate_entity_id(&edge.source_edge_id)?;
            let key =
                sorted_segment([node_index[&edge.node_ids[0]], node_index[&edge.node_ids[1]]]);
            Ok((key, (edge.edge_id.clone(), edge.source_edge_id.clone())))
        })
        .collect::<Result<BTreeMap<_, _>, DelaunayConstraintError>>()?;

    let mut segment_keys = BTreeSet::new();
    let mut facets = Vec::with_capacity(plc.facets.len());
    for (index, facet) in plc.facets.iter().enumerate() {
        checkpoint(index, options, cancellation)?;
        validate_entity_id(&facet.facet_id)?;
        validate_entity_id(&facet.source_face_id)?;
        for interface_id in &facet.material_interface_ids {
            validate_token("material interface", interface_id)?;
        }
        let vertex_indices = facet.node_ids.each_ref().map(|id| node_index[id]);
        for edge in 0..3 {
            segment_keys.insert(sorted_segment([
                vertex_indices[edge],
                vertex_indices[(edge + 1) % 3],
            ]));
        }
        let mut material_interface_ids = facet.material_interface_ids.clone();
        material_interface_ids.sort();
        facets.push(DelaunayConstraintFacet {
            facet_id: facet.facet_id.clone(),
            vertex_indices,
            source_face_id: facet.source_face_id.clone(),
            material_interface_ids,
        });
    }
    if segment_keys.len() as u64 > options.maximum_segments {
        return Err(resource("PLC segment inventory exceeds its hard limit"));
    }
    facets.sort_by_key(|facet| {
        let mut key = facet.vertex_indices;
        key.sort_unstable();
        key
    });
    let segments = segment_keys
        .into_iter()
        .map(|vertex_indices| {
            let provenance = protected.get(&vertex_indices);
            DelaunayConstraintSegment {
                vertex_indices,
                protected_edge_id: provenance.map(|(edge, _)| edge.clone()),
                source_edge_id: provenance.map(|(_, source)| source.clone()),
            }
        })
        .collect();
    Ok(DelaunayConstraints {
        nodes,
        segments,
        facets,
    })
}

fn node_identity(entity_id: &TopologyEntityId) -> StableDigest {
    let mut hasher = Sha256::new();
    hasher.update(NODE_IDENTITY_DOMAIN);
    hasher.update([stage_tag(entity_id.stage)]);
    hasher.update((entity_id.id.len() as u64).to_be_bytes());
    hasher.update(entity_id.id.as_bytes());
    StableDigest::from_bytes(hasher.finalize().into())
}

fn stage_tag(stage: MeshingStage) -> u8 {
    match stage {
        MeshingStage::CadTopology => 1,
        MeshingStage::Sizing => 2,
        MeshingStage::CurveMesh => 3,
        MeshingStage::SurfaceMesh => 4,
        MeshingStage::ProtectedBoundaryComplex => 5,
        MeshingStage::TetrahedronMesh => 6,
        MeshingStage::ConstraintRecovery => 7,
        MeshingStage::Optimization => 8,
        MeshingStage::SolveReadiness => 9,
    }
}

fn sorted_segment(mut vertices: [u32; 2]) -> [u32; 2] {
    vertices.sort_unstable();
    vertices
}

fn validate_options(options: DelaunayConstraintOptions) -> Result<(), DelaunayConstraintError> {
    if options.maximum_nodes == 0
        || options.maximum_segments == 0
        || options.maximum_facets == 0
        || options.cancellation_check_interval == 0
    {
        return Err(error(
            DelaunayConstraintErrorKind::InvalidOptions,
            "constraint inventory limits and cancellation interval must be nonzero",
        ));
    }
    Ok(())
}

fn validate_entity_id(entity_id: &TopologyEntityId) -> Result<(), DelaunayConstraintError> {
    validate_token("topology entity", &entity_id.id)
}

fn validate_token(field: &str, value: &str) -> Result<(), DelaunayConstraintError> {
    if value.is_empty()
        || value.len() > MAXIMUM_ENTITY_ID_BYTES
        || !value.is_ascii()
        || value.chars().any(char::is_control)
        || value.trim() != value
    {
        return Err(error(
            DelaunayConstraintErrorKind::InvalidIdentity,
            format!(
                "{field} identity must be 1..={MAXIMUM_ENTITY_ID_BYTES} printable ASCII bytes without surrounding whitespace"
            ),
        ));
    }
    Ok(())
}

fn checkpoint(
    index: usize,
    options: DelaunayConstraintOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayConstraintError> {
    if (index as u64).is_multiple_of(options.cancellation_check_interval)
        && cancellation.is_cancelled()
    {
        return Err(error(DelaunayConstraintErrorKind::Cancelled, "cancelled"));
    }
    Ok(())
}

fn resource(reason: impl Into<String>) -> DelaunayConstraintError {
    error(DelaunayConstraintErrorKind::ResourceLimit, reason)
}

fn error(kind: DelaunayConstraintErrorKind, reason: impl Into<String>) -> DelaunayConstraintError {
    DelaunayConstraintError {
        kind,
        reason: reason.into(),
    }
}

#[cfg(test)]
#[path = "constraints/tests.rs"]
mod tests;
