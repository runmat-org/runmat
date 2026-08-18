use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::DelaunayVolumeNode;

mod exact;
mod validation;

pub use exact::build_delaunay_constraints;
pub use validation::validate_delaunay_constraints;

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
    pub source_vertex_id: Option<PersistentEntityId>,
    pub coordinates_m: [f64; 3],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayConstraintSegment {
    pub vertex_indices: [u32; 2],
    /// Present exactly for segments on an authoritative exact curve.
    pub source_edge_id: Option<PersistentEntityId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DelaunayConstraintFacetSide {
    Region(PersistentEntityId),
    Exterior,
    Void,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayConstraintFacet {
    pub facet_id: StableDigest,
    pub vertex_indices: [u32; 3],
    pub source_face_id: PersistentEntityId,
    /// Side reached by an exact positive orientation against the oriented facet.
    pub positive_side: DelaunayConstraintFacetSide,
    /// Side reached by an exact negative orientation against the oriented facet.
    pub negative_side: DelaunayConstraintFacetSide,
    /// Canonical exact contact identities authored on this face, if any.
    pub contact_ids: Vec<PersistentEntityId>,
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
    InvalidGeometry,
    InvalidBoundary,
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

pub(super) fn sorted_segment(mut vertices: [u32; 2]) -> [u32; 2] {
    vertices.sort_unstable();
    vertices
}

pub(super) fn validate_options(
    options: DelaunayConstraintOptions,
) -> Result<(), DelaunayConstraintError> {
    if options.maximum_nodes == 0
        || options.maximum_nodes > u32::MAX as u64
        || options.maximum_segments == 0
        || options.maximum_facets == 0
        || options.cancellation_check_interval == 0
    {
        return Err(error(
            DelaunayConstraintErrorKind::InvalidOptions,
            "constraint inventory limits and cancellation interval must be nonzero, and nodes must fit the u32 topology index space",
        ));
    }
    Ok(())
}

pub(super) fn checkpoint(
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

pub(super) fn resource(reason: impl Into<String>) -> DelaunayConstraintError {
    error(DelaunayConstraintErrorKind::ResourceLimit, reason)
}

pub(super) fn error(
    kind: DelaunayConstraintErrorKind,
    reason: impl Into<String>,
) -> DelaunayConstraintError {
    DelaunayConstraintError {
        kind,
        reason: reason.into(),
    }
}

#[cfg(test)]
#[path = "constraints/tests.rs"]
mod tests;
