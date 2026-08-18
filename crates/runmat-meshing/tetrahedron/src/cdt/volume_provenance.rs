use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{volume_quality::context::DelaunayVolumeMetricContext, DelaunayVolumeTopology};

mod construction;
mod derivation;
mod validation;

pub use construction::{
    build_delaunay_volume_provenance, validate_delaunay_volume_provenance_sources,
};
pub(super) use derivation::derive_delaunay_volume_metric_contexts;
pub use validation::validate_delaunay_volume_provenance;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayNodeProvenance {
    pub node_identity: StableDigest,
    pub entity_ids: Vec<PersistentEntityId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunaySegmentProvenance {
    pub node_identities: [StableDigest; 2],
    pub entity_ids: Vec<PersistentEntityId>,
    /// Exact source-edge parameters aligned with the canonical node identities.
    pub edge_parameters: [f64; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayFacetProvenance {
    pub node_identities: [StableDigest; 3],
    pub chart_id: StableDigest,
    pub entity_ids: Vec<PersistentEntityId>,
    pub region_ids: Vec<PersistentEntityId>,
}

/// Canonical persistent incidence for protected PLC simplices. Per-tetrahedron metric context is
/// derived from this authority after every topology mutation; it is never copied heuristically.
#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayVolumeProvenance {
    pub nodes: Vec<DelaunayNodeProvenance>,
    pub segments: Vec<DelaunaySegmentProvenance>,
    pub facets: Vec<DelaunayFacetProvenance>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeProvenanceOptions {
    pub maximum_node_bindings: u64,
    pub maximum_segment_bindings: u64,
    pub maximum_facet_bindings: u64,
    pub cancellation_check_interval: u64,
}

impl Default for DelaunayVolumeProvenanceOptions {
    fn default() -> Self {
        Self {
            maximum_node_bindings: 1_000_000_000,
            maximum_segment_bindings: 3_000_000_000,
            maximum_facet_bindings: 2_000_000_000,
            cancellation_check_interval: 1_024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayVolumeProvenanceErrorKind {
    InvalidOptions,
    InvalidTopology,
    InvalidProvenance,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeProvenanceError {
    pub kind: DelaunayVolumeProvenanceErrorKind,
    pub reason: String,
}

impl std::fmt::Display for DelaunayVolumeProvenanceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "3D Delaunay volume provenance {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunayVolumeProvenanceError {}

pub(super) fn validate_options(
    options: DelaunayVolumeProvenanceOptions,
) -> Result<(), DelaunayVolumeProvenanceError> {
    if options.maximum_node_bindings == 0
        || options.maximum_segment_bindings == 0
        || options.maximum_facet_bindings == 0
        || options.cancellation_check_interval == 0
    {
        return Err(error(
            DelaunayVolumeProvenanceErrorKind::InvalidOptions,
            "provenance inventory limits and cancellation interval must be nonzero",
        ));
    }
    Ok(())
}

pub(super) fn checkpoint(
    step: u64,
    options: DelaunayVolumeProvenanceOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeProvenanceError> {
    if step.is_multiple_of(options.cancellation_check_interval) && cancellation.is_cancelled() {
        return Err(error(
            DelaunayVolumeProvenanceErrorKind::Cancelled,
            "cancelled",
        ));
    }
    Ok(())
}

pub(super) fn error(
    kind: DelaunayVolumeProvenanceErrorKind,
    reason: impl Into<String>,
) -> DelaunayVolumeProvenanceError {
    DelaunayVolumeProvenanceError {
        kind,
        reason: reason.into(),
    }
}

#[cfg(test)]
#[path = "volume_provenance/tests.rs"]
mod tests;
