use std::collections::VecDeque;

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use serde::{Deserialize, Serialize};

use super::{
    DelaunayConstraints, DelaunaySegmentRecovery, DelaunaySegmentRecoveryError,
    DelaunaySegmentRecoveryErrorKind, DelaunaySegmentRecoveryOptions, DelaunayVolumeTopology,
};

mod cavity;
mod conforming_split;
mod execution;
mod facet_delaunay;
mod flip;
mod protection;
mod support;
mod support_refinement;
mod validation;
mod work;
mod working;
use crate::cdt::insertion::topology_access::TetrahedralTopology;
pub(super) use execution::construct_delaunay_facet_recovery;
use execution::support_edge_exists;
use protection::FacetProtection;
use working::FacetRecoveryWorking;

use cavity::{
    try_recover_facet_patch_in_place, try_recover_facet_with_cavity,
    try_recover_facet_with_cavity_in_place,
};
use flip::try_recover_facet_with_local_edge_flip;
use support::facet_support_on;
use support_refinement::{
    ensure_support_triangle_edges, ensure_support_triangle_edges_in_place, interior_support_node,
    recovered_segment_edges, refine_support_triangle_interior,
    refine_support_triangle_interior_in_place, split_segment_encroached_by_support_edge,
    triangle_has_positive_area_on,
    PendingSupportTriangle, SupportRefinementContext, SupportTriangleEdgeState,
};
pub use validation::validate_delaunay_facet_recovery;
use validation::{face_exists, validate_inputs};
use work::FacetRecoveryWork;

pub(in crate::cdt) fn facet_steiner_identity(
    support: [StableDigest; 3],
    insertion_round: u64,
    candidate_rank: u64,
) -> StableDigest {
    cavity::steiner_identity(support, insertion_round, candidate_rank)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayFacetRecoveryOptions {
    pub segment_recovery: DelaunaySegmentRecoveryOptions,
    pub maximum_search_steps: u64,
    pub maximum_flip_attempts: u64,
    pub maximum_support_steps: u64,
    pub maximum_pending_support_triangles: u64,
    pub maximum_cavity_steps: u64,
    pub maximum_cavity_tetrahedra: u64,
    pub maximum_cavity_nodes: u64,
    pub maximum_cavity_boundary_faces: u64,
    pub maximum_cavity_apex_attempts: u64,
    pub maximum_cavity_candidate_tetrahedra: u64,
    pub maximum_cavity_candidate_evaluations: u64,
    pub maximum_cavity_exact_cover_attempts: u64,
    pub maximum_cavity_expansion_rounds: u32,
    pub maximum_cavity_steiner_nodes: u64,
    pub maximum_cavity_steiner_candidates: u64,
    pub maximum_cavity_steiner_candidate_evaluations_per_round: u64,
}

impl Default for DelaunayFacetRecoveryOptions {
    fn default() -> Self {
        Self {
            segment_recovery: DelaunaySegmentRecoveryOptions::default(),
            maximum_search_steps: 1_000_000_000,
            maximum_flip_attempts: 100_000_000,
            maximum_support_steps: 100_000_000,
            maximum_pending_support_triangles: 100_000,
            maximum_cavity_steps: 100_000_000,
            maximum_cavity_tetrahedra: 10_000,
            maximum_cavity_nodes: 10_000,
            maximum_cavity_boundary_faces: 20_000,
            maximum_cavity_apex_attempts: 20_000,
            maximum_cavity_candidate_tetrahedra: 1_000_000,
            maximum_cavity_candidate_evaluations: 100_000_000,
            maximum_cavity_exact_cover_attempts: 1_000_000,
            maximum_cavity_expansion_rounds: 1_024,
            maximum_cavity_steiner_nodes: 64,
            maximum_cavity_steiner_candidates: 4_096,
            maximum_cavity_steiner_candidate_evaluations_per_round: 1_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayRecoveredFacetTriangle {
    pub node_identities: [StableDigest; 3],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayRecoveredFacet {
    pub constraint_index: u32,
    pub triangles: Vec<DelaunayRecoveredFacetTriangle>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayFacetSteinerInsertion {
    pub constraint_index: u32,
    pub support_node_identities: [StableDigest; 3],
    pub insertion_round: u64,
    pub candidate_rank: u64,
    pub node_identity: StableDigest,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayFacetRecovery {
    /// Immutable prerequisite used to replay this stage independently.
    pub segment_recovery: DelaunaySegmentRecovery,
    /// Final recovered segment chains after conforming facet recovery split any encroached edge.
    pub recovered_segments: Vec<super::DelaunayRecoveredSegment>,
    /// Canonical topology after every recovered facet has been inserted.
    pub topology: DelaunayVolumeTopology,
    pub facets: Vec<DelaunayRecoveredFacet>,
    pub steiner_insertions: Vec<DelaunayFacetSteinerInsertion>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayFacetRecoveryErrorKind {
    Accounting(runmat_simulation_usage::LedgerError),
    InvalidOptions,
    InvalidConstraints,
    InvalidTopology,
    UnsatisfiableConstraint,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayFacetRecoveryError {
    pub kind: DelaunayFacetRecoveryErrorKind,
    pub constraint_index: Option<u32>,
    pub reason: String,
}

impl std::fmt::Display for DelaunayFacetRecoveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "3D Delaunay facet recovery {:?} at constraint {:?}: {}",
            self.kind, self.constraint_index, self.reason
        )
    }
}

impl std::error::Error for DelaunayFacetRecoveryError {}

pub fn recover_delaunay_facets(
    segment_recovery: DelaunaySegmentRecovery,
    constraints: &DelaunayConstraints,
    options: DelaunayFacetRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayFacetRecovery, DelaunayFacetRecoveryError> {
    validate_options(options)?;
    validate_inputs(&segment_recovery, constraints, options, cancellation)?;
    if !segment_recovery.topology.incidence.regions.is_empty() {
        return Err(error(
            DelaunayFacetRecoveryErrorKind::InvalidTopology,
            None,
            "facet recovery must precede region assignment",
        ));
    }
    let recovery =
        construct_delaunay_facet_recovery(segment_recovery, constraints, options, cancellation)?;
    validate_delaunay_facet_recovery(&recovery, constraints, options, cancellation)?;
    Ok(recovery)
}

fn triangle_edges(triangle: [StableDigest; 3]) -> [[StableDigest; 2]; 3] {
    [
        [triangle[0], triangle[1]],
        [triangle[1], triangle[2]],
        [triangle[2], triangle[0]],
    ]
}

fn node_index(
    topology: &impl crate::cdt::insertion::topology_access::TetrahedralTopology,
    identity: StableDigest,
) -> Option<usize> {
    topology.node_index(identity).map(|index| index as usize)
}

fn validate_options(
    options: DelaunayFacetRecoveryOptions,
) -> Result<(), DelaunayFacetRecoveryError> {
    if options.maximum_search_steps == 0
        || options.maximum_flip_attempts == 0
        || options.maximum_support_steps == 0
        || options.maximum_pending_support_triangles == 0
        || options.maximum_cavity_steps == 0
        || options.maximum_cavity_tetrahedra == 0
        || options.maximum_cavity_nodes == 0
        || options.maximum_cavity_boundary_faces == 0
        || options.maximum_cavity_apex_attempts == 0
        || options.maximum_cavity_candidate_tetrahedra == 0
        || options.maximum_cavity_candidate_evaluations == 0
        || options.maximum_cavity_exact_cover_attempts == 0
        || options.maximum_cavity_expansion_rounds == 0
        || options.maximum_cavity_steiner_nodes == 0
        || options.maximum_cavity_steiner_candidates == 0
        || options.maximum_cavity_steiner_candidate_evaluations_per_round == 0
    {
        return Err(error(
            DelaunayFacetRecoveryErrorKind::InvalidOptions,
            None,
            "facet recovery limits must be nonzero",
        ));
    }
    Ok(())
}

fn ensure_pending_capacity(
    pending: &VecDeque<PendingSupportTriangle>,
    additional: usize,
    maximum: u64,
    constraint_index: u32,
) -> Result<(), DelaunayFacetRecoveryError> {
    if (pending.len() as u64).saturating_add(additional as u64) > maximum {
        return Err(resource(
            constraint_index,
            "facet support queue exceeded its configured triangle limit",
        ));
    }
    Ok(())
}

fn segment_error(error_value: DelaunaySegmentRecoveryError) -> DelaunayFacetRecoveryError {
    let kind = match error_value.kind {
        DelaunaySegmentRecoveryErrorKind::InvalidOptions => {
            DelaunayFacetRecoveryErrorKind::InvalidOptions
        }
        DelaunaySegmentRecoveryErrorKind::ResourceLimit => {
            DelaunayFacetRecoveryErrorKind::ResourceLimit
        }
        DelaunaySegmentRecoveryErrorKind::Cancelled => DelaunayFacetRecoveryErrorKind::Cancelled,
        DelaunaySegmentRecoveryErrorKind::InvalidConstraints
        | DelaunaySegmentRecoveryErrorKind::IdentityCollision
        | DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint => {
            DelaunayFacetRecoveryErrorKind::InvalidConstraints
        }
        DelaunaySegmentRecoveryErrorKind::InvalidTopology => {
            DelaunayFacetRecoveryErrorKind::InvalidTopology
        }
    };
    error(kind, error_value.constraint_index, error_value.to_string())
}

fn resource(constraint_index: u32, reason: &'static str) -> DelaunayFacetRecoveryError {
    error(
        DelaunayFacetRecoveryErrorKind::ResourceLimit,
        Some(constraint_index),
        reason,
    )
}

fn error(
    kind: DelaunayFacetRecoveryErrorKind,
    constraint_index: Option<u32>,
    reason: impl Into<String>,
) -> DelaunayFacetRecoveryError {
    DelaunayFacetRecoveryError {
        kind,
        constraint_index,
        reason: reason.into(),
    }
}

#[cfg(test)]
#[path = "facet_recovery/tests.rs"]
mod tests;
