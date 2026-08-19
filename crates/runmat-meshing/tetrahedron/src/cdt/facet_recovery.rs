use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    DelaunayConstraints, DelaunaySegmentRecovery, DelaunaySegmentRecoveryError,
    DelaunaySegmentRecoveryErrorKind, DelaunaySegmentRecoveryOptions, DelaunayVolumeTopology,
};

mod cavity;
mod flip;
mod support;
mod validation;
mod work;

use cavity::try_recover_facet_with_edge_star_cavity;
use flip::try_recover_facet_with_edge_flip;
use support::facet_support;
pub use validation::validate_delaunay_facet_recovery;
use validation::{face_exists, validate_inputs};
use work::FacetRecoveryWork;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayFacetRecoveryOptions {
    pub segment_recovery: DelaunaySegmentRecoveryOptions,
    pub maximum_search_steps: u64,
    pub maximum_flip_attempts: u64,
    pub maximum_support_steps: u64,
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

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayFacetRecovery {
    pub segment_recovery: DelaunaySegmentRecovery,
    pub facets: Vec<DelaunayRecoveredFacet>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayFacetRecoveryErrorKind {
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
    mut segment_recovery: DelaunaySegmentRecovery,
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
    let mut work = FacetRecoveryWork::new(options, cancellation);
    let mut facets = Vec::with_capacity(constraints.facets.len());
    let mut protected_triangles = Vec::new();
    for constraint_index in 0..constraints.facets.len() {
        let support = facet_support(
            &segment_recovery,
            constraints,
            constraint_index as u32,
            &mut work,
        )?;
        for triangle in &support {
            if !face_exists(
                &segment_recovery.topology,
                triangle.node_identities,
                constraint_index as u32,
                &mut work,
            )? {
                let updated = if let Some(updated) = try_recover_facet_with_edge_flip(
                    &segment_recovery,
                    triangle.node_identities,
                    &protected_triangles,
                    constraint_index as u32,
                    &mut work,
                )? {
                    updated
                } else if let Some(updated) = try_recover_facet_with_edge_star_cavity(
                    &segment_recovery,
                    triangle.node_identities,
                    &protected_triangles,
                    constraint_index as u32,
                    &mut work,
                )? {
                    updated
                } else {
                    return Err(error(
                        DelaunayFacetRecoveryErrorKind::UnsatisfiableConstraint,
                        Some(constraint_index as u32),
                        "facet support is absent and no legal protected flip or edge-star cavity recovers it",
                    ));
                };
                segment_recovery.topology = updated;
            }
            protected_triangles.push(*triangle);
        }
        facets.push(DelaunayRecoveredFacet {
            constraint_index: constraint_index as u32,
            triangles: support,
        });
    }
    let recovery = DelaunayFacetRecovery {
        segment_recovery,
        facets,
    };
    validate_delaunay_facet_recovery(&recovery, constraints, options, cancellation)?;
    Ok(recovery)
}

fn node_index(topology: &DelaunayVolumeTopology, identity: StableDigest) -> Option<usize> {
    topology
        .nodes
        .binary_search_by_key(&identity, |node| node.identity)
        .ok()
}

fn validate_options(
    options: DelaunayFacetRecoveryOptions,
) -> Result<(), DelaunayFacetRecoveryError> {
    if options.maximum_search_steps == 0
        || options.maximum_flip_attempts == 0
        || options.maximum_support_steps == 0
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
