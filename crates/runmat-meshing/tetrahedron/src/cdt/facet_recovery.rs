use std::collections::VecDeque;

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use serde::{Deserialize, Serialize};

use super::{
    DelaunayConstraints, DelaunaySegmentRecovery, DelaunaySegmentRecoveryError,
    DelaunaySegmentRecoveryErrorKind, DelaunaySegmentRecoveryOptions, DelaunayVolumeTopology,
};

mod cavity;
mod conforming_split;
mod flip;
mod support;
mod support_refinement;
mod validation;
mod work;

use cavity::{try_recover_facet_patch_with_cavity, try_recover_facet_with_cavity};
use conforming_split::apply_protected_facet_splits;
use flip::try_recover_facet_with_edge_flip;
use support::facet_support;
use support_refinement::{
    ensure_support_triangle_edges, recovered_segment_edges, refine_support_triangle_interior,
    triangle_has_positive_area, PendingSupportTriangle, SupportRefinementContext,
    SupportTriangleEdgeState,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

pub(super) fn construct_delaunay_facet_recovery(
    segment_recovery: DelaunaySegmentRecovery,
    constraints: &DelaunayConstraints,
    options: DelaunayFacetRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayFacetRecovery, DelaunayFacetRecoveryError> {
    let mut working = segment_recovery.clone();
    let mut work = FacetRecoveryWork::new(options, cancellation);
    let mut facets = Vec::with_capacity(constraints.facets.len());
    let mut protected_triangles = Vec::new();
    let mut protected_faces = Vec::new();
    let mut protected_edges = recovered_segment_edges(&working);
    let mut steiner_insertions = Vec::new();
    for constraint_index in 0..constraints.facets.len() {
        let progress_span = constraint_index.is_multiple_of(64).then(|| {
            tracing::info_span!(
                "meshing.tetrahedron.facet_recovery.progress",
                constraint_index,
                facet_count = constraints.facets.len()
            )
        });
        let _progress_guard = progress_span.as_ref().map(|span| span.enter());
        work.begin_constraint();
        // Recovery mutates the tetrahedral topology. Deriving every support up front would leave
        // later facets holding diagonals that an earlier recovery may legitimately remove.
        let initial_support =
            facet_support(&working, constraints, constraint_index as u32, &mut work)?;
        let support_is_topological = initial_support.iter().all(|triangle| {
            triangle_edges(triangle.node_identities)
                .into_iter()
                .all(|edge| support_edge_exists(&working.topology, edge))
        });
        if !support_is_topological {
            let footprint = constraints.facets[constraint_index]
                .vertex_indices
                .map(|index| constraints.nodes[index as usize].identity);
            if let Some(topology) = try_recover_facet_patch_with_cavity(
                &working,
                footprint,
                &initial_support,
                &protected_triangles,
                constraint_index as u32,
                &mut work,
            )? {
                working.topology = topology;
                for triangle in &initial_support {
                    if !protected_triangles.contains(triangle) {
                        protected_triangles.push(*triangle);
                    }
                }
                protected_faces = canonical_protected_faces(&protected_triangles);
                facets.push(DelaunayRecoveredFacet {
                    constraint_index: constraint_index as u32,
                    triangles: initial_support,
                });
                continue;
            }
        }
        let mut pending = initial_support
            .into_iter()
            .map(|triangle| PendingSupportTriangle {
                lineage_support: triangle.node_identities,
                triangle,
            })
            .collect::<VecDeque<_>>();
        if pending.len() as u64 > options.maximum_pending_support_triangles {
            return Err(resource(
                constraint_index as u32,
                "facet support queue exceeded its configured triangle limit",
            ));
        }
        let mut support = Vec::new();
        let mut refinement_rank = 0_u64;
        while let Some(pending_triangle) = pending.pop_front() {
            if !triangle_has_positive_area(
                &working,
                pending_triangle.triangle.node_identities,
                constraint_index as u32,
            )? {
                continue;
            }
            work.clear_rejection();
            let mut refinement_rejection = None;
            match ensure_support_triangle_edges(
                &mut working,
                &pending_triangle,
                SupportRefinementContext {
                    protected_triangles: &protected_triangles,
                    protected_faces: &protected_faces,
                    protected_edges: &protected_edges,
                    constraint_index: constraint_index as u32,
                    refinement_rank,
                },
                &mut work,
            )? {
                SupportTriangleEdgeState::Ready => {}
                SupportTriangleEdgeState::Unavailable => {
                    refinement_rejection = work.last_rejection().map(str::to_owned);
                }
                SupportTriangleEdgeState::Split {
                    triangles,
                    insertions,
                    protected_splits,
                } => {
                    tracing::info!(
                        event_code = "meshing.tetrahedron.facet_recovery.support_split",
                        constraint_index,
                        refinement_rank,
                        child_count = triangles.len(),
                        inserted_node_count = insertions.len(),
                        "facet support split"
                    );
                    let protected_changed = !protected_splits.is_empty();
                    apply_protected_facet_splits(
                        &mut facets,
                        &mut protected_triangles,
                        protected_splits,
                    );
                    if protected_changed {
                        protected_faces = canonical_protected_faces(&protected_triangles);
                    }
                    refinement_rank += 1;
                    steiner_insertions.extend(insertions);
                    ensure_pending_capacity(
                        &pending,
                        triangles.len(),
                        options.maximum_pending_support_triangles,
                        constraint_index as u32,
                    )?;
                    for triangle in triangles.into_iter().rev() {
                        pending.push_front(triangle);
                    }
                    continue;
                }
                SupportTriangleEdgeState::RefinedSegment { protected_splits } => {
                    let split_span = tracing::info_span!(
                        "meshing.tetrahedron.facet_recovery.segment_split",
                        constraint_index,
                        refined_segment_node_count = working.steiner_node_identities.len(),
                        protected_facet_split_count = protected_splits.len(),
                    );
                    let _split_guard = split_span.enter();
                    tracing::info!(
                        event_code = "meshing.tetrahedron.facet_recovery.segment_split",
                        constraint_index,
                        refined_segment_node_count = working.steiner_node_identities.len(),
                        protected_facet_split_count = protected_splits.len(),
                        "facet recovery split a protected segment"
                    );
                    let protected_changed = !protected_splits.is_empty();
                    apply_protected_facet_splits(
                        &mut facets,
                        &mut protected_triangles,
                        protected_splits,
                    );
                    if protected_changed {
                        protected_faces = canonical_protected_faces(&protected_triangles);
                    }
                    pending.push_front(pending_triangle);
                    protected_edges = recovered_segment_edges(&working);
                    continue;
                }
            }
            let triangle = pending_triangle.triangle;
            if !triangle_has_positive_area(
                &working,
                triangle.node_identities,
                constraint_index as u32,
            )? {
                continue;
            }
            if !face_exists(
                &working.topology,
                triangle.node_identities,
                constraint_index as u32,
                &mut work,
            )? {
                let updated = if let Some(updated) = try_recover_facet_with_edge_flip(
                    &working,
                    triangle.node_identities,
                    &protected_triangles,
                    constraint_index as u32,
                    &mut work,
                )? {
                    updated
                } else if let Some(updated) = try_recover_facet_with_cavity(
                    &working,
                    triangle.node_identities,
                    &protected_triangles,
                    constraint_index as u32,
                    &mut work,
                )? {
                    steiner_insertions.extend_from_slice(&updated.steiner_insertions);
                    updated.topology
                } else {
                    let recovery_rejection = work.last_rejection().map(str::to_owned);
                    match refine_support_triangle_interior(
                        &mut working,
                        &pending_triangle,
                        SupportRefinementContext {
                            protected_triangles: &protected_triangles,
                            protected_faces: &protected_faces,
                            protected_edges: &protected_edges,
                            constraint_index: constraint_index as u32,
                            refinement_rank,
                        },
                        &mut work,
                    )? {
                        SupportTriangleEdgeState::Split {
                            triangles,
                            insertions,
                            protected_splits,
                        } => {
                            tracing::info!(
                                event_code = "meshing.tetrahedron.facet_recovery.interior_split",
                                constraint_index,
                                refinement_rank,
                                child_count = triangles.len(),
                                inserted_node_count = insertions.len(),
                                "facet interior split"
                            );
                            let protected_changed = !protected_splits.is_empty();
                            apply_protected_facet_splits(
                                &mut facets,
                                &mut protected_triangles,
                                protected_splits,
                            );
                            if protected_changed {
                                protected_faces = canonical_protected_faces(&protected_triangles);
                            }
                            refinement_rank += 1;
                            steiner_insertions.extend(insertions);
                            ensure_pending_capacity(
                                &pending,
                                triangles.len(),
                                options.maximum_pending_support_triangles,
                                constraint_index as u32,
                            )?;
                            for triangle in triangles.into_iter().rev() {
                                pending.push_front(triangle);
                            }
                            continue;
                        }
                        SupportTriangleEdgeState::RefinedSegment { protected_splits } => {
                            tracing::info!(
                                event_code = "meshing.tetrahedron.facet_recovery.segment_split",
                                constraint_index,
                                refined_segment_node_count = working.steiner_node_identities.len(),
                                protected_facet_split_count = protected_splits.len(),
                                "facet recovery split a protected segment"
                            );
                            let protected_changed = !protected_splits.is_empty();
                            apply_protected_facet_splits(
                                &mut facets,
                                &mut protected_triangles,
                                protected_splits,
                            );
                            if protected_changed {
                                protected_faces = canonical_protected_faces(&protected_triangles);
                            }
                            pending.push_front(pending_triangle);
                            protected_edges = recovered_segment_edges(&working);
                            continue;
                        }
                        SupportTriangleEdgeState::Ready | SupportTriangleEdgeState::Unavailable => {
                        }
                    }
                    let recovery_rejection =
                        work.last_rejection()
                            .unwrap_or(recovery_rejection.as_deref().unwrap_or(
                            "neither the protected flip nor cavity recovery admitted a candidate",
                        ));
                    let detail = refinement_rejection.map_or_else(
                        || recovery_rejection.to_owned(),
                        |refinement| {
                            format!(
                                "support refinement rejected: {refinement}; facet recovery rejected: {recovery_rejection}"
                            )
                        },
                    );
                    let coordinates = triangle.node_identities.map(|identity| {
                        node_index(&working.topology, identity)
                            .map(|index| working.topology.nodes[index].coordinates_m)
                    });
                    return Err(error(
                        DelaunayFacetRecoveryErrorKind::UnsatisfiableConstraint,
                        Some(constraint_index as u32),
                        format!(
                            "facet support {:?} at {coordinates:?} is absent and recovery rejected every candidate: {detail}",
                            triangle.node_identities,
                        ),
                    ));
                };
                working.topology = updated;
            }
            if !protected_triangles.contains(&triangle) {
                protected_triangles.push(triangle);
                let mut face = triangle.node_identities;
                face.sort_unstable();
                match protected_faces.binary_search(&face) {
                    Ok(_) => {}
                    Err(index) => protected_faces.insert(index, face),
                }
            }
            support.push(triangle);
        }
        facets.push(DelaunayRecoveredFacet {
            constraint_index: constraint_index as u32,
            triangles: support,
        });
    }
    steiner_insertions.sort_by_key(|insertion| {
        (
            insertion.constraint_index,
            insertion.support_node_identities,
            insertion.insertion_round,
            insertion.candidate_rank,
            insertion.node_identity,
        )
    });
    let recovery = DelaunayFacetRecovery {
        segment_recovery,
        recovered_segments: working.segments,
        topology: working.topology,
        facets,
        steiner_insertions,
    };
    Ok(recovery)
}

fn canonical_protected_faces(
    triangles: &[DelaunayRecoveredFacetTriangle],
) -> Vec<[StableDigest; 3]> {
    triangles
        .iter()
        .map(|triangle| {
            let mut face = triangle.node_identities;
            face.sort_unstable();
            face
        })
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn support_edge_exists(topology: &DelaunayVolumeTopology, endpoints: [StableDigest; 2]) -> bool {
    let [Some(first), Some(second)] = endpoints.map(|identity| node_index(topology, identity))
    else {
        return false;
    };
    topology.incidence.vertex_stars[first]
        .iter()
        .any(|tetrahedron| {
            topology.tetrahedra[*tetrahedron as usize]
                .vertex_indices
                .contains(&(second as u32))
        })
}

fn triangle_edges(triangle: [StableDigest; 3]) -> [[StableDigest; 2]; 3] {
    [
        [triangle[0], triangle[1]],
        [triangle[1], triangle[2]],
        [triangle[2], triangle[0]],
    ]
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
