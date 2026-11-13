//! Recover an edge through a bounded, coverage-preserving physical transaction.
//! Exact equal spheres or complete changed-adjacency checks admit the candidate.
//! No canonical clone, global search, or tolerance-based sphere equivalence.
use super::*;
use crate::cdt::insertion::{
    cospherical::{
        admits_boundary_diagonal, admits_interior_diagonal, boundary_diagonal_geometry,
        interior_diagonal_geometry,
    },
    local_admission::{admits_replacement, LocalAdmissionError},
};

pub(super) fn recover(
    topology: &mut TopologyMutation,
    left: StableDigest,
    right: StableDigest,
    protected_edges: &[[StableDigest; 2]],
    protected_faces: &[[StableDigest; 3]],
    index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let Some(left_vertex) = topology.node_index(left) else {
        return Ok(false);
    };
    let Some(right_vertex) = topology.node_index(right) else {
        return Ok(false);
    };
    for first in topology.vertex_star(left_vertex) {
        work.search_flip_step(index)?;
        let old = topology
            .tetrahedron(first)
            .expect("active vertex star")
            .clone();
        let slot = old
            .vertex_indices
            .iter()
            .position(|v| *v == left_vertex)
            .expect("vertex star contains the endpoint");
        if let Some(second) = old.neighbors[slot] {
            let other = topology.tetrahedron(second).expect("active neighbor");
            if other.vertex_indices.contains(&right_vertex) && other.region_id == old.region_id {
                let face = face(old.vertex_indices, slot);
                if !protected_faces.contains(&stable_face(topology, face)) {
                    work.flip_attempt(index)?;
                    let created = [
                        [left_vertex, right_vertex, face[0], face[1]],
                        [left_vertex, right_vertex, face[1], face[2]],
                        [left_vertex, right_vertex, face[2], face[0]],
                    ];
                    if admits_interior_diagonal(
                        topology,
                        old.vertex_indices,
                        [left_vertex, right_vertex],
                        face,
                    )
                    .map_err(|failure| predicate_error(index, failure))?
                        || (interior_diagonal_geometry(topology, [left_vertex, right_vertex], face)
                            .map_err(|failure| predicate_error(index, failure))?
                            && admits_changed_cavity(
                                topology,
                                [first, second],
                                &created,
                                index,
                                work,
                            )?)
                    {
                        topology
                            .replace_cavity(
                                &BTreeSet::from([first, second]),
                                created
                                    .into_iter()
                                    .map(|v| (v, old.region_id.clone()))
                                    .collect(),
                            )
                            .map_err(mutation_error)?;
                        return Ok(true);
                    }
                }
            }
        }
        for opposite in 0..4 {
            if old.neighbors[opposite].is_some() {
                continue;
            }
            let old_face = face(old.vertex_indices, opposite);
            if !old_face.contains(&left_vertex) || old_face.contains(&right_vertex) {
                continue;
            }
            for next in topology.boundary_neighbors_with_checkpoint(old_face, &mut || {
                work.search_flip_step(index)
            })? {
                work.search_flip_step(index)?;
                if !next.contains(&right_vertex) || next.contains(&left_vertex) {
                    continue;
                }
                let shared = old_face
                    .into_iter()
                    .filter(|v| next.contains(v))
                    .collect::<Vec<_>>();
                if shared.len() != 2 {
                    continue;
                }
                let edge = stable_edge(topology, [shared[0], shared[1]]);
                if protected_edges.contains(&edge)
                    || protected_faces.contains(&stable_face(topology, old_face))
                    || protected_faces.contains(&stable_face(topology, next))
                    || protected_faces.contains(&stable_face(
                        topology,
                        [shared[0], shared[1], old.vertex_indices[opposite]],
                    ))
                {
                    continue;
                }
                let Some((second, other_opposite)) = topology.boundary_face_owner(next) else {
                    continue;
                };
                let other = topology.tetrahedron(second).expect("active boundary owner");
                let apex = old.vertex_indices[opposite];
                if other.vertex_indices[other_opposite] != apex || old.region_id != other.region_id
                {
                    continue;
                }
                work.flip_attempt(index)?;
                // Coplanarity plus opposite sides of BOTH diagonals proves a
                // convex quadrilateral with unchanged physical hull coverage.
                if !(admits_boundary_diagonal(
                    topology,
                    old.vertex_indices,
                    [left_vertex, right_vertex],
                    [shared[0], shared[1]],
                    apex,
                )
                .map_err(|failure| predicate_error(index, failure))?
                    || boundary_diagonal_geometry(
                        topology,
                        [left_vertex, right_vertex],
                        [shared[0], shared[1]],
                        apex,
                    )
                    .map_err(|failure| predicate_error(index, failure))?
                        && admits_changed_cavity(
                            topology,
                            [first, second],
                            &[
                                [left_vertex, right_vertex, shared[0], apex],
                                [left_vertex, right_vertex, shared[1], apex],
                            ],
                            index,
                            work,
                        )?)
                {
                    continue;
                }
                topology
                    .replace_cavity(
                        &BTreeSet::from([first, second]),
                        vec![
                            (
                                [left_vertex, right_vertex, shared[0], apex],
                                old.region_id.clone(),
                            ),
                            (
                                [left_vertex, right_vertex, shared[1], apex],
                                old.region_id.clone(),
                            ),
                        ],
                    )
                    .map_err(mutation_error)?;
                return Ok(true);
            }
        }
    }
    flip::try_recover_admitted_planar_diagonal(
        topology,
        left_vertex,
        right_vertex,
        protected_edges,
        protected_faces,
        index,
        work,
    )
}

fn predicate_error(
    index: u32,
    error_value: runmat_meshing_core::quality::predicate::SpatialPredicateError,
) -> DelaunaySegmentRecoveryError {
    error(
        DelaunaySegmentRecoveryErrorKind::InvalidTopology,
        Some(index),
        format!("local reconnection physical predicate failed: {error_value:?}"),
    )
}
fn face(vertices: [u32; 4], opposite: usize) -> [u32; 3] {
    let mut result = [0; 3];
    let mut slot = 0;
    for (i, v) in vertices.into_iter().enumerate() {
        if i != opposite {
            result[slot] = v;
            slot += 1;
        }
    }
    result.sort_unstable();
    result
}
fn stable_face(topology: &TopologyMutation, vertices: [u32; 3]) -> [StableDigest; 3] {
    let mut result = vertices.map(|v| topology.nodes()[v as usize].identity);
    result.sort_unstable();
    result
}
fn stable_edge(topology: &TopologyMutation, vertices: [u32; 2]) -> [StableDigest; 2] {
    let mut result = vertices.map(|v| topology.nodes()[v as usize].identity);
    result.sort_unstable();
    result
}

fn admits_changed_cavity(
    topology: &TopologyMutation,
    removed: [u32; 2],
    created: &[[u32; 4]],
    index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    admits_replacement(topology, &BTreeSet::from(removed), created, &mut || {
        work.search_flip_step(index)
    })
    .map_err(|failure| match failure {
        LocalAdmissionError::Work(failure) => failure,
        LocalAdmissionError::Predicate(failure) => predicate_error(index, failure),
    })
}
