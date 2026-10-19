mod complete;
pub(super) use complete::legalize_delaunay_volume_topology_complete;
mod global_queries;
use global_queries::{
    first_global_delaunay_violation, global_delaunay_violation_queries,
    indexed_tetrahedra_global_violation_count, tetrahedra_global_violation_count,
};
mod edge_star;
mod progress;
use edge_star::replacement as edge_star_replacement;
mod search_state;
mod two_to_three;
#[cfg(test)]
use two_to_three::apply_two_to_three_flip;
mod boundary;
pub(super) mod mutable_repair;
use boundary::first_illegal_boundary_edge;
mod admission;
use admission::{
    apply_reconnection, prospective_tetrahedra_form_embedded_manifold,
    tetrahedra_are_materializable,
};
mod edge;
use edge::first_reducible_edge;
mod duplicates;
use duplicates::replacement_duplicates_retained_tetrahedron;
mod reinsertion;
use reinsertion::{vertex_reinsertion_candidate, vertex_reinsertion_geometry};
mod predicate;
use predicate::{repair_sphere_relation, SphereRelationPolicy};
pub(super) mod local;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_meshing_core::{quality::predicate::PredicateSign, StableDigest};

use super::{
    cavity::{circumsphere_relation, stable_face},
    error, predicate_error, topology_error, two_to_three_topology_is_admissible,
    DelaunayInsertionError, DelaunayInsertionErrorKind, DelaunayVolumeTopology, InsertionWork,
};
use crate::{
    cdt::topology::{
        build_delaunay_volume_topology_with_regions, orient_materialized_volume_nodes,
    },
    reconnect::{
        local_tetrahedron_boundary_faces, three_to_two_edge_flip_candidate,
        two_to_three_geometry_is_admissible, LocalTetrahedron,
    },
};

use crate::cdt::topology::insphere_materialized_volume_nodes as insphere_volume_nodes;
use crate::cdt::topology::orient_materialized_volume_nodes as orient_volume_nodes;

type CavityReplacement = (Vec<usize>, Vec<[u32; 4]>);
type ScoredCavityReplacement = (Vec<usize>, Vec<[u32; 4]>, u64, u64);

mod canonical_local;

pub(super) fn legalize_delaunay_volume_topology(
    topology: DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    focus_identity: Option<StableDigest>,
    work: &mut InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    canonical_local::restore(
        topology,
        protected_faces,
        protect_region_boundaries,
        focus_identity,
        work,
    )
}

#[cfg(test)]
fn reference_canonical_legalization(
    mut topology: DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    focus_identity: Option<StableDigest>,
    work: &mut InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let focus = focus_identity
        .map(|identity| {
            topology
                .nodes
                .binary_search_by_key(&identity, |node| node.identity)
                .map(|index| index as u32)
                .map_err(|_| {
                    error(
                        DelaunayInsertionErrorKind::InvalidTopology,
                        "local legalization focus node is absent",
                    )
                })
        })
        .transpose()?;
    loop {
        let Some((left, right, face)) = first_illegal_face(
            &topology,
            protected_faces,
            protect_region_boundaries,
            focus,
            SphereRelationPolicy::CanonicalInsertion,
            work,
        )?
        else {
            return Ok(topology);
        };
        record_flip(work)?;
        topology = apply_two_to_three_flip(topology, left, right, face, work)?;
    }
}

fn topology_has_edge(
    topology: &impl super::topology_access::TetrahedralTopology,
    edge: [u32; 2],
) -> bool {
    topology.has_edge(edge)
}

fn removed_internal_face_is_protected(
    topology: &impl super::topology_access::TetrahedralTopology,
    removed: &[usize],
    protected_faces: &BTreeSet<[StableDigest; 3]>,
) -> bool {
    removed.iter().any(|index| {
        let tetrahedron = topology.tetrahedron(*index as u32).expect("live cavity");
        tetrahedron
            .neighbors
            .iter()
            .enumerate()
            .any(|(opposite, neighbor)| {
                neighbor.is_some_and(|neighbor| removed.contains(&(neighbor as usize)))
                    && protected_faces.contains(&stable_face(
                        topology,
                        tetrahedron.vertex_indices,
                        opposite,
                    ))
            })
    })
}

fn record_flip(work: &mut InsertionWork<'_>) -> Result<(), DelaunayInsertionError> {
    work.legalization_flips = work.legalization_flips.checked_add(1).ok_or_else(|| {
        error(
            DelaunayInsertionErrorKind::ResourceLimit,
            "Delaunay legalization flip count overflow",
        )
    })?;
    if work.legalization_flips > work.options.maximum_cavity_tetrahedra {
        return Err(error(
            DelaunayInsertionErrorKind::ResourceLimit,
            "Delaunay legalization flip limit exceeded",
        ));
    }
    Ok(())
}

fn first_illegal_face(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    focus: Option<u32>,
    policy: SphereRelationPolicy,
    work: &mut InsertionWork<'_>,
) -> Result<Option<(usize, usize, [u32; 3])>, DelaunayInsertionError> {
    for (left_index, left) in topology.tetrahedra.iter().enumerate() {
        if focus.is_some_and(|focus| !left.vertex_indices.contains(&focus)) {
            continue;
        }
        for (opposite, neighbor) in left.neighbors.iter().enumerate() {
            if focus.is_some_and(|focus| left.vertex_indices[opposite] != focus) {
                continue;
            }
            let Some(right_index) = neighbor.map(|index| index as usize) else {
                continue;
            };
            if focus.is_none() && right_index <= left_index {
                continue;
            }
            work.checkpoint()?;
            let right = &topology.tetrahedra[right_index];
            let stable = stable_face(topology, left.vertex_indices, opposite);
            if protected_faces.contains(&stable)
                || protect_region_boundaries && left.region_id != right.region_id
            {
                continue;
            }
            let right_apex = right
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !left.vertex_indices.contains(vertex))
                .ok_or_else(|| {
                    error(
                        DelaunayInsertionErrorKind::InvalidTopology,
                        "legalization neighbors do not share exactly one triangular face",
                    )
                })?;
            let left_apex = left
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !right.vertex_indices.contains(vertex))
                .ok_or_else(|| {
                    error(
                        DelaunayInsertionErrorKind::InvalidTopology,
                        "legalization neighbors do not have distinct opposite vertices",
                    )
                })?;
            let mut face = [0; 3];
            let mut cursor = 0;
            for vertex in left.vertex_indices {
                if right.vertex_indices.contains(&vertex) {
                    face[cursor] = vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            if repair_sphere_relation(
                topology,
                left.vertex_indices,
                topology.nodes[right_apex as usize],
                policy,
                work,
            )? != PredicateSign::Positive
                || !two_to_three_topology_is_admissible(
                    topology,
                    left_index as u32,
                    right_index as u32,
                    left_apex,
                    right_apex,
                )
                || !two_to_three_geometry_is_admissible(
                    left.vertex_indices,
                    right.vertex_indices,
                    |node| {
                        let node = topology.nodes[node as usize];
                        node.materialized_predicate_point()
                    },
                )
                .map_err(super::predicate_error)?
            {
                continue;
            }
            return Ok(Some((left_index, right_index, face)));
        }
    }
    Ok(None)
}
