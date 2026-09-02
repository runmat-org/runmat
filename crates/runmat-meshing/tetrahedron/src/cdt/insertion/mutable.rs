use std::collections::BTreeSet;

use runmat_meshing_core::{
    quality::predicate::PredicateSign, MeshingCancellationSignal, StableDigest,
};

use super::{
    cavity::{circumsphere_relation, stable_face},
    error, mutation_topology_error, prepare_insertion_from_hint,
    two_to_three_topology_is_admissible, validate_new_node, validate_options,
    DelaunayInsertionError, DelaunayInsertionErrorKind, DelaunayInsertionOptions,
    DelaunayVolumeNode, InsertionWork, PreparedInsertion, TopologyMutation,
};
use crate::cdt::topology::orient_volume_nodes;
use crate::reconnect::two_to_three_geometry_is_admissible;

pub(in crate::cdt) fn insert_delaunay_volume_node_into_mutation(
    topology: &mut TopologyMutation,
    node: DelaunayVolumeNode,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    insert_delaunay_volume_node_into_mutation_with_barriers(
        topology,
        node,
        &[],
        false,
        options,
        cancellation,
    )
}

pub(in crate::cdt) fn insert_delaunay_volume_node_into_mutation_with_barriers(
    topology: &mut TopologyMutation,
    node: DelaunayVolumeNode,
    protected_faces: &[[StableDigest; 3]],
    protect_region_boundaries: bool,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    insert_delaunay_volume_node_into_mutation_with_barriers_from_hint(
        topology,
        node,
        protected_faces,
        protect_region_boundaries,
        None,
        options,
        cancellation,
    )
}

pub(in crate::cdt) fn insert_delaunay_volume_node_into_mutation_with_barriers_from_hint(
    topology: &mut TopologyMutation,
    node: DelaunayVolumeNode,
    protected_faces: &[[StableDigest; 3]],
    protect_region_boundaries: bool,
    hint: Option<u32>,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_options(options)?;
    validate_new_node(topology, node)?;
    if protected_faces.len() as u64 > options.maximum_protected_faces {
        return Err(super::resource("protected-face inventory limit exceeded"));
    }
    let protected_faces = protected_faces.iter().copied().collect::<BTreeSet<_>>();
    insert_delaunay_volume_node_into_mutation_with_prepared_barriers_from_hint(
        topology,
        node,
        &protected_faces,
        protect_region_boundaries,
        hint,
        options,
        cancellation,
    )
}

pub(in crate::cdt) fn insert_delaunay_volume_node_into_mutation_with_prepared_barriers_from_hint(
    topology: &mut TopologyMutation,
    node: DelaunayVolumeNode,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    hint: Option<u32>,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_options(options)?;
    validate_new_node(topology, node)?;
    if protected_faces.len() as u64 > options.maximum_protected_faces {
        return Err(super::resource("protected-face inventory limit exceeded"));
    }
    let mut work = InsertionWork::new(options, cancellation);
    let prepared = prepare_insertion_from_hint(
        topology,
        node,
        protected_faces,
        protect_region_boundaries,
        hint,
        &mut work,
    )?;
    apply_prepared_delaunay_volume_node_insertion_with_work(
        topology,
        prepared,
        protected_faces,
        protect_region_boundaries,
        &mut work,
    )
}

pub(in crate::cdt) fn apply_prepared_delaunay_volume_node_insertion(
    topology: &mut TopologyMutation,
    prepared: PreparedInsertion,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_options(options)?;
    validate_new_node(topology, prepared.node)?;
    if protected_faces.len() as u64 > options.maximum_protected_faces {
        return Err(super::resource("protected-face inventory limit exceeded"));
    }
    let mut work = InsertionWork::new(options, cancellation);
    apply_prepared_delaunay_volume_node_insertion_with_work(
        topology,
        prepared,
        protected_faces,
        protect_region_boundaries,
        &mut work,
    )
}

fn apply_prepared_delaunay_volume_node_insertion_with_work(
    topology: &mut TopologyMutation,
    prepared: PreparedInsertion,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<(), DelaunayInsertionError> {
    let inserted = topology
        .append_node(prepared.node)
        .map_err(mutation_topology_error)?;
    let created = prepared
        .boundary
        .into_iter()
        .map(|face| {
            (
                [face[0], face[1], face[2], inserted],
                prepared.region_id.clone(),
            )
        })
        .collect();
    topology
        .replace_cavity(&prepared.removed, created)
        .map_err(mutation_topology_error)?;
    legalize_inserted_node(
        topology,
        inserted,
        protected_faces,
        protect_region_boundaries,
        work,
    )
}

fn legalize_inserted_node(
    topology: &mut TopologyMutation,
    inserted: u32,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<(), DelaunayInsertionError> {
    let mut flip_count = 0_u64;
    while let Some(flip) = first_illegal_inserted_face(
        topology,
        inserted,
        protected_faces,
        protect_region_boundaries,
        work,
    )? {
        flip_count = flip_count.checked_add(1).ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::ResourceLimit,
                "mutable legalization flip count overflow",
            )
        })?;
        if flip_count > work.options.maximum_cavity_tetrahedra {
            return Err(error(
                DelaunayInsertionErrorKind::ResourceLimit,
                "mutable legalization flip limit exceeded",
            ));
        }
        apply_two_to_three_flip(topology, flip).map_err(mutation_topology_error)?;
    }
    Ok(())
}

struct TwoToThreeFlip {
    left: u32,
    right: u32,
    face: [u32; 3],
}

fn first_illegal_inserted_face(
    topology: &TopologyMutation,
    inserted: u32,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<Option<TwoToThreeFlip>, DelaunayInsertionError> {
    for left_index in topology.vertex_stars[inserted as usize].iter().copied() {
        let left = topology
            .tetrahedron(left_index)
            .expect("inserted-node star references an active tetrahedron");
        let opposite = left
            .vertex_indices
            .iter()
            .position(|vertex| *vertex == inserted)
            .expect("inserted-node star tetrahedron contains the inserted node");
        let Some(right_index) = left.neighbors[opposite] else {
            continue;
        };
        work.checkpoint()?;
        let right = topology.tetrahedron(right_index).ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "mutable legalization neighbor is inactive",
            )
        })?;
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
                    "mutable legalization neighbors do not share one face",
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
        if circumsphere_relation(
            topology,
            left.vertex_indices,
            topology.nodes[right_apex as usize],
            work,
        )? != PredicateSign::Positive
            || !two_to_three_topology_is_admissible(
                topology,
                left_index,
                right_index,
                inserted,
                right_apex,
            )
            || !two_to_three_geometry_is_admissible(
                left.vertex_indices,
                right.vertex_indices,
                |node| {
                    let node = topology.nodes[node as usize];
                    DelaunayVolumeNode::explicit(node.identity, node.coordinates_m)
                        .predicate_point()
                },
            )
            .map_err(super::predicate_error)?
            || !created_tetrahedra_are_exactly_nondegenerate(topology, inserted, right_apex, face)?
        {
            continue;
        }
        return Ok(Some(TwoToThreeFlip {
            left: left_index,
            right: right_index,
            face,
        }));
    }
    Ok(None)
}

fn created_tetrahedra_are_exactly_nondegenerate(
    topology: &TopologyMutation,
    left_apex: u32,
    right_apex: u32,
    face: [u32; 3],
) -> Result<bool, DelaunayInsertionError> {
    for edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
        let sign = orient_volume_nodes([
            topology.nodes[left_apex as usize],
            topology.nodes[right_apex as usize],
            topology.nodes[edge[0] as usize],
            topology.nodes[edge[1] as usize],
        ])
        .map_err(super::predicate_error)?;
        if sign == PredicateSign::Zero {
            return Ok(false);
        }
    }
    Ok(true)
}

fn apply_two_to_three_flip(
    topology: &mut TopologyMutation,
    flip: TwoToThreeFlip,
) -> Result<(), super::TopologyMutationError> {
    let left = topology
        .tetrahedron(flip.left)
        .expect("legalization left tetrahedron remains active")
        .clone();
    let right = topology
        .tetrahedron(flip.right)
        .expect("legalization right tetrahedron remains active")
        .clone();
    if left.region_id != right.region_id {
        return Err(crate::cdt::mutation::TopologyMutationError {
            kind: crate::cdt::mutation::TopologyMutationErrorKind::InvalidTopology,
            reason: "mutable legalization cannot cross a region boundary".to_owned(),
        });
    }
    let left_apex = left
        .vertex_indices
        .iter()
        .copied()
        .find(|vertex| !flip.face.contains(vertex))
        .expect("legalization face leaves one left apex");
    let right_apex = right
        .vertex_indices
        .iter()
        .copied()
        .find(|vertex| !flip.face.contains(vertex))
        .expect("legalization face leaves one right apex");
    let removed = [flip.left, flip.right].into_iter().collect();
    let created = [
        [left_apex, right_apex, flip.face[0], flip.face[1]],
        [left_apex, right_apex, flip.face[1], flip.face[2]],
        [left_apex, right_apex, flip.face[2], flip.face[0]],
    ]
    .into_iter()
    .map(|tetrahedron| (tetrahedron, left.region_id.clone()))
    .collect();
    topology.replace_cavity(&removed, created)?;
    Ok(())
}
