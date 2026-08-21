use std::collections::{BTreeSet, VecDeque};

use runmat_meshing_core::{
    quality::predicate::{insphere3d_symbolic, orient3d, PredicateSign, SpatialPredicatePoint},
    StableDigest,
};

use super::{
    error, predicate_error, resource, DelaunayInsertionError, DelaunayInsertionErrorKind,
    DelaunayVolumeNode, DelaunayVolumeTopology, InsertionWork,
};

pub(super) fn connected_cavity(
    topology: &DelaunayVolumeTopology,
    node: DelaunayVolumeNode,
    seed: usize,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<BTreeSet<usize>, DelaunayInsertionError> {
    let mut cavity = BTreeSet::new();
    let mut examined = BTreeSet::new();
    let mut queue = VecDeque::from([(seed, false)]);
    while let Some((index, forced)) = queue.pop_front() {
        work.checkpoint()?;
        if cavity.contains(&index) || !forced && !examined.insert(index) {
            continue;
        }
        let tetrahedron = &topology.tetrahedra[index];
        if !forced && !in_circumsphere(topology, tetrahedron.vertex_indices, node, work)? {
            continue;
        }
        cavity.insert(index);
        if cavity.len() as u64 > work.options.maximum_cavity_tetrahedra {
            return Err(resource("cavity tetrahedron limit exceeded"));
        }
        for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            if let Some(neighbor) = neighbor {
                if protected_faces.contains(&stable_face(
                    topology,
                    tetrahedron.vertex_indices,
                    opposite,
                )) || protect_region_boundaries
                    && topology.tetrahedra[*neighbor as usize].region_id != tetrahedron.region_id
                {
                    continue;
                }
                // A symbolic in-sphere tie must never leave a physical zero-volume
                // replacement across a face containing the inserted node.
                let coplanar = node_coplanar_with_face(
                    topology,
                    tetrahedron.vertex_indices,
                    opposite,
                    node,
                    work,
                )?;
                queue.push_back((*neighbor as usize, coplanar));
            }
        }
    }
    if !cavity.contains(&seed) {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "the containing tetrahedron does not contain the node in its circumsphere",
        ));
    }
    Ok(cavity)
}

pub(super) fn stable_face(
    topology: &DelaunayVolumeTopology,
    vertices: [u32; 4],
    opposite: usize,
) -> [StableDigest; 3] {
    let mut face = [StableDigest::ZERO; 3];
    let mut cursor = 0;
    for (vertex_index, vertex) in vertices.iter().enumerate() {
        if vertex_index != opposite {
            face[cursor] = topology.nodes[*vertex as usize].identity;
            cursor += 1;
        }
    }
    face.sort_unstable();
    face
}

fn node_coplanar_with_face(
    topology: &DelaunayVolumeTopology,
    vertices: [u32; 4],
    opposite: usize,
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    work.predicate()?;
    let mut points = [[0.0; 3]; 4];
    let mut cursor = 0;
    for (vertex_index, vertex) in vertices.iter().enumerate() {
        if vertex_index != opposite {
            points[cursor] = topology.nodes[*vertex as usize].coordinates_m;
            cursor += 1;
        }
    }
    points[3] = node.coordinates_m;
    orient3d(points)
        .map(|sign| sign == PredicateSign::Zero)
        .map_err(predicate_error)
}

fn in_circumsphere(
    topology: &DelaunayVolumeTopology,
    vertices: [u32; 4],
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    work.predicate()?;
    let points = vertices.map(|vertex| predicate_point(topology.nodes[vertex as usize]));
    let query = predicate_point(node);
    insphere3d_symbolic([points[0], points[1], points[2], points[3], query])
        .map(|sign| sign == PredicateSign::Positive)
        .map_err(predicate_error)
}

fn predicate_point(node: DelaunayVolumeNode) -> SpatialPredicatePoint {
    SpatialPredicatePoint {
        identity: node.identity,
        coordinates: node.coordinates_m,
    }
}
