use std::collections::{BTreeSet, VecDeque};

use runmat_meshing_core::{
    quality::predicate::{insphere3d, orient2d, orient3d, PredicateSign},
    StableDigest,
};

use super::{
    error, predicate_error, resource, topology_access::TetrahedralTopology, DelaunayInsertionError,
    DelaunayInsertionErrorKind, DelaunayVolumeNode, InsertionWork,
};

pub(super) fn connected_cavity<T: TetrahedralTopology>(
    topology: &T,
    node: DelaunayVolumeNode,
    seed: u32,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<BTreeSet<u32>, DelaunayInsertionError> {
    let mut cavity = BTreeSet::new();
    let mut examined = BTreeSet::new();
    let mut queue = VecDeque::from([(seed, false)]);
    while let Some((index, forced)) = queue.pop_front() {
        work.checkpoint()?;
        if cavity.contains(&index) || !forced && !examined.insert(index) {
            continue;
        }
        let tetrahedron = topology.tetrahedron(index).ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "insertion cavity reached an inactive tetrahedron",
            )
        })?;
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
                    && topology
                        .tetrahedron(*neighbor)
                        .is_none_or(|adjacent| adjacent.region_id != tetrahedron.region_id)
                {
                    continue;
                }
                // A symbolic in-sphere tie must never leave a physical zero-volume
                // replacement across a face containing the inserted node.
                let lies_on_face =
                    node_lies_on_face(topology, tetrahedron.vertex_indices, opposite, node, work)?;
                queue.push_back((*neighbor, lies_on_face));
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

fn node_lies_on_face(
    topology: &impl TetrahedralTopology,
    vertices: [u32; 4],
    opposite: usize,
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    if !node_coplanar_with_face(topology, vertices, opposite, node, work)? {
        return Ok(false);
    }
    let mut triangle = [[0.0; 3]; 3];
    let mut cursor = 0;
    for (vertex_index, vertex) in vertices.iter().enumerate() {
        if vertex_index != opposite {
            triangle[cursor] = topology.nodes()[*vertex as usize].coordinates_m;
            cursor += 1;
        }
    }
    let axes = projection_axes(triangle, work)?;
    let project = |point: [f64; 3]| [point[axes[0]], point[axes[1]]];
    let query = project(node.coordinates_m);
    let signs = [
        planar_orientation([project(triangle[0]), project(triangle[1]), query], work)?,
        planar_orientation([project(triangle[1]), project(triangle[2]), query], work)?,
        planar_orientation([project(triangle[2]), project(triangle[0]), query], work)?,
    ];
    let has_positive = signs.contains(&PredicateSign::Positive);
    let has_negative = signs.contains(&PredicateSign::Negative);
    Ok(!(has_positive && has_negative))
}

fn projection_axes(
    triangle: [[f64; 3]; 3],
    work: &mut InsertionWork<'_>,
) -> Result<[usize; 2], DelaunayInsertionError> {
    for axes in [[0, 1], [0, 2], [1, 2]] {
        if planar_orientation(triangle.map(|point| [point[axes[0]], point[axes[1]]]), work)?
            != PredicateSign::Zero
        {
            return Ok(axes);
        }
    }
    Err(error(
        DelaunayInsertionErrorKind::InvalidTopology,
        "tetrahedron face is exactly collinear",
    ))
}

fn planar_orientation(
    points: [[f64; 2]; 3],
    work: &mut InsertionWork<'_>,
) -> Result<PredicateSign, DelaunayInsertionError> {
    work.predicate()?;
    orient2d(points).map_err(|failure| {
        error(
            DelaunayInsertionErrorKind::InvalidNode,
            format!("planar predicate rejected coordinates: {failure:?}"),
        )
    })
}

pub(super) fn stable_face(
    topology: &impl TetrahedralTopology,
    vertices: [u32; 4],
    opposite: usize,
) -> [StableDigest; 3] {
    let mut face = [StableDigest::ZERO; 3];
    let mut cursor = 0;
    for (vertex_index, vertex) in vertices.iter().enumerate() {
        if vertex_index != opposite {
            face[cursor] = topology.nodes()[*vertex as usize].identity;
            cursor += 1;
        }
    }
    face.sort_unstable();
    face
}

pub(super) fn node_coplanar_with_face(
    topology: &impl TetrahedralTopology,
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
            points[cursor] = topology.nodes()[*vertex as usize].coordinates_m;
            cursor += 1;
        }
    }
    points[3] = node.coordinates_m;
    orient3d(points)
        .map(|sign| sign == PredicateSign::Zero)
        .map_err(predicate_error)
}

fn in_circumsphere(
    topology: &impl TetrahedralTopology,
    vertices: [u32; 4],
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    // Taking the closed circumsphere keeps the cavity coherent when several
    // tetrahedra are exactly cospherical.
    circumsphere_relation(topology, vertices, node, work)
        .map(|sign| matches!(sign, PredicateSign::Positive | PredicateSign::Zero))
}

pub(super) fn circumsphere_relation(
    topology: &impl TetrahedralTopology,
    vertices: [u32; 4],
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<PredicateSign, DelaunayInsertionError> {
    work.predicate()?;
    let points = vertices.map(|vertex| topology.nodes()[vertex as usize].coordinates_m);
    insphere3d([
        points[0],
        points[1],
        points[2],
        points[3],
        node.coordinates_m,
    ])
    .map_err(predicate_error)
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::NeverCancelled;

    use super::*;
    use crate::cdt::{
        build_delaunay_volume_topology, DelaunayInsertionOptions, DelaunayTopologyOptions,
    };

    fn node(identity: u8, coordinates_m: [f64; 3]) -> DelaunayVolumeNode {
        DelaunayVolumeNode {
            identity: StableDigest::from_bytes([identity; 32]),
            coordinates_m,
        }
    }

    #[test]
    fn coplanarity_does_not_force_cavity_traversal_outside_the_face() {
        let topology = build_delaunay_volume_topology(
            vec![
                node(10, [-2.0, -2.0, -2.0]),
                node(20, [2.0, -2.0, -2.0]),
                node(30, [0.0, 2.0, -2.0]),
                node(40, [0.0, 0.0, 2.0]),
            ],
            vec![[0, 1, 2, 3]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let vertices = topology.tetrahedra[0].vertex_indices;
        let opposite = vertices
            .iter()
            .position(|vertex| topology.nodes[*vertex as usize].coordinates_m[2] == 2.0)
            .unwrap();
        let mut work = InsertionWork::new(DelaunayInsertionOptions::default(), &NeverCancelled);

        assert!(node_lies_on_face(
            &topology,
            vertices,
            opposite,
            node(50, [0.0, -2.0 / 3.0, -2.0]),
            &mut work,
        )
        .unwrap());
        assert!(!node_lies_on_face(
            &topology,
            vertices,
            opposite,
            node(60, [10.0, 10.0, -2.0]),
            &mut work,
        )
        .unwrap());
    }
}
