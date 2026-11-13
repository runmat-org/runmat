//! Independent hull convexity proof. The boundary of the tetrahedral complex is a closed,
//! connected, consistently oriented surface; such a surface that is convex across each of its
//! edges bounds a convex body, so the complex covers the convex hull of its vertices. The proof
//! costs a constant number of predicates per boundary edge rather than one per point and plane.
use super::{
    error, work::HullWork, DelaunayPointSetError, DelaunayPointSetErrorKind,
    DelaunayPointSetOptions, DelaunayVolumeTopology,
};
use runmat_meshing_core::{quality::predicate::PredicateSign, MeshingCancellationSignal};
use std::collections::{BTreeMap, VecDeque};

type HullFace = ([u32; 3], u32);

pub(super) fn validate_convex_hull_closure(
    topology: &DelaunayVolumeTopology,
    options: DelaunayPointSetOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayPointSetError> {
    let mut work = HullWork::new(options, cancellation);
    let mut faces = Vec::<HullFace>::new();
    let mut edges = BTreeMap::<[u32; 2], Vec<usize>>::new();
    for (index, tet) in topology.tetrahedra.iter().enumerate() {
        if (index as u64).is_multiple_of(options.insertion.topology.cancellation_check_interval) {
            work.check_cancellation()?;
        }
        for opposite in 0..4 {
            if tet.neighbors[opposite].is_some() {
                continue;
            }
            let vertices = tet
                .vertex_indices
                .iter()
                .enumerate()
                .filter_map(|(slot, vertex)| (slot != opposite).then_some(*vertex))
                .collect::<Vec<_>>();
            let face = [vertices[0], vertices[1], vertices[2]];
            for mut edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
                edge.sort_unstable();
                edges.entry(edge).or_default().push(faces.len());
            }
            faces.push((face, tet.vertex_indices[opposite]));
        }
    }
    if edges.values().any(|uses| uses.len() != 2) {
        return Err(error(
            DelaunayPointSetErrorKind::InvalidTopology,
            "convex-hull boundary edge is not manifold",
        ));
    }
    if faces.is_empty() {
        return Err(error(
            DelaunayPointSetErrorKind::InvalidTopology,
            "convex-hull complex has no boundary",
        ));
    }
    let mut retained = vec![false; topology.nodes.len()];
    for tet in &topology.tetrahedra {
        for vertex in tet.vertex_indices {
            retained[vertex as usize] = true;
        }
    }
    if let Some(index) = retained.iter().position(|retained| !retained) {
        return Err(error(
            DelaunayPointSetErrorKind::InvalidTopology,
            format!("node {index} is not a vertex of the convex-hull complex; it lies on the exterior side of the hull"),
        ));
    }
    let mut interior_signs = Vec::with_capacity(faces.len());
    for (index, (face, interior)) in faces.iter().enumerate() {
        if (index as u64).is_multiple_of(options.insertion.topology.cancellation_check_interval) {
            work.check_cancellation()?;
        }
        let points = face.map(|vertex| {
            topology.nodes[vertex as usize]
                .coordinates_m
                .canonical_value()
        });
        let interior_sign = work.orient3d([
            points[0],
            points[1],
            points[2],
            topology.nodes[*interior as usize]
                .coordinates_m
                .canonical_value(),
        ])?;
        if interior_sign == PredicateSign::Zero {
            return Err(error(
                DelaunayPointSetErrorKind::InvalidTopology,
                "convex-hull face has a coplanar interior apex",
            ));
        }
        interior_signs.push(interior_sign);
    }
    for (edge, uses) in &edges {
        for (face_index, other_index) in [(uses[0], uses[1]), (uses[1], uses[0])] {
            let (face, _) = faces[face_index];
            let (other, _) = faces[other_index];
            let apex = other
                .into_iter()
                .find(|vertex| !edge.contains(vertex))
                .expect("a boundary face has a vertex off each of its edges");
            let points = face.map(|vertex| {
                topology.nodes[vertex as usize]
                    .coordinates_m
                    .canonical_value()
            });
            let sign = work.orient3d([
                points[0],
                points[1],
                points[2],
                topology.nodes[apex as usize]
                    .coordinates_m
                    .canonical_value(),
            ])?;
            if sign != PredicateSign::Zero && sign != interior_signs[face_index] {
                return Err(error(DelaunayPointSetErrorKind::InvalidTopology,
                    format!("boundary face {face:?} does not support the admitted convex hull; node {apex} lies on its exterior side")));
            }
        }
    }
    let mut reached = vec![false; faces.len()];
    let mut pending = VecDeque::from([0]);
    reached[0] = true;
    while let Some(index) = pending.pop_front() {
        let (face, _) = faces[index];
        for mut edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
            edge.sort_unstable();
            for next in &edges[&edge] {
                if !reached[*next] {
                    reached[*next] = true;
                    pending.push_back(*next);
                }
            }
        }
    }
    if reached.iter().any(|reached| !reached) {
        return Err(error(
            DelaunayPointSetErrorKind::InvalidTopology,
            "convex-hull boundary is not one connected surface",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdt::{build_delaunay_volume_point_set, DelaunayVolumeNode};
    use runmat_meshing_core::{NeverCancelled, StableDigest};

    fn dense_cube() -> DelaunayVolumeTopology {
        let mut nodes = Vec::new();
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    let identity = StableDigest::from_bytes([nodes.len() as u8 + 1; 32]);
                    nodes.push(DelaunayVolumeNode::explicit(
                        identity,
                        [x as f64, y as f64, z as f64],
                    ));
                }
            }
        }
        build_delaunay_volume_point_set(nodes, DelaunayPointSetOptions::default(), &NeverCancelled)
            .unwrap()
    }

    #[test]
    fn coplanar_hull_patches_cover_all_points_under_the_original_work_ceiling() {
        let topology = dense_cube();
        crate::cdt::insertion::validate_delaunay_volume_topology(
            &topology,
            crate::cdt::DelaunayInsertionOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let mut options = DelaunayPointSetOptions::default();
        options.insertion.maximum_predicate_evaluations = 300;
        validate_convex_hull_closure(&topology, options, &NeverCancelled).unwrap();
    }

    #[test]
    fn grouped_halfspaces_still_reject_an_exterior_point() {
        let mut topology = dense_cube();
        topology.nodes.push(DelaunayVolumeNode::explicit(
            StableDigest::from_bytes([99; 32]),
            [-1., -1., -1.],
        ));
        let failure = validate_convex_hull_closure(
            &topology,
            DelaunayPointSetOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err();
        assert_eq!(failure.kind, DelaunayPointSetErrorKind::InvalidTopology);
        assert!(failure.reason.contains("exterior side"));
    }
}
