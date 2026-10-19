//! Independent hull support proof. Adjacent exactly coplanar triangles share
//! one halfspace; prove that halfspace for every point rather than repeating it.
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
    let mut covered = vec![false; faces.len()];
    for first in 0..faces.len() {
        if (first as u64).is_multiple_of(options.insertion.topology.cancellation_check_interval) {
            work.check_cancellation()?;
        }
        if covered[first] {
            continue;
        }
        let (face, interior) = faces[first];
        let points = face.map(|vertex| topology.nodes[vertex as usize].coordinates_m);
        let interior_sign = work.orient3d([
            points[0],
            points[1],
            points[2],
            topology.nodes[interior as usize].coordinates_m,
        ])?;
        if interior_sign == PredicateSign::Zero {
            return Err(error(
                DelaunayPointSetErrorKind::InvalidTopology,
                "convex-hull face has a coplanar interior apex",
            ));
        }
        for (index, node) in topology.nodes.iter().enumerate() {
            let sign = work.orient3d([points[0], points[1], points[2], node.coordinates_m])?;
            if sign != PredicateSign::Zero && sign != interior_sign {
                return Err(error(DelaunayPointSetErrorKind::InvalidTopology,
                    format!("boundary face {face:?} does not support the admitted convex hull; node {index} lies on its exterior side")));
            }
        }
        covered[first] = true;
        let mut pending = VecDeque::from([first]);
        while let Some(index) = pending.pop_front() {
            let (current, _) = faces[index];
            for mut edge in [
                [current[0], current[1]],
                [current[1], current[2]],
                [current[2], current[0]],
            ] {
                edge.sort_unstable();
                for next in &edges[&edge] {
                    if covered[*next] {
                        continue;
                    }
                    if coplanar(topology, points, faces[*next].0, &mut work)? {
                        covered[*next] = true;
                        pending.push_back(*next);
                    }
                }
            }
        }
    }
    Ok(())
}

fn coplanar(
    topology: &DelaunayVolumeTopology,
    plane: [[f64; 3]; 3],
    face: [u32; 3],
    work: &mut HullWork<'_>,
) -> Result<bool, DelaunayPointSetError> {
    for vertex in face {
        if work.orient3d([
            plane[0],
            plane[1],
            plane[2],
            topology.nodes[vertex as usize].coordinates_m,
        ])? != PredicateSign::Zero
        {
            return Ok(false);
        }
    }
    Ok(true)
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
        options.insertion.maximum_predicate_evaluations = 1000;
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
