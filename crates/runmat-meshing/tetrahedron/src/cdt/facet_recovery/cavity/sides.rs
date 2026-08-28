//! Construction of the two closed cavity halves separated by one exact recovered facet.

use std::collections::BTreeSet;

use runmat_meshing_core::quality::predicate::{orient3d, PredicateSign};

use super::{invalid_topology, BoundaryFace};
use crate::{
    cavity::constrained::{ConstrainedCavity, ConstrainedCavityBoundaryFace},
    cdt::{DelaunayFacetRecoveryError, DelaunayVolumeTopology},
};

pub(super) fn side_cavities(
    topology: &DelaunayVolumeTopology,
    facet: [u32; 3],
    signs: &[PredicateSign],
    removed: &BTreeSet<u32>,
    boundary: &[BoundaryFace],
    constraint_index: u32,
) -> Result<Option<(ConstrainedCavity, ConstrainedCavity)>, DelaunayFacetRecoveryError> {
    let mut positive = Vec::new();
    let mut negative = Vec::new();
    for face in boundary {
        let face_signs = face.nodes.map(|node| signs[node as usize]);
        let destination = if face_signs.contains(&PredicateSign::Positive) {
            &mut positive
        } else if face_signs.contains(&PredicateSign::Negative) {
            &mut negative
        } else {
            return Ok(None);
        };
        destination.push(oriented_external_face(topology, face, constraint_index)?);
    }
    let mut positive_facet = facet;
    positive_facet.swap(0, 1);
    positive.push(boundary_contract(positive_facet));
    negative.push(boundary_contract(facet));
    let removed_tetrahedron_ids = removed.iter().copied().collect::<Vec<_>>();
    let positive_volume = closed_surface_volume(topology, &positive);
    let negative_volume = closed_surface_volume(topology, &negative);
    if positive_volume <= 0.0 || negative_volume <= 0.0 {
        return Ok(None);
    }
    Ok(Some((
        ConstrainedCavity {
            removed_tetrahedron_ids: removed_tetrahedron_ids.clone(),
            boundary_faces: positive,
            protected_node_ids: facet.to_vec(),
            target_volume_m3: positive_volume,
        },
        ConstrainedCavity {
            removed_tetrahedron_ids,
            boundary_faces: negative,
            protected_node_ids: facet.to_vec(),
            target_volume_m3: negative_volume,
        },
    )))
}

fn oriented_external_face(
    topology: &DelaunayVolumeTopology,
    face: &BoundaryFace,
    constraint_index: u32,
) -> Result<ConstrainedCavityBoundaryFace, DelaunayFacetRecoveryError> {
    let mut nodes = face.nodes;
    let coordinates = |node: u32| topology.nodes[node as usize].coordinates_m;
    match orient3d([
        coordinates(nodes[0]),
        coordinates(nodes[1]),
        coordinates(nodes[2]),
        coordinates(face.interior_node),
    ])
    .map_err(|failure| {
        invalid_topology(
            constraint_index,
            format!("cavity boundary predicate failed: {failure:?}"),
        )
    })? {
        PredicateSign::Positive => nodes.swap(0, 1),
        PredicateSign::Negative => {}
        PredicateSign::Zero => {
            return Err(invalid_topology(
                constraint_index,
                "facet cavity boundary belongs to a degenerate tetrahedron",
            ));
        }
    }
    Ok(boundary_contract(nodes))
}

fn boundary_contract(node_ids: [u32; 3]) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: Vec::new(),
        source_face_id: None,
        source_edge_ids: [None; 3],
        region_ids: Vec::new(),
    }
}

fn closed_surface_volume(
    topology: &DelaunayVolumeTopology,
    faces: &[ConstrainedCavityBoundaryFace],
) -> f64 {
    // Translating to a boundary-derived reference reduces cancellation without changing the
    // oriented closed-surface volume. Boundary faces are oriented away from each cavity half.
    let reference = faces
        .iter()
        .flat_map(|face| face.node_ids)
        .fold([0.0; 3], |mut sum, node| {
            for (axis, value) in topology.nodes[node as usize]
                .coordinates_m
                .iter()
                .enumerate()
            {
                sum[axis] += value;
            }
            sum
        });
    let count = (faces.len() * 3) as f64;
    let reference = reference.map(|value| value / count);
    faces
        .iter()
        .map(|face| {
            let points = face.node_ids.map(|node| {
                let point = topology.nodes[node as usize].coordinates_m;
                std::array::from_fn(|axis| point[axis] - reference[axis])
            });
            dot(points[0], cross(points[1], points[2])) / 6.0
        })
        .sum::<f64>()
        .abs()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}
