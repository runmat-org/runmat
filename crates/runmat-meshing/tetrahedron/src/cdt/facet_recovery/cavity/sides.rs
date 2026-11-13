//! Construction of the closed cavity side or sides bounded by one exact recovered facet.
use crate::cdt::insertion::topology_access::TetrahedralTopology;

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_meshing_core::quality::predicate::{orient3d, PredicateSign};

use super::{invalid_topology, BoundaryFace};
use crate::{
    cavity::constrained::{
        validate_constrained_cavity, ConstrainedCavity, ConstrainedCavityBoundaryFace,
    },
    cdt::DelaunayFacetRecoveryError,
};

pub(super) enum SideCavityConstruction {
    Ready(Vec<ConstrainedCavity>),
    Unavailable {
        reason: String,
        open_edge: Option<[u32; 2]>,
    },
}

pub(super) fn side_cavities(
    topology: &impl TetrahedralTopology,
    cap: &[[u32; 3]],
    removed: &BTreeSet<u32>,
    boundary: &[BoundaryFace],
    constraint_index: u32,
) -> Result<SideCavityConstruction, DelaunayFacetRecoveryError> {
    let external = boundary
        .iter()
        .map(|face| oriented_external_face(topology, face, constraint_index))
        .collect::<Result<Vec<_>, _>>()?;
    let mut cap_edge_uses = BTreeMap::<[u32; 2], usize>::new();
    for triangle in cap {
        for edge in facet_edges(*triangle) {
            *cap_edge_uses.entry(sorted_edge(edge)).or_default() += 1;
        }
    }
    if cap_edge_uses.values().any(|uses| *uses > 2) {
        return Ok(SideCavityConstruction::Unavailable {
            reason: "target facet cap has a nonmanifold edge".into(),
            open_edge: None,
        });
    }
    let target_edges = cap_edge_uses
        .iter()
        .filter_map(|(edge, uses)| (*uses == 1).then_some(*edge))
        .collect::<BTreeSet<_>>();
    let mut edge_owners = BTreeMap::<[u32; 2], Vec<usize>>::new();
    for (face_index, face) in external.iter().enumerate() {
        for edge in facet_edges(face.node_ids) {
            edge_owners
                .entry(sorted_edge(edge))
                .or_default()
                .push(face_index);
        }
    }
    if let Some((edge, owners)) = edge_owners.iter().find(|(_, owners)| owners.len() != 2) {
        return Ok(SideCavityConstruction::Unavailable {
            reason: format!(
                "intersected tetrahedra have nonmanifold boundary edge {edge:?} with {} incident faces",
                owners.len()
            ),
            open_edge: Some(*edge),
        });
    }
    if let Some(edge) = target_edges
        .iter()
        .find(|edge| edge_owners.get(*edge).map(Vec::len) != Some(2))
    {
        return Ok(SideCavityConstruction::Unavailable {
            reason: format!("target facet edge {edge:?} is absent from the cavity boundary"),
            open_edge: Some(*edge),
        });
    }
    let mut adjacent = vec![Vec::new(); external.len()];
    for (edge, owners) in &edge_owners {
        if target_edges.contains(edge) {
            continue;
        }
        adjacent[owners[0]].push(owners[1]);
        adjacent[owners[1]].push(owners[0]);
    }
    for neighbors in &mut adjacent {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    let mut unseen = (0..external.len()).collect::<BTreeSet<_>>();
    let mut components = Vec::new();
    while let Some(first) = unseen.pop_first() {
        let mut queue = VecDeque::from([first]);
        let mut component = Vec::new();
        while let Some(face_index) = queue.pop_front() {
            component.push(face_index);
            for neighbor in &adjacent[face_index] {
                if unseen.remove(neighbor) {
                    queue.push_back(*neighbor);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    if components.is_empty() || components.len() > 2 {
        return Ok(SideCavityConstruction::Unavailable {
            reason: format!(
                "target facet cut produced {} cavity-boundary components instead of one or two",
                components.len()
            ),
            open_edge: None,
        });
    }
    let removed_tetrahedron_ids = removed.iter().copied().collect::<Vec<_>>();
    let mut cavities = Vec::with_capacity(2);
    for component in components {
        let mut faces = component
            .iter()
            .map(|face_index| external[*face_index].clone())
            .collect::<Vec<_>>();
        let Some(cap) = oriented_cap(cap, &faces, &target_edges) else {
            return Ok(SideCavityConstruction::Unavailable {
                reason: "cavity boundary patch cannot be oriented against the target facet cut"
                    .into(),
                open_edge: None,
            });
        };
        faces.extend(cap.iter().copied().map(boundary_contract));
        let target_volume_m3 = closed_surface_volume(topology, &faces);
        if target_volume_m3 <= 0.0 {
            continue;
        }
        let cavity = ConstrainedCavity {
            removed_tetrahedron_ids: removed_tetrahedron_ids.clone(),
            boundary_faces: faces,
            protected_node_ids: cap
                .iter()
                .flat_map(|face| *face)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            target_volume_m3,
        };
        if let Err(error) = validate_constrained_cavity(&cavity) {
            let open_edge = match error {
                crate::cavity::constrained::ConstrainedCavityValidationError::NonManifoldBoundaryEdge {
                    node_ids,
                    ..
                } => Some(node_ids),
                _ => None,
            };
            return Ok(SideCavityConstruction::Unavailable {
                reason: format!(
                    "cavity boundary component is not closed by the target facet cap: {error:?}"
                ),
                open_edge,
            });
        }
        cavities.push(cavity);
    }
    if cavities.is_empty() {
        Ok(SideCavityConstruction::Unavailable {
            reason: "cavity has no tetrahedra on either side of the target facet".into(),
            open_edge: None,
        })
    } else {
        Ok(SideCavityConstruction::Ready(cavities))
    }
}

fn oriented_cap(
    cap: &[[u32; 3]],
    patch: &[ConstrainedCavityBoundaryFace],
    target_edges: &BTreeSet<[u32; 2]>,
) -> Option<Vec<[u32; 3]>> {
    for reverse in [false, true] {
        let candidate = cap
            .iter()
            .copied()
            .map(|mut triangle| {
                if reverse {
                    triangle.swap(0, 1);
                }
                triangle
            })
            .collect::<Vec<_>>();
        let closes_every_edge = target_edges.iter().all(|edge| {
            let patch_direction = patch
                .iter()
                .find_map(|face| directed_edge(face.node_ids, *edge));
            let cap_direction = candidate
                .iter()
                .find_map(|triangle| directed_edge(*triangle, *edge));
            matches!((patch_direction, cap_direction), (Some(left), Some(right)) if left == [right[1], right[0]])
        });
        if closes_every_edge && cap_is_oriented_manifold(&candidate, target_edges) {
            return Some(candidate);
        }
    }
    None
}

fn cap_is_oriented_manifold(cap: &[[u32; 3]], boundary: &BTreeSet<[u32; 2]>) -> bool {
    let mut directions = BTreeMap::<[u32; 2], Vec<[u32; 2]>>::new();
    for triangle in cap {
        for edge in facet_edges(*triangle) {
            directions.entry(sorted_edge(edge)).or_default().push(edge);
        }
    }
    directions.into_iter().all(|(edge, uses)| {
        if boundary.contains(&edge) {
            uses.len() == 1
        } else {
            uses.len() == 2 && uses[0] == [uses[1][1], uses[1][0]]
        }
    })
}

fn directed_edge(face: [u32; 3], target: [u32; 2]) -> Option<[u32; 2]> {
    facet_edges(face)
        .into_iter()
        .find(|edge| sorted_edge(*edge) == target)
}

fn facet_edges(face: [u32; 3]) -> [[u32; 2]; 3] {
    [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]]
}

fn sorted_edge(mut edge: [u32; 2]) -> [u32; 2] {
    edge.sort_unstable();
    edge
}

fn oriented_external_face(
    topology: &impl TetrahedralTopology,
    face: &BoundaryFace,
    constraint_index: u32,
) -> Result<ConstrainedCavityBoundaryFace, DelaunayFacetRecoveryError> {
    let mut nodes = face.nodes;
    let coordinates = |node: u32| {
        topology.nodes()[node as usize]
            .coordinates_m
            .canonical_value()
    };
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
    topology: &impl TetrahedralTopology,
    faces: &[ConstrainedCavityBoundaryFace],
) -> f64 {
    // Translating to a boundary-derived reference reduces cancellation without changing the
    // oriented closed-surface volume. Boundary faces are oriented away from each cavity half.
    let reference = faces
        .iter()
        .flat_map(|face| face.node_ids)
        .fold([0.0; 3], |mut sum, node| {
            for (axis, value) in topology.nodes()[node as usize]
                .coordinates_m
                .canonical_value()
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
                let point = topology.nodes()[node as usize]
                    .coordinates_m
                    .canonical_value();
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

#[cfg(test)]
mod tests {
    use super::{boundary_contract, cap_is_oriented_manifold, oriented_cap};
    use std::collections::BTreeSet;

    #[test]
    fn triangulated_cap_is_oriented_as_one_manifold_patch() {
        let cap = [[0, 1, 2], [0, 2, 3]];
        let boundary = BTreeSet::from([[0, 1], [0, 3], [1, 2], [2, 3]]);
        assert!(cap_is_oriented_manifold(&cap, &boundary));

        let opposing_patch = [[1, 0, 4], [2, 1, 4], [3, 2, 4], [0, 3, 4]].map(boundary_contract);
        assert_eq!(
            oriented_cap(&cap, &opposing_patch, &boundary),
            Some(cap.to_vec())
        );

        let matching_patch = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]].map(boundary_contract);
        assert_eq!(
            oriented_cap(&cap, &matching_patch, &boundary),
            Some(vec![[1, 0, 2], [2, 0, 3]])
        );
    }

    #[test]
    fn triangulated_cap_rejects_inconsistent_internal_edge_orientation() {
        let boundary = BTreeSet::from([[0, 1], [0, 3], [1, 2], [2, 3]]);
        assert!(!cap_is_oriented_manifold(
            &[[0, 1, 2], [0, 3, 2]],
            &boundary
        ));
    }
}
