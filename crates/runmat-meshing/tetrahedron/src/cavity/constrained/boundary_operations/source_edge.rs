use runmat_meshing_core::predicate::{
    orient_tetrahedron_node_ids, tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian,
};

use super::*;

pub fn split_constrained_cavity_source_edge(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    source_nodes: &[ConstrainedCavityNode],
    source_tetrahedra: &[CavityTetrahedron],
    edge: [u32; 2],
) -> Result<ConstrainedCavitySourceEdgeSplit, ConstrainedCavitySourceEdgeSplitError> {
    let target_edge = sorted_edge(edge);
    let boundary_node_map = boundary_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for node_id in target_edge {
        if !boundary_node_map.contains_key(&node_id) {
            return Err(ConstrainedCavitySourceEdgeSplitError::MissingBoundaryNode { node_id });
        }
    }
    if !cavity.boundary_faces.iter().any(|face| {
        face_edges(face.node_ids)
            .into_iter()
            .any(|candidate| sorted_edge(candidate) == target_edge)
    }) {
        return Err(ConstrainedCavitySourceEdgeSplitError::MissingBoundaryEdge {
            node_ids: target_edge,
        });
    }

    let source_node_map = source_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for tetrahedron in source_tetrahedra {
        for node_id in tetrahedron.node_ids {
            if !source_node_map.contains_key(&node_id) {
                return Err(ConstrainedCavitySourceEdgeSplitError::MissingSourceNode { node_id });
            }
        }
    }
    for tetrahedron_id in &cavity.removed_tetrahedron_ids {
        if !source_tetrahedra
            .iter()
            .any(|tetrahedron| tetrahedron.tetrahedron_id == *tetrahedron_id)
        {
            return Err(
                ConstrainedCavitySourceEdgeSplitError::MissingRemovedSourceTetrahedron {
                    tetrahedron_id: *tetrahedron_id,
                },
            );
        }
    }

    let mut split_node_id = source_node_map
        .keys()
        .chain(boundary_node_map.keys())
        .copied()
        .max()
        .unwrap_or_default()
        .saturating_add(1);
    while source_node_map.contains_key(&split_node_id)
        || boundary_node_map.contains_key(&split_node_id)
    {
        split_node_id = split_node_id.saturating_add(1);
    }
    let points = target_edge.map(|node_id| boundary_node_map[&node_id]);
    let split_node = ConstrainedCavityNode {
        node_id: split_node_id,
        coordinates_m: [
            0.5 * (points[0][0] + points[1][0]),
            0.5 * (points[0][1] + points[1][1]),
            0.5 * (points[0][2] + points[1][2]),
        ],
    };
    let mut node_map_with_split = source_node_map;
    node_map_with_split.insert(split_node.node_id, split_node.coordinates_m);

    let selected_original_tetrahedron_ids = cavity
        .removed_tetrahedron_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut selected_tetrahedron_ids = BTreeSet::<u32>::new();
    let mut split_source_tetrahedra =
        Vec::<CavityTetrahedron>::with_capacity(source_tetrahedra.len() + 8);
    let mut next_tetrahedron_id = source_tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.tetrahedron_id)
        .max()
        .unwrap_or_default()
        .saturating_add(1);
    let mut incident_count = 0_usize;

    for tetrahedron in source_tetrahedra {
        let incident = target_edge
            .iter()
            .all(|node_id| tetrahedron.node_ids.contains(node_id));
        if !incident {
            if selected_original_tetrahedron_ids.contains(&tetrahedron.tetrahedron_id) {
                selected_tetrahedron_ids.insert(tetrahedron.tetrahedron_id);
            }
            split_source_tetrahedra.push(tetrahedron.clone());
            continue;
        }

        incident_count += 1;
        let opposite_nodes = tetrahedron
            .node_ids
            .into_iter()
            .filter(|node_id| !target_edge.contains(node_id))
            .collect::<Vec<_>>();
        if opposite_nodes.len() != 2 {
            return Err(
                ConstrainedCavitySourceEdgeSplitError::DegenerateSplitTetrahedron {
                    tetrahedron_id: tetrahedron.tetrahedron_id,
                },
            );
        }
        let child_node_ids = [
            [
                target_edge[0],
                split_node.node_id,
                opposite_nodes[0],
                opposite_nodes[1],
            ],
            [
                split_node.node_id,
                target_edge[1],
                opposite_nodes[0],
                opposite_nodes[1],
            ],
        ];
        for child in child_node_ids {
            let points = child.map(|node_id| node_map_with_split[&node_id]);
            let (oriented_node_ids, volume_m3) = orient_tetrahedron_node_ids(child, points);
            if volume_m3 <= 0.0 {
                return Err(
                    ConstrainedCavitySourceEdgeSplitError::DegenerateSplitTetrahedron {
                        tetrahedron_id: tetrahedron.tetrahedron_id,
                    },
                );
            }
            let oriented_points = oriented_node_ids.map(|node_id| node_map_with_split[&node_id]);
            let child_tetrahedron = CavityTetrahedron {
                tetrahedron_id: next_tetrahedron_id,
                component_id: tetrahedron.component_id,
                node_ids: oriented_node_ids,
                source_surface_element_id: tetrahedron.source_surface_element_id,
                region_ids: tetrahedron.region_ids.clone(),
                volume_m3,
                aspect_ratio: tetrahedron_edge_aspect_ratio(oriented_points),
                exact_scaled_jacobian: tetrahedron_scaled_jacobian(oriented_points).abs(),
            };
            if selected_original_tetrahedron_ids.contains(&tetrahedron.tetrahedron_id) {
                selected_tetrahedron_ids.insert(child_tetrahedron.tetrahedron_id);
            }
            split_source_tetrahedra.push(child_tetrahedron);
            next_tetrahedron_id = next_tetrahedron_id.saturating_add(1);
        }
    }

    if incident_count == 0 {
        return Err(
            ConstrainedCavitySourceEdgeSplitError::NoIncidentSourceTetrahedron {
                node_ids: target_edge,
            },
        );
    }

    let selected_indices = split_source_tetrahedra
        .iter()
        .enumerate()
        .filter_map(|(index, tetrahedron)| {
            selected_tetrahedron_ids
                .contains(&tetrahedron.tetrahedron_id)
                .then_some(index)
        })
        .collect::<BTreeSet<_>>();
    let split_cavity = build_constrained_cavity_from_index_set(
        &split_source_tetrahedra,
        &selected_indices,
        cavity.protected_node_ids.clone(),
    );
    validate_constrained_cavity(&split_cavity)
        .map_err(ConstrainedCavitySourceEdgeSplitError::Validation)?;

    Ok(ConstrainedCavitySourceEdgeSplit {
        cavity: split_cavity,
        split_node,
        source_tetrahedra: split_source_tetrahedra,
    })
}
