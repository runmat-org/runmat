use super::*;

pub(in super::super::super) fn split_child_boundary_face(
    parent: &ConstrainedCavityBoundaryFace,
    node_ids: [u32; 3],
    perimeter_source_edges: &BTreeMap<[u32; 2], Option<u32>>,
) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: parent.outside_tetrahedron_ids.clone(),
        source_face_id: parent.source_face_id,
        source_edge_ids: face_edges(node_ids).map(|edge| {
            perimeter_source_edges
                .get(&sorted_edge(edge))
                .copied()
                .flatten()
        }),
        region_ids: parent.region_ids.clone(),
    }
}

pub(in super::super::super) fn three_edge_split_child_boundary_face(
    parent: &ConstrainedCavityBoundaryFace,
    node_ids: [u32; 3],
    perimeter_source_edges: &BTreeMap<[u32; 2], Option<u32>>,
    edge_split_node_ids: &BTreeMap<[u32; 2], u32>,
) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: parent.outside_tetrahedron_ids.clone(),
        source_face_id: parent.source_face_id,
        source_edge_ids: face_edges(node_ids).map(|edge| {
            let original_edge =
                edge_split_node_ids
                    .iter()
                    .find_map(|(split_edge, split_node_id)| {
                        if edge.contains(split_node_id)
                            && edge.into_iter().any(|node_id| {
                                node_id != *split_node_id && split_edge.contains(&node_id)
                            })
                        {
                            Some(*split_edge)
                        } else {
                            None
                        }
                    })?;
            perimeter_source_edges
                .get(&original_edge)
                .copied()
                .flatten()
        }),
        region_ids: parent.region_ids.clone(),
    }
}

pub(in super::super::super) fn edge_split_child_boundary_face(
    parent: &ConstrainedCavityBoundaryFace,
    node_ids: [u32; 3],
    split_node_id: u32,
    split_edge: [u32; 2],
    split_edge_source_id: Option<u32>,
    perimeter_source_edges: &BTreeMap<[u32; 2], Option<u32>>,
) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: parent.outside_tetrahedron_ids.clone(),
        source_face_id: parent.source_face_id,
        source_edge_ids: face_edges(node_ids).map(|edge| {
            let sorted = sorted_edge(edge);
            if edge.contains(&split_node_id)
                && edge
                    .into_iter()
                    .any(|node_id| node_id != split_node_id && split_edge.contains(&node_id))
            {
                split_edge_source_id
            } else {
                perimeter_source_edges.get(&sorted).copied().flatten()
            }
        }),
        region_ids: parent.region_ids.clone(),
    }
}
