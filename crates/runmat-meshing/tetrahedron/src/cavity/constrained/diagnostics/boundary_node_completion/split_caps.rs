use super::*;

pub(super) fn diagnostic_split_cap_min_scaled_jacobian(
    face: [u32; 3],
    cap_node_id: u32,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<(f64, &'static str)> {
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    boundary_face_split_node_candidates(face, boundary_nodes)
        .into_iter()
        .filter_map(|split_node| {
            split_completion_tetrahedra_for_node(
                face,
                cap_node_id,
                &split_node,
                boundary_nodes,
                diagnostic_options,
            )
            .map(|tetrahedra| {
                tetrahedra
                    .iter()
                    .map(|tetrahedron| {
                        let points = tetrahedron.node_ids.map(|node_id| {
                            if node_id == split_node.node_id {
                                split_node.coordinates_m
                            } else {
                                boundary_nodes[&node_id]
                            }
                        });
                        (
                            tetrahedron.exact_scaled_jacobian,
                            diagnostic_scaled_jacobian_worst_corner_label(points),
                        )
                    })
                    .min_by(|left, right| left.0.total_cmp(&right.0))
                    .unwrap_or((f64::INFINITY, "face_vertex"))
            })
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
}

pub(super) fn diagnostic_edge_split_cap_min_scaled_jacobian(
    face: [u32; 3],
    cap_node_id: u32,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<(f64, &'static str)> {
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    boundary_face_edge_split_node_candidates(face, boundary_nodes)
        .into_iter()
        .filter_map(|(edge, split_node)| {
            edge_split_completion_tetrahedra_for_node(
                face,
                edge,
                cap_node_id,
                &split_node,
                boundary_nodes,
                diagnostic_options,
            )
            .map(|tetrahedra| {
                tetrahedra
                    .iter()
                    .map(|tetrahedron| {
                        let points = tetrahedron.node_ids.map(|node_id| {
                            if node_id == split_node.node_id {
                                split_node.coordinates_m
                            } else {
                                boundary_nodes[&node_id]
                            }
                        });
                        (
                            tetrahedron.exact_scaled_jacobian,
                            diagnostic_scaled_jacobian_worst_corner_label(points),
                        )
                    })
                    .min_by(|left, right| left.0.total_cmp(&right.0))
                    .unwrap_or((f64::INFINITY, "face_vertex"))
            })
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
}

pub(super) fn diagnostic_three_edge_split_cap_min_scaled_jacobian(
    face: [u32; 3],
    cap_node_id: u32,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<(f64, &'static str)> {
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let split_nodes = boundary_face_mid_edge_split_nodes(face, boundary_nodes);
    let split_node_by_edge = face_edges(face)
        .into_iter()
        .zip(split_nodes.iter())
        .map(|(edge, node)| (sorted_edge(edge), node.node_id))
        .collect::<BTreeMap<_, _>>();
    let split_node_coordinates = split_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    three_edge_split_completion_tetrahedra_for_node(
        face,
        cap_node_id,
        &split_node_by_edge,
        &split_node_coordinates,
        boundary_nodes,
        diagnostic_options,
    )
    .map(|tetrahedra| {
        tetrahedra
            .iter()
            .map(|tetrahedron| {
                let points = tetrahedron.node_ids.map(|node_id| {
                    split_node_coordinates
                        .get(&node_id)
                        .copied()
                        .unwrap_or_else(|| boundary_nodes[&node_id])
                });
                (
                    tetrahedron.exact_scaled_jacobian,
                    diagnostic_scaled_jacobian_worst_corner_label(points),
                )
            })
            .min_by(|left, right| left.0.total_cmp(&right.0))
            .unwrap_or((f64::INFINITY, "face_vertex"))
    })
}
