use std::collections::BTreeMap;

use super::super::super::{geometry::sorted_node_pair, SurfaceElement};

pub(in crate::param_tri::tests) fn assert_local_surface_edges_are_recovered(
    elements: &[SurfaceElement],
) {
    assert_surface_edges_are_recovered(elements, &[[0, 1], [0, 2], [1, 2]]);
}

pub(in crate::param_tri::tests) fn assert_surface_edges_are_recovered(
    elements: &[SurfaceElement],
    boundary_edges: &[[u32; 2]],
) {
    let mut counts = BTreeMap::<[u32; 2], usize>::new();
    for element in elements {
        for edge in [
            sorted_node_pair(element.node_ids[0], element.node_ids[1]),
            sorted_node_pair(element.node_ids[1], element.node_ids[2]),
            sorted_node_pair(element.node_ids[2], element.node_ids[0]),
        ] {
            *counts.entry(edge).or_default() += 1;
        }
    }
    for (edge, count) in counts {
        let is_boundary = boundary_edges.contains(&edge);
        assert_eq!(
            count,
            if is_boundary { 1 } else { 2 },
            "unexpected local surface edge count for {edge:?}"
        );
    }
}
